"""
Step 2 - Data cleaning and transformation.

Takes the raw survey export and returns an analysis-ready table plus a log of
every change, so each cleaning decision can be defended in the report.

Run:  python clean_data.py
"""
import numpy as np
import pandas as pd

LIKERT = ["speed_importance", "price_sensitivity", "freshness_satisfaction", "stockout_frequency",
          "rider_safety_concern", "app_experience", "overall_satisfaction"]
# values outside these ranges cannot be real answers and are treated as missing
VALID = {"age": (18, 80), "household_size": (1, 12), "orders_per_month": (1, 60),
         "acceptable_wait_min": (5, 90), "max_small_order_fee": (0, 100), "nps_score": (0, 10),
         "monthly_income": (5000, None), "avg_basket_value": (50, None), "monthly_spend": (100, None)}
# long right tails: cap extreme values at the 1.5 x IQR upper fence instead of deleting people
CAP = ["monthly_income", "avg_basket_value", "monthly_spend"]
LABELS = {
    "primary_app": {"zepto": "Zepto", "zepto app": "Zepto", "blinkit": "Blinkit", "blink it": "Blinkit",
                    "instamart": "Instamart", "swiggy instamart": "Instamart", "insta mart": "Instamart", "other": "Other"},
    "gender": {"m": "Male", "male": "Male", "f": "Female", "female": "Female", "other": "Other"},
    "city_tier": {"metro": "Metro", "tier 1": "Tier 1", "tier-1": "Tier 1", "tier1": "Tier 1",
                  "tier 2": "Tier 2", "tier-2": "Tier 2", "tier2": "Tier 2"},
    "would_join_membership": {"yes": "Yes", "y": "Yes", "no": "No", "n": "No"},
}
CATEGORIES = ["Fruits & Vegetables", "Dairy & Bread", "Staples", "Snacks", "Beverages", "Instant Food",
              "Household Cleaning", "Personal Care", "Baby Care", "Pet Care", "Electronics Accessories",
              "Pharma & Wellness"]


def cat_col(name):
    return "cat_" + name.lower().replace(" & ", "_").replace(" ", "_")


def clean(raw):
    df = raw.copy()
    log = []

    def note(step, detail, n):
        log.append({"step": step, "what_was_done": detail, "cells_or_rows_affected": int(n)})

    # 1 duplicates
    n0 = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="respondent_id").reset_index(drop=True)
    note("Remove duplicates", "Dropped repeated submissions of the same respondent", n0 - len(df))

    # 2 standardise labels
    for col, mp in LABELS.items():
        before = df[col].copy()
        key = df[col].astype("string").str.strip().str.lower()
        df[col] = key.map(mp).astype(object)
        note("Standardise labels", f"{col}: merged spelling and case variants into {sorted(set(mp.values()))}",
             (before.astype("string") != df[col].astype("string")).sum())

    # 3 impossible values -> missing
    for col in LIKERT:
        bad = df[col].notna() & ~df[col].between(1, 5)
        df.loc[bad, col] = np.nan
        if bad.sum():
            note("Range check", f"{col}: answers outside the 1-5 scale set to missing", bad.sum())
    for col, (lo, hi) in VALID.items():
        bad = df[col].notna() & ((df[col] < lo) | ((df[col] > hi) if hi is not None else False))
        df.loc[bad, col] = np.nan
        if bad.sum():
            rng_txt = f"below {lo}" if hi is None else f"outside {lo}-{hi}"
            note("Range check", f"{col}: values {rng_txt} set to missing", bad.sum())

    # 4 cap outliers (winsorise at the upper IQR fence)
    for col in CAP:
        q1, q3 = df[col].quantile([0.25, 0.75])
        fence = q3 + 1.5 * (q3 - q1)
        high = df[col] > fence
        df.loc[high, col] = fence
        note("Cap outliers", f"{col}: values above the upper fence ({fence:,.0f}) capped to it", high.sum())

    # 5 impute what is still missing
    num_cols = LIKERT + list(VALID)
    for col in num_cols:
        miss = df[col].isna()
        if miss.sum():
            # median within the same city tier keeps local differences
            df[col] = df[col].fillna(df.groupby("city_tier")[col].transform("median")).fillna(df[col].median())
            note("Impute", f"{col}: missing filled with the median of the respondent's city tier", miss.sum())
    for col in ["occupation", "usual_order_time", "gender", "primary_app", "city_tier", "would_join_membership"]:
        miss = df[col].isna()
        if miss.sum():
            df[col] = df[col].fillna(df[col].mode()[0])
            note("Impute", f"{col}: missing filled with the most common answer", miss.sum())

    # 6 types
    ints = LIKERT + ["age", "household_size", "orders_per_month", "acceptable_wait_min", "max_small_order_fee", "nps_score"]
    df[ints] = df[ints].round().astype(int)
    df["response_date"] = pd.to_datetime(df["response_date"])

    # 7 transformation / feature engineering
    df["log_income"] = np.log(df["monthly_income"])
    df["spend_share_of_income_pct"] = (df["monthly_spend"] / df["monthly_income"] * 100).round(2)
    df["is_metro"] = (df["city_tier"] == "Metro").astype(int)
    df["membership_yes"] = (df["would_join_membership"] == "Yes").astype(int)
    df["retained_yes"] = (df["will_use_in_6_months"] == "Yes").astype(int)
    df["nps_group"] = pd.cut(df["nps_score"], [-1, 6, 8, 10], labels=["Detractor", "Passive", "Promoter"]).astype(str)
    df["age_band"] = pd.cut(df["age"], [17, 24, 34, 44, 54, 80], labels=["18-24", "25-34", "35-44", "45-54", "55+"]).astype(str)
    items = df["categories_bought"].fillna("").str.split("; ")
    for c in CATEGORIES:
        df[cat_col(c)] = items.apply(lambda lst, c=c: int(c in lst))
    df["n_categories"] = df[[cat_col(c) for c in CATEGORIES]].sum(axis=1)
    note("Transform", "Added log income, spend share of income, metro flag, 0/1 targets, NPS group, age band and one 0/1 column per product category", len(df))

    return df, pd.DataFrame(log)


if __name__ == "__main__":
    raw = pd.read_csv("data/zepto_survey_raw.csv")
    clean_df, log = clean(raw)
    clean_df.to_csv("data/zepto_survey_clean.csv", index=False)
    log.to_csv("data/cleaning_log.csv", index=False)
    print(log.to_string(index=False))
    print("clean rows:", len(clean_df), "| missing cells left:", int(clean_df.isna().sum().sum()))
