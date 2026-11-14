"""
Step 1 - Synthetic survey data for validating Zepto's strategy.

Simulates responses to the 26-question customer survey (see QUESTIONNAIRE.md).
Respondents are drawn from four latent customer types so that the data has
real structure for regression, classification, clustering and association
rule mining. Noise, missing values, duplicates, inconsistent labels and
outliers are then injected on purpose, to be removed in clean_data.py.

Anchors taken from public figures (Nov 2026): net value per order of roughly
Rs 390-525 across platforms, a Rs 199 free-delivery minimum, a Rs 30 delivery
fee below it and a Rs 99 per month membership.

Run:  python generate_data.py
"""
import numpy as np
import pandas as pd

SEED = 42
N = 3000
rng = np.random.default_rng(SEED)

SEGMENTS = {
    # name: share, age(mean, sd), income(median, sigma), household(lo, hi),
    #       basket(mean, sd), spend base, speed, price, metro share
    "Convenience professional": dict(p=0.30, age=(30, 5), inc=(120000, 0.35), hh=(1, 3),
                                     basket=(520, 120), base=5200, speed=4.4, price=2.4, metro=0.75),
    "Value family": dict(p=0.28, age=(40, 7), inc=(85000, 0.35), hh=(3, 6),
                         basket=(650, 160), base=3600, speed=3.2, price=4.2, metro=0.50),
    "Student impulse": dict(p=0.22, age=(22, 2.5), inc=(35000, 0.40), hh=(1, 4),
                            basket=(230, 70), base=1500, speed=4.0, price=4.5, metro=0.55),
    "Occasional top-up": dict(p=0.20, age=(48, 10), inc=(70000, 0.40), hh=(2, 5),
                              basket=(380, 110), base=600, speed=2.8, price=3.5, metro=0.35),
}
CATEGORIES = ["Fruits & Vegetables", "Dairy & Bread", "Staples", "Snacks", "Beverages",
              "Instant Food", "Household Cleaning", "Personal Care", "Baby Care",
              "Pet Care", "Electronics Accessories", "Pharma & Wellness"]
MISSIONS = {  # groups of products bought together
    "fresh": ["Fruits & Vegetables", "Dairy & Bread", "Staples"],
    "munchies": ["Snacks", "Beverages", "Instant Food"],
    "home": ["Household Cleaning", "Personal Care", "Baby Care"],
}
MISSION_P = {  # probability a customer type shops each mission in a month
    "Convenience professional": dict(fresh=0.55, munchies=0.60, home=0.35),
    "Value family": dict(fresh=0.85, munchies=0.35, home=0.65),
    "Student impulse": dict(fresh=0.20, munchies=0.85, home=0.15),
    "Occasional top-up": dict(fresh=0.50, munchies=0.25, home=0.25),
}


def likert(mean, n, sd=0.8):
    return np.clip(np.rint(rng.normal(mean, sd, n)), 1, 5).astype(int)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


names = list(SEGMENTS)
seg = rng.choice(names, size=N, p=[SEGMENTS[s]["p"] for s in names])
df = pd.DataFrame({"respondent_id": [f"R{i:05d}" for i in range(1, N + 1)], "_segment": seg})

for col in ["age", "monthly_income", "household_size", "avg_basket_value", "speed_importance",
            "price_sensitivity", "_base", "_metro"]:
    df[col] = 0.0
for s, c in SEGMENTS.items():
    m = df["_segment"] == s
    k = int(m.sum())
    df.loc[m, "age"] = np.clip(np.rint(rng.normal(*c["age"], k)), 18, 70)
    df.loc[m, "monthly_income"] = np.rint(rng.lognormal(np.log(c["inc"][0]), c["inc"][1], k) / 1000) * 1000
    df.loc[m, "household_size"] = rng.integers(c["hh"][0], c["hh"][1] + 1, k)
    df.loc[m, "avg_basket_value"] = np.clip(np.rint(rng.normal(*c["basket"], k) / 10) * 10, 80, 2000)
    df.loc[m, "speed_importance"] = likert(c["speed"], k)
    df.loc[m, "price_sensitivity"] = likert(c["price"], k)
    df.loc[m, "_base"] = c["base"]
    df.loc[m, "_metro"] = (rng.random(k) < c["metro"]).astype(float)

metro = df["_metro"].astype(bool).to_numpy()
df["city_tier"] = np.where(metro, "Metro", rng.choice(["Tier 1", "Tier 2"], N, p=[0.6, 0.4]))
df["gender"] = rng.choice(["Male", "Female", "Other"], N, p=[0.52, 0.46, 0.02])
occ_map = {
    "Convenience professional": (["Salaried", "Self-employed", "Student", "Homemaker", "Retired"], [0.75, 0.2, 0.03, 0.02, 0.0]),
    "Value family": (["Salaried", "Self-employed", "Student", "Homemaker", "Retired"], [0.45, 0.2, 0.0, 0.33, 0.02]),
    "Student impulse": (["Salaried", "Self-employed", "Student", "Homemaker", "Retired"], [0.15, 0.03, 0.82, 0.0, 0.0]),
    "Occasional top-up": (["Salaried", "Self-employed", "Student", "Homemaker", "Retired"], [0.35, 0.2, 0.0, 0.2, 0.25]),
}
df["occupation"] = [rng.choice(occ_map[s][0], p=occ_map[s][1]) for s in df["_segment"]]

# attitudes (1-5 Likert)
df["freshness_satisfaction"] = likert(3.5, N, 0.95)
df["stockout_frequency"] = likert(2.7, N, 1.0)
df["rider_safety_concern"] = likert(3.4, N, 1.0)
df["app_experience"] = likert(3.9, N, 0.8)
overall = 0.9 + 0.38 * df["freshness_satisfaction"] - 0.30 * df["stockout_frequency"] + 0.42 * df["app_experience"] + rng.normal(0, 0.55, N)
df["overall_satisfaction"] = np.clip(np.rint(overall), 1, 5).astype(int)

# Q24 monthly spend (regression target): a linear signal plus noise
spend = (df["_base"] + 0.014 * df["monthly_income"] + 330 * df["household_size"]
         + 360 * (df["speed_importance"] - 3) - 280 * (df["price_sensitivity"] - 3)
         + 220 * (df["overall_satisfaction"] - 3) + 500 * metro + rng.normal(0, 750, N))
spend = np.clip(spend, 150, None)
orders = np.clip(np.rint(spend / df["avg_basket_value"] * rng.normal(1, 0.10, N)), 1, 45)
df["orders_per_month"] = orders.astype(int)
df["monthly_spend"] = np.rint(spend * rng.normal(1, 0.06, N) / 10) * 10  # people report roughly

# which app they use most: happier, faster-minded people lean to Zepto
pz = np.clip(0.24 + 0.05 * (df["overall_satisfaction"] - 3) + 0.03 * (df["speed_importance"] - 3) + 0.06 * (df["_segment"] == "Student impulse"), 0.05, 0.6)
u = rng.random(N)
df["primary_app"] = np.where(u < pz, "Zepto", np.where(u < pz + (1 - pz) * 0.58, "Blinkit", np.where(u < pz + (1 - pz) * 0.90, "Instamart", "Other")))
df["usual_order_time"] = rng.choice(["Morning", "Afternoon", "Evening", "Late night"], N, p=[0.22, 0.18, 0.42, 0.18])

# Q11 categories bought (association rules): missions create co-purchase patterns
baskets = []
for s in df["_segment"]:
    items = set()
    for mission, members in MISSIONS.items():
        if rng.random() < MISSION_P[s][mission]:
            items.update(i for i in members if rng.random() < 0.80)
    items.update(c for c in CATEGORIES if rng.random() < 0.07)
    if s == "Value family" and "Baby Care" in items and rng.random() < 0.3:
        items.add("Pharma & Wellness")
    if not items:
        items.add(rng.choice(CATEGORIES))
    baskets.append("; ".join(sorted(items)))
df["categories_bought"] = baskets

# strategy tests
df["acceptable_wait_min"] = np.clip(np.rint(31 - 4.2 * df["speed_importance"] + 1.8 * df["rider_safety_concern"] + rng.normal(0, 4, N)), 8, 60).astype(int)
fee = 12 + 6 * df["speed_importance"] - 5.5 * df["price_sensitivity"] + df["monthly_income"] / 16000 + rng.normal(0, 7, N)
df["max_small_order_fee"] = (np.clip(np.rint(fee / 5) * 5, 0, 80)).astype(int)

react = []
for b, p, sp in zip(df["avg_basket_value"], df["price_sensitivity"], df["speed_importance"]):
    if b >= 260 and rng.random() < 0.9:
        react.append("Already order above Rs 199")
        continue
    w = np.array([0.42, 0.10 + 0.05 * (sp - 3), 0.14 + 0.07 * (p - 3), 0.14 + 0.03 * (p - 3)])
    w = np.clip(w, 0.02, None)
    react.append(rng.choice(["Add items to reach Rs 199", "Pay the delivery fee", "Switch to another app", "Order less often"], p=w / w.sum()))
df["reaction_to_199_minimum"] = react

pf = sigmoid(-0.4 + 0.65 * (df["price_sensitivity"] - 3))
u = rng.random(N)
df["would_use_fresh_deals"] = np.where(u < pf, "Yes", np.where(u < pf + (1 - pf) * 0.5, "Maybe", "No"))

# Q23 membership intent (classification target)
z = (-3.05 + 0.105 * df["orders_per_month"] + 0.55 * (df["speed_importance"] - 3) - 0.40 * (df["price_sensitivity"] - 3)
     + 0.0000045 * df["monthly_income"] + 0.35 * (df["overall_satisfaction"] - 3) + 0.35 * metro + rng.normal(0, 0.5, N))
df["would_join_membership"] = np.where(rng.random(N) < sigmoid(z), "Yes", "No")

df["nps_score"] = np.clip(np.rint(1.2 + 1.75 * df["overall_satisfaction"] + rng.normal(0, 1.3, N)), 0, 10).astype(int)
zr = -0.3 + 0.75 * (df["overall_satisfaction"] - 3) - 0.35 * (df["stockout_frequency"] - 3) + 0.04 * df["orders_per_month"]
df["will_use_in_6_months"] = np.where(rng.random(N) < sigmoid(zr), "Yes", "No")
df["response_date"] = pd.to_datetime("2026-10-05") + pd.to_timedelta(rng.integers(0, 28, N), unit="D")

int_cols = ["age", "household_size", "speed_importance", "price_sensitivity"]
df[int_cols] = df[int_cols].astype(int)
df = df.drop(columns=["_segment", "_base", "_metro"])

# ---------------------------------------------------------------- mess it up
raw = df.copy()
num_float = ["age", "monthly_income", "household_size", "avg_basket_value", "orders_per_month", "monthly_spend",
             "acceptable_wait_min", "max_small_order_fee", "speed_importance", "price_sensitivity",
             "freshness_satisfaction", "stockout_frequency", "rider_safety_concern", "app_experience",
             "overall_satisfaction", "nps_score"]
raw[num_float] = raw[num_float].astype(float)


def pick(frac):
    return rng.choice(raw.index, size=int(len(raw) * frac), replace=False)


# 1 missing values (people skip questions)
for col, frac in [("monthly_income", 0.06), ("age", 0.02), ("avg_basket_value", 0.03), ("monthly_spend", 0.03),
                  ("freshness_satisfaction", 0.025), ("rider_safety_concern", 0.025), ("max_small_order_fee", 0.03),
                  ("occupation", 0.02), ("usual_order_time", 0.02), ("nps_score", 0.02)]:
    raw.loc[pick(frac), col] = np.nan
# 2 outliers and impossible values (typos, extra zeros, joke answers)
raw.loc[pick(0.008), "monthly_income"] *= 10
raw.loc[pick(0.004), "monthly_income"] = 0
raw.loc[pick(0.006), "age"] = rng.choice([2, 5, 120, 150, 999], int(N * 0.006))
raw.loc[pick(0.007), "avg_basket_value"] *= rng.choice([10, 20], int(N * 0.007))
raw.loc[pick(0.006), "monthly_spend"] *= 12  # gave a yearly figure
raw.loc[pick(0.004), "monthly_spend"] *= -1
raw.loc[pick(0.005), "orders_per_month"] = rng.choice([150, 300, 500], int(N * 0.005))
raw.loc[pick(0.005), "acceptable_wait_min"] = rng.choice([0, 240, 600], int(N * 0.005))
raw.loc[pick(0.004), "max_small_order_fee"] = rng.choice([-10, 500, 999], int(N * 0.004))
raw.loc[pick(0.004), "household_size"] = rng.choice([0, 25, 40], int(N * 0.004))
for col in ["speed_importance", "price_sensitivity", "overall_satisfaction"]:
    raw.loc[pick(0.005), col] = rng.choice([0, 6, 7, 9], int(N * 0.005))
raw.loc[pick(0.004), "nps_score"] = rng.choice([11, 15, 100], int(N * 0.004))
# 3 inconsistent labels (free-text style answers)
variants = {
    "primary_app": {"Zepto": ["zepto", "ZEPTO", " Zepto ", "Zepto app"], "Blinkit": ["blinkit", "Blink it", "BLINKIT"],
                    "Instamart": ["instamart", "Swiggy Instamart", "Insta mart"]},
    "gender": {"Male": ["M", "male", "MALE"], "Female": ["F", "female", "FEMALE"]},
    "city_tier": {"Metro": ["metro", "METRO"], "Tier 1": ["Tier-1", "tier 1", "Tier1"], "Tier 2": ["Tier-2", "tier 2", "Tier2"]},
    "would_join_membership": {"Yes": ["yes", "Y", "YES"], "No": ["no", "N", "NO"]},
}
for col, mp in variants.items():
    for good, bad in mp.items():
        idx = raw.index[raw[col] == good]
        chosen = rng.choice(idx, size=int(len(idx) * 0.08), replace=False)
        raw.loc[chosen, col] = rng.choice(bad, len(chosen))
# 4 duplicate submissions
dups = raw.loc[pick(0.02)].copy()
raw = pd.concat([raw, dups], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
raw["response_date"] = raw["response_date"].dt.strftime("%Y-%m-%d")

raw.to_csv("data/zepto_survey_raw.csv", index=False)
print("raw rows:", len(raw), "| columns:", raw.shape[1])
print("missing cells:", int(raw.isna().sum().sum()), "| duplicate rows:", int(raw.duplicated().sum()))
