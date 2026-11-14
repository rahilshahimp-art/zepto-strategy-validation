"""
Zepto strategy validation dashboard (Streamlit).

Run locally:  streamlit run app.py
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis
from clean_data import clean

st.set_page_config(page_title="Zepto strategy validation", page_icon="🛒", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "zepto_survey_raw.csv")


@st.cache_data(show_spinner="Preparing survey data...")
def load():
    if not os.path.exists(RAW):  # first run on a fresh deployment: rebuild the data from the seed
        os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
        subprocess.run([sys.executable, "generate_data.py"], cwd=HERE, check=True)
    raw = pd.read_csv(RAW)
    df, log = clean(raw)
    return raw, df, log


@st.cache_data(show_spinner="Running the four algorithms...")
def models(df):
    return analysis.regression(df), analysis.classification(df), analysis.clustering(df), analysis.association(df)


def show(fig, explain):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**What this shows.** {explain}")


def rupees(x):
    return f"₹{x:,.0f}"


def pct(x):
    return f"{x * 100:.0f}%"


raw, full, log = load()
reg, clf, clu, aso = models(full)
full = full.assign(segment=clu["labels"])

st.title("Is Zepto's strategy sustainable? A survey-based validation")
st.caption("Synthetic survey of quick-commerce customers in India, simulated to test the decisions of the "
           "leadership review of 9 November 2026. The data is simulated, so the findings show how the "
           "validation works and what to look for; they are not evidence about real customers.")

with st.sidebar:
    st.header("Filters")
    st.caption("Filters change the descriptive and diagnostic tabs. The four models always use all respondents.")
    tiers = st.multiselect("City type", sorted(full["city_tier"].unique()), default=sorted(full["city_tier"].unique()))
    apps = st.multiselect("Main app", sorted(full["primary_app"].unique()), default=sorted(full["primary_app"].unique()))
    ages = st.slider("Age", int(full["age"].min()), int(full["age"].max()), (int(full["age"].min()), int(full["age"].max())))
df = full[full["city_tier"].isin(tiers) & full["primary_app"].isin(apps) & full["age"].between(*ages)]
if len(df) < 50:
    st.warning("Fewer than 50 respondents match these filters. Widen them to see the charts.")
    st.stop()

tabs = st.tabs(["Verdict", "Questionnaire", "Data cleaning", "Descriptive", "Diagnostic", "Regression",
                "Classification", "Clustering", "Association rules", "Data"])

# ------------------------------------------------------------------ verdict
with tabs[0]:
    at_risk = full["reaction_to_199_minimum"].isin(["Switch to another app", "Order less often"]).mean()
    below = full["reaction_to_199_minimum"] != "Already order above Rs 199"
    risk_among_below = full.loc[below, "reaction_to_199_minimum"].isin(["Switch to another app", "Order less often"]).mean()
    pay30 = (full["max_small_order_fee"] >= 30).mean()
    member = full["membership_yes"].mean()
    wait15 = (full["acceptable_wait_min"] >= 15).mean()
    fresh = (full["would_use_fresh_deals"] == "Yes").mean()
    retained = full["retained_yes"].mean()
    nps = (full["nps_group"] == "Promoter").mean() - (full["nps_group"] == "Detractor").mean()
    prof = clu["profile"]
    top = prof.sort_values("share_of_spend", ascending=False).iloc[0]

    c = st.columns(4)
    c[0].metric("Respondents after cleaning", f"{len(full):,}")
    c[1].metric("Average monthly spend", rupees(full["monthly_spend"].mean()))
    c[2].metric("Would join ₹99 membership", pct(member))
    c[3].metric("Net promoter score", f"{nps * 100:+.0f}")

    checks = pd.DataFrame([
        ["Riders: remove the timer", f"{pct(wait15)} accept 15 minutes or more; median acceptable wait is {full['acceptable_wait_min'].median():.0f} minutes",
         "Pass" if wait15 > 0.7 else "Watch"],
        ["Inventory: Fresh Deals", f"{pct(fresh)} say yes to evening discounts on near-expiry items", "Pass" if fresh > 0.4 else "Watch"],
        ["Profit: ₹199 free-delivery minimum", f"{pct(at_risk)} of all customers would switch or order less ({pct(risk_among_below)} of those who order below ₹199)",
         "Pass" if at_risk < 0.12 else "Watch"],
        ["Profit: small-order fee", f"Only {pct(pay30)} would pay ₹30 or more; the median ceiling is {rupees(full['max_small_order_fee'].median())}",
         "Pass" if pay30 > 0.5 else "Watch"],
        ["Profit: paid membership", f"{pct(member)} would join; the model finds them with AUC {clf['models']['Logistic regression']['auc']:.2f}",
         "Pass" if member > 0.25 else "Watch"],
        ["Loyalty", f"{pct(retained)} expect to be using the app in six months; NPS is {nps * 100:+.0f}", "Pass" if retained > 0.7 and nps > 0 else "Fail"],
        ["Dependence on one segment", f"{top['name']} are {pct(top['share_of_respondents'])} of customers but {pct(top['share_of_spend'])} of spend",
         "Watch" if top["share_of_spend"] > 0.4 else "Pass"],
    ], columns=["Strategy test", "What the survey says", "Result"])
    passes = (checks["Result"] == "Pass").sum()
    fails = (checks["Result"] == "Fail").sum()
    verdict = "Sustainable" if passes >= 6 and fails == 0 else ("Conditionally sustainable" if passes >= 3 else "Not sustainable as planned")
    st.subheader(f"Verdict: {verdict}")
    st.markdown(
        f"{passes} of {len(checks)} strategy tests pass. The safety and freshness decisions cost little customer goodwill, "
        f"and membership and bigger baskets have real demand. The weak points are loyalty and the small-order fee: "
        f"customers tolerate a higher minimum order far better than a ₹30 fee, and revenue leans heavily on one segment.")
    st.dataframe(checks, hide_index=True, use_container_width=True)
    st.markdown(
        "**How to read the result column.** Pass means the survey supports the decision at the threshold set in the code "
        "(for example, fewer than 12% of customers at risk from the ₹199 minimum). Watch means go ahead with a safeguard. "
        "Fail means the plan needs a fix before it can be called sustainable.")
    st.markdown(
        f"**What to do.** (1) Lead with the ₹199 minimum and membership, not the small-order fee. "
        f"(2) Aim membership at frequent, speed-minded customers, the strongest drivers in the classification model. "
        f"(3) Protect {top['name'].lower()}, who carry {pct(top['share_of_spend'])} of spend. "
        f"(4) Fix stock-outs and freshness first, because they drive satisfaction and satisfaction drives retention.")

# ------------------------------------------------------------ questionnaire
with tabs[1]:
    with open(os.path.join(HERE, "QUESTIONNAIRE.md"), encoding="utf-8") as fh:
        st.markdown(fh.read())

# ----------------------------------------------------------------- cleaning
with tabs[2]:
    st.subheader("From raw export to analysis-ready data")
    c = st.columns(4)
    c[0].metric("Raw rows", f"{len(raw):,}")
    c[1].metric("Duplicate rows removed", f"{len(raw) - len(full):,}")
    c[2].metric("Missing cells in raw data", f"{int(raw.isna().sum().sum()):,}")
    c[3].metric("Missing cells after cleaning", f"{int(full.drop(columns='segment').isna().sum().sum()):,}")
    miss = (raw.isna().mean() * 100).sort_values()
    miss = miss[miss > 0]
    show(px.bar(miss, orientation="h", title="Share of answers missing in the raw data, by question (%)",
                labels={"value": "% missing", "index": ""}).update_layout(showlegend=False),
         f"Income is skipped most often ({miss.max():.1f}%), which is normal for a sensitive question. No question loses more "
         "than about 6%, so filling gaps with the median of the respondent's city type is safer than deleting people.")
    col = st.selectbox("Compare a numeric question before and after cleaning",
                       ["monthly_income", "monthly_spend", "avg_basket_value", "age", "orders_per_month", "acceptable_wait_min"])
    both = pd.concat([raw[[col]].assign(stage="Raw"), full[[col]].assign(stage="Cleaned")])
    show(px.box(both, x="stage", y=col, color="stage", title=f"{col}: raw against cleaned", log_y=True).update_layout(showlegend=False),
         f"The raw box has points far from the rest (maximum {raw[col].max():,.0f}), caused by typing errors such as an extra "
         f"zero or a yearly figure. After cleaning the maximum is {full[col].max():,.0f}. Impossible values were set to missing "
         "and then imputed; genuine but extreme values were capped at the upper fence (third quartile plus 1.5 times the "
         "interquartile range) so that a few people do not dominate averages and models. The axis is logarithmic so both fit.")
    labels = pd.DataFrame({"Raw spellings": [raw[c].nunique() for c in ["primary_app", "gender", "city_tier", "would_join_membership"]],
                           "After cleaning": [full[c].nunique() for c in ["primary_app", "gender", "city_tier", "would_join_membership"]]},
                          index=["Main app", "Gender", "City type", "Membership answer"])
    st.dataframe(labels, use_container_width=True)
    st.markdown("**What this shows.** Free-text style answers such as 'zepto', 'ZEPTO' and 'Zepto app' were merged into one label "
                "each. Without this step every spelling would be counted as a different app.")
    st.subheader("Cleaning log")
    st.dataframe(log, hide_index=True, use_container_width=True)

# -------------------------------------------------------------- descriptive
with tabs[3]:
    st.subheader(f"Who answered and how they shop ({len(df):,} respondents in view)")
    c = st.columns(4)
    c[0].metric("Median age", f"{df['age'].median():.0f}")
    c[1].metric("Median monthly income", rupees(df["monthly_income"].median()))
    c[2].metric("Orders per month", f"{df['orders_per_month'].mean():.1f}")
    c[3].metric("Average order value", rupees(df["avg_basket_value"].mean()))
    a, b = st.columns(2)
    with a:
        show(px.histogram(df, x="monthly_spend", nbins=40, title="Monthly spend on quick commerce (₹)"),
             f"Half of customers spend less than {rupees(df['monthly_spend'].median())} a month and the average is "
             f"{rupees(df['monthly_spend'].mean())}. The top 20% of customers account for "
             f"{pct(df['monthly_spend'].nlargest(int(len(df) * 0.2)).sum() / df['monthly_spend'].sum())} of all spend, so losing a few of them hurts.")
    with b:
        share = df["primary_app"].value_counts(normalize=True).mul(100).round(1).reset_index()
        share.columns = ["app", "share"]
        show(px.bar(share, x="app", y="share", text="share", title="Main app used (% of respondents)"),
             f"{share.iloc[0]['app']} is the most-used app at {share.iloc[0]['share']:.0f}%. Zepto is the main app for "
             f"{(df['primary_app'] == 'Zepto').mean() * 100:.0f}%, so most of its potential customers currently default to a rival.")
    a, b = st.columns(2)
    with a:
        show(px.histogram(df, x="age_band", color="city_tier", barmode="group", category_orders={"age_band": ["18-24", "25-34", "35-44", "45-54", "55+"]},
                          title="Respondents by age band and city type"),
             f"The largest group is {df['age_band'].mode()[0]}. Metro respondents are {pct((df['city_tier'] == 'Metro').mean())} of the sample, "
             "which matches where dark stores are densest.")
    with b:
        order = ["Already order above Rs 199", "Add items to reach Rs 199", "Pay the delivery fee", "Order less often", "Switch to another app"]
        r = df["reaction_to_199_minimum"].value_counts(normalize=True).reindex(order).mul(100).round(1).reset_index()
        r.columns = ["reaction", "share"]
        show(px.bar(r, x="share", y="reaction", orientation="h", text="share", title="Reaction to a ₹199 free-delivery minimum (%)"),
             f"{r.iloc[0]['share']:.0f}% already order above ₹199 and are unaffected. Of the rest, more would add items than leave, "
             f"but {r.iloc[3]['share'] + r.iloc[4]['share']:.0f}% of all customers say they would order less or switch. "
             "That is the price of the higher minimum.")
    a, b = st.columns(2)
    with a:
        show(px.histogram(df, x="max_small_order_fee", nbins=14, title="Highest fee accepted on a small order (₹)").add_vline(x=30, line_dash="dash", annotation_text="₹30 fee"),
             f"Only {pct((df['max_small_order_fee'] >= 30).mean())} would accept the ₹30 fee marked by the dashed line; the median ceiling is "
             f"{rupees(df['max_small_order_fee'].median())}. A flat ₹30 small-order fee would be refused by most customers.")
    with b:
        show(px.histogram(df, x="acceptable_wait_min", nbins=20, title="Delivery time customers would accept (minutes)").add_vline(x=10, line_dash="dash", annotation_text="10 minutes"),
             f"The median acceptable wait is {df['acceptable_wait_min'].median():.0f} minutes and {pct((df['acceptable_wait_min'] > 10).mean())} "
             "accept more than 10. Removing the rider timer therefore carries little demand risk.")
    lk = ["speed_importance", "price_sensitivity", "freshness_satisfaction", "stockout_frequency", "rider_safety_concern", "app_experience", "overall_satisfaction"]
    m = df[lk].mean().sort_values().round(2)
    show(px.bar(m, orientation="h", range_x=[1, 5], text=m, title="Average score on each attitude question (1 to 5)", labels={"value": "average", "index": ""}).update_layout(showlegend=False),
         f"The highest average is {m.index[-1].replace('_', ' ')} ({m.iloc[-1]}) and the lowest is {m.index[0].replace('_', ' ')} ({m.iloc[0]}). "
         "Stock-out frequency is the one item where a lower score is better.")

# --------------------------------------------------------------- diagnostic
with tabs[4]:
    st.subheader("Why the numbers look the way they do")
    num = ["monthly_spend", "orders_per_month", "avg_basket_value", "monthly_income", "age", "household_size", "speed_importance",
           "price_sensitivity", "freshness_satisfaction", "stockout_frequency", "app_experience", "overall_satisfaction",
           "rider_safety_concern", "acceptable_wait_min", "max_small_order_fee", "nps_score", "membership_yes", "retained_yes"]
    corr = df[num].corr().round(2)
    show(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Correlation matrix (Pearson)").update_layout(height=650),
         "Each cell is the correlation between two answers, from -1 (move in opposite directions) through 0 (unrelated) to +1 "
         f"(move together). Monthly spend rises with income ({corr.loc['monthly_spend', 'monthly_income']:+.2f}) and with how much speed matters "
         f"({corr.loc['monthly_spend', 'speed_importance']:+.2f}), and falls with price sensitivity ({corr.loc['monthly_spend', 'price_sensitivity']:+.2f}). "
         f"Overall satisfaction tracks app experience ({corr.loc['overall_satisfaction', 'app_experience']:+.2f}) and freshness "
         f"({corr.loc['overall_satisfaction', 'freshness_satisfaction']:+.2f}) and falls with stock-outs ({corr.loc['overall_satisfaction', 'stockout_frequency']:+.2f}). "
         "Correlation shows association, not cause.")
    target = st.selectbox("Rank what moves with", ["monthly_spend", "membership_yes", "retained_yes", "overall_satisfaction", "max_small_order_fee", "acceptable_wait_min"])
    r = corr[target].drop(target).sort_values()
    show(px.bar(r, orientation="h", title=f"Correlation of every question with {target}", labels={"value": "correlation", "index": ""}, color=r, color_continuous_scale="RdBu_r", range_color=[-1, 1]).update_layout(coloraxis_showscale=False),
         f"The strongest positive link with {target.replace('_', ' ')} is {r.index[-1].replace('_', ' ')} ({r.iloc[-1]:+.2f}); the strongest negative is "
         f"{r.index[0].replace('_', ' ')} ({r.iloc[0]:+.2f}). Bars near zero are answers that tell us little about it.")
    a, b = st.columns(2)
    with a:
        s = df.sample(min(1200, len(df)), random_state=1)
        show(px.scatter(s, x="monthly_income", y="monthly_spend", color="segment", opacity=0.6, title="Income against monthly spend, by customer type"),
             f"Spend climbs with income (correlation {corr.loc['monthly_spend', 'monthly_income']:+.2f}), but the colours show the link is partly "
             "about customer type: professionals sit top right, young snackers bottom left. Income alone is not the full explanation.")
    with b:
        g = df.groupby("stockout_frequency")["retained_yes"].mean().mul(100).round(1).reset_index()
        show(px.line(g, x="stockout_frequency", y="retained_yes", markers=True, title="Customers expecting to stay, by how often they hit stock-outs (%)", labels={"retained_yes": "% staying", "stockout_frequency": "stock-out frequency (1-5)"}),
             f"Retention falls from {g['retained_yes'].iloc[0]:.0f}% among customers who rarely see stock-outs to {g['retained_yes'].iloc[-1]:.0f}% "
             "among those who see them most. This is the customer case for the forecasting pilot in the minutes.")
    a, b = st.columns(2)
    with a:
        g = df.groupby("overall_satisfaction")[["retained_yes", "membership_yes"]].mean().mul(100).round(1).reset_index().melt("overall_satisfaction", var_name="outcome", value_name="share")
        show(px.line(g, x="overall_satisfaction", y="share", color="outcome", markers=True, title="Retention and membership intent by overall satisfaction (%)"),
             f"Between the least and most satisfied customers, retention moves from {g[g['outcome'] == 'retained_yes']['share'].iloc[0]:.0f}% to "
             f"{g[g['outcome'] == 'retained_yes']['share'].iloc[-1]:.0f}% and membership intent from {g[g['outcome'] == 'membership_yes']['share'].iloc[0]:.0f}% to "
             f"{g[g['outcome'] == 'membership_yes']['share'].iloc[-1]:.0f}%. Satisfaction mainly protects retention; membership depends more on how often people order.")
    with b:
        g = df.groupby("price_sensitivity")["reaction_to_199_minimum"].apply(lambda s: s.isin(["Switch to another app", "Order less often"]).mean() * 100).round(1).reset_index(name="at_risk")
        show(px.bar(g, x="price_sensitivity", y="at_risk", text="at_risk", title="Customers who would switch or order less at a ₹199 minimum, by price sensitivity (%)"),
             f"The share at risk grows from {g['at_risk'].iloc[0]:.0f}% among the least price-sensitive to {g['at_risk'].iloc[-1]:.0f}% among the most. "
             "The higher minimum should be paired with targeted offers for price-sensitive customers, as the minutes propose.")
    g = df.groupby("segment").agg(spend=("monthly_spend", "mean"), membership=("membership_yes", "mean"), retained=("retained_yes", "mean"), fee=("max_small_order_fee", "mean")).round(2).reset_index()
    show(px.bar(g, x="segment", y="spend", text=g["spend"].round(0), title="Average monthly spend by customer type (₹)"),
         f"{g.sort_values('spend').iloc[-1]['segment']} spend the most ({rupees(g['spend'].max())} a month) and {g.sort_values('spend').iloc[0]['segment'].lower()} "
         f"the least ({rupees(g['spend'].min())}). The gap explains why one fee policy cannot suit everyone.")

# --------------------------------------------------------------- regression
with tabs[5]:
    st.subheader("Regression: what drives monthly spend?")
    st.markdown("Linear regression of monthly spend (Q24) on income, household size, age, attitudes and city type. "
                "Order count and basket size are left out on purpose: spend is their product, so including them would explain nothing new. "
                "Trained on 75% of respondents and tested on the other 25%.")
    c = st.columns(3)
    c[0].metric("R² on unseen respondents", f"{reg['r2']:.2f}")
    c[1].metric("Typical error (MAE)", rupees(reg["mae"]))
    c[2].metric("Average monthly spend", rupees(reg["mean_spend"]))
    co = reg["coef"].sort_values("rupees_per_std_dev")
    show(px.bar(co, x="rupees_per_std_dev", y="feature", orientation="h", color="rupees_per_std_dev", color_continuous_scale="RdBu_r", title="Change in monthly spend for a one standard deviation rise in each driver (₹)").update_layout(coloraxis_showscale=False),
         f"Bars to the right raise spend and bars to the left lower it. The strongest driver is {reg['coef'].iloc[0]['feature'].replace('_', ' ')} "
         f"({reg['coef'].iloc[0]['rupees_per_std_dev']:+,.0f} rupees per standard deviation), followed by {reg['coef'].iloc[1]['feature'].replace('_', ' ')} "
         f"({reg['coef'].iloc[1]['rupees_per_std_dev']:+,.0f}). Standardising puts all drivers on the same footing so they can be compared.")
    lim = [0, float(max(reg["actual"].max(), reg["pred"].max()))]
    f = px.scatter(x=reg["actual"], y=reg["pred"], opacity=0.5, labels={"x": "actual spend (₹)", "y": "predicted spend (₹)"}, title="Predicted against actual spend on the test set")
    f.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="perfect prediction", line=dict(dash="dash")))
    show(f, f"Points on the dashed line are predicted exactly. The model explains {reg['r2'] * 100:.0f}% of the variation in spend and is off by "
            f"{rupees(reg['mae'])} on a typical customer. That is good enough to rank customers by value, not to forecast one person's bill.")
    tbl = reg["coef"].rename(columns={"rupees_per_unit": "₹ per one unit", "rupees_per_std_dev": "₹ per std dev"}).round(3)
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    st.markdown("**Business reading.** High-value customers are higher-income, speed-minded, less price-sensitive and in metros. "
                "Discounts aimed at price-sensitive customers buy little spend; reliability and speed for the others buy more.")

# ----------------------------------------------------------- classification
with tabs[6]:
    st.subheader("Classification: who will join a ₹99 membership?")
    st.markdown(f"Target: answer to Q23. {pct(clf['base_rate'])} said yes, so a model that always says 'no' would be right "
                f"{pct(1 - clf['base_rate'])} of the time; that is the bar to beat. Two models are compared on a 25% test set.")
    met = pd.DataFrame({k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1", "auc"]} for k, v in clf["models"].items()}).T.round(3)
    st.dataframe(met, use_container_width=True)
    best = met["auc"].idxmax()
    st.markdown(f"**What this shows.** {best} separates joiners from non-joiners best (AUC {met.loc[best, 'auc']:.2f}; 0.5 is guessing and 1 is perfect). "
                f"Its precision of {met.loc[best, 'precision']:.2f} means that of the customers it flags, {met.loc[best, 'precision'] * 100:.0f}% do say yes; "
                f"its recall of {met.loc[best, 'recall']:.2f} means it finds {met.loc[best, 'recall'] * 100:.0f}% of all the joiners.")
    a, b = st.columns(2)
    with a:
        f = go.Figure()
        for k, v in clf["models"].items():
            f.add_trace(go.Scatter(x=v["fpr"], y=v["tpr"], mode="lines", name=f"{k} (AUC {v['auc']:.2f})"))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="guessing", line=dict(dash="dash")))
        f.update_layout(title="ROC curves", xaxis_title="false positive rate", yaxis_title="true positive rate")
        show(f, "The further a curve bows towards the top left, the better the model tells joiners from non-joiners at every cut-off. "
                "Both curves sit well above the diagonal of pure guessing, and the simpler logistic model does at least as well as the forest.")
    with b:
        cm = clf["models"][best]["cm"]
        show(px.imshow(cm, text_auto=True, x=["predicted no", "predicted yes"], y=["actual no", "actual yes"], color_continuous_scale="Blues", title=f"Confusion matrix: {best}"),
             f"Of {cm.sum()} test customers the model is right on {cm[0, 0] + cm[1, 1]}. It misses {cm[1, 0]} real joiners and wrongly flags {cm[0, 1]}. "
             "Missed joiners are lost revenue; false flags only cost a wasted offer, so the cut-off could be lowered for a marketing campaign.")
    show(px.bar(clf["logit_coef"], x="effect", y="feature", orientation="h", color="effect", color_continuous_scale="RdBu_r", title="What pushes a customer towards membership (logistic regression, standardised)").update_layout(coloraxis_showscale=False),
         f"The strongest push towards joining is {clf['logit_coef'].iloc[-1]['feature'].replace('_', ' ')}, then {clf['logit_coef'].iloc[-2]['feature'].replace('_', ' ')}; "
         f"the strongest push away is {clf['logit_coef'].iloc[0]['feature'].replace('_', ' ')}. Membership should be offered to frequent, speed-minded customers first.")

# --------------------------------------------------------------- clustering
with tabs[7]:
    st.subheader("Clustering: which customer types exist?")
    st.markdown("K-means on eight standardised answers (age, income, household size, orders, basket, spend, speed importance, price sensitivity). "
                "The algorithm is not told the customer types used to simulate the data.")
    el = clu["elbow"]
    a, b = st.columns(2)
    with a:
        show(px.line(el, x="k", y="inertia", markers=True, title="Elbow chart: within-cluster spread by number of clusters"),
             "Spread falls quickly up to 4 clusters and slowly after that. The bend, or elbow, marks where extra clusters stop adding much.")
    with b:
        bk = int(el.loc[el["silhouette"].idxmax(), "k"])
        show(px.line(el, x="k", y="silhouette", markers=True, title="Silhouette score by number of clusters"),
             f"Silhouette measures how cleanly clusters separate (higher is better). It peaks at {bk} and is close at 4 "
             f"({el.loc[el['k'] == 4, 'silhouette'].iloc[0]:.2f} against {el['silhouette'].max():.2f}). Four is used because the fourth group, "
             "occasional shoppers, behaves differently enough to need its own treatment.")
    p = clu["profile"]
    s = full.sample(1500, random_state=2)
    show(px.scatter(s, x="orders_per_month", y="avg_basket_value", color="segment", opacity=0.6, title="The four customer types: order frequency against basket size"),
         "Each dot is a respondent. Young snackers order often but small; families order less often but big; professionals are high on both; "
         "occasional shoppers are low on frequency. Frequency and basket size together separate the groups.")
    pp = p.melt(id_vars="name", value_vars=["share_of_respondents", "share_of_spend"], var_name="measure", value_name="share")
    pp["share"] = (pp["share"] * 100).round(1)
    show(px.bar(pp, x="name", y="share", color="measure", barmode="group", text="share", title="Share of customers against share of spend, by customer type (%)"),
         f"{top['name']} are {pct(top['share_of_respondents'])} of customers but {pct(top['share_of_spend'])} of spend. "
         "Where the spend bar is taller than the customer bar, the segment is worth more than its headcount suggests.")
    view = p[["name", "respondents", "age", "monthly_income", "household_size", "orders_per_month", "avg_basket_value", "monthly_spend",
              "speed_importance", "price_sensitivity", "membership_rate", "retained_rate"]].round(2)
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.markdown("**Business reading.** Membership intent is highest among professionals; small-basket snackers are the ones the ₹199 minimum touches; "
                "families respond to value; occasional shoppers need a reason to order more often.")

# -------------------------------------------------------------- association
with tabs[8]:
    st.subheader("Association rules: what is bought together?")
    st.markdown("Apriori on the categories ticked in Q11. **Support** is the share of customers buying both; **confidence** is the chance of the second "
                "given the first; **lift** above 1 means the pair occurs more often than chance.")
    it = (aso["item_share"] * 100).round(1)
    show(px.bar(it, orientation="h", text=it, title="Customers buying each category in the last month (%)", labels={"value": "%", "index": ""}).update_layout(showlegend=False),
         f"{it.index[-1]} is the most-bought category ({it.iloc[-1]:.0f}%) and {it.index[0]} the least ({it.iloc[0]:.0f}%). "
         "Categories below the 8% support cut-off cannot form rules.")
    rules = aso["rules"]
    pairs = rules[rules["if_bought"] < rules["then_also"]].head(12).copy()
    pairs["pair"] = pairs["if_bought"] + " and " + pairs["then_also"]
    show(px.bar(pairs.sort_values("lift"), x="lift", y="pair", orientation="h", text=pairs.sort_values("lift")["lift"].round(2), title="Strongest product pairs by lift").add_vline(x=1, line_dash="dash"),
         f"The strongest pair is {pairs.iloc[0]['pair']} with lift {pairs.iloc[0]['lift']:.2f}: customers buy them together "
         f"{pairs.iloc[0]['lift']:.1f} times as often as chance would predict. The pairs fall into three shopping missions: home care, fresh food and snacks.")
    show(px.imshow((aso["co"] * 100).round(0), text_auto=True, color_continuous_scale="Blues", aspect="auto", title="Customers buying both categories (%)").update_layout(height=600),
         "The diagonal is each category on its own. Dark blocks off the diagonal are the missions: fresh food with dairy and staples, "
         "snacks with beverages and instant food, and cleaning with personal and baby care.")
    st.dataframe(rules.round(3), hide_index=True, use_container_width=True)
    st.markdown("**Business reading.** These pairs are the 'frequently bought together' prompts the CTO proposed. Suggesting the missing half of a mission "
                "lifts basket size, which is how a customer below ₹199 reaches free delivery without a fee.")

# --------------------------------------------------------------------- data
with tabs[9]:
    st.subheader("Download the data")
    a, b, c = st.columns(3)
    a.download_button("Raw survey data (CSV)", raw.to_csv(index=False), "zepto_survey_raw.csv", "text/csv")
    b.download_button("Cleaned data (CSV)", full.to_csv(index=False), "zepto_survey_clean.csv", "text/csv")
    c.download_button("Cleaning log (CSV)", log.to_csv(index=False), "cleaning_log.csv", "text/csv")
    st.dataframe(df.head(200), use_container_width=True)
