"""
Step 3 - The four algorithms, each tied to a block of survey questions.

regression      -> what drives monthly spend (Q24)
classification  -> who would join a paid membership (Q23)
clustering      -> which customer types exist
association     -> which product categories are bought together (Q11)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

SEED = 42
REG_FEATURES = ["monthly_income", "household_size", "age", "speed_importance", "price_sensitivity",
                "overall_satisfaction", "is_metro"]
CLF_FEATURES = ["orders_per_month", "avg_basket_value", "monthly_income", "age", "household_size",
                "speed_importance", "price_sensitivity", "overall_satisfaction", "stockout_frequency", "is_metro"]
CLUSTER_FEATURES = ["age", "monthly_income", "household_size", "orders_per_month", "avg_basket_value",
                    "monthly_spend", "speed_importance", "price_sensitivity"]


def regression(df):
    X, y = df[REG_FEATURES], df["monthly_spend"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    sc = StandardScaler().fit(Xtr)
    m = LinearRegression().fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    raw = LinearRegression().fit(Xtr, ytr)
    coef = pd.DataFrame({"feature": REG_FEATURES, "rupees_per_unit": raw.coef_,
                         "rupees_per_std_dev": m.coef_}).sort_values("rupees_per_std_dev", key=abs, ascending=False)
    return {"r2": r2_score(yte, pred), "mae": mean_absolute_error(yte, pred), "coef": coef,
            "actual": yte.to_numpy(), "pred": pred, "mean_spend": float(y.mean())}


def classification(df):
    X, y = df[CLF_FEATURES], df["membership_yes"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    logit = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED).fit(Xtr, ytr)
    out = {"base_rate": float(y.mean()), "models": {}}
    for name, proba in [("Logistic regression", logit.predict_proba(sc.transform(Xte))[:, 1]),
                        ("Random forest", forest.predict_proba(Xte)[:, 1])]:
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(yte, proba)
        out["models"][name] = {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
                               "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
                               "auc": roc_auc_score(yte, proba), "cm": confusion_matrix(yte, pred),
                               "fpr": fpr, "tpr": tpr}
    out["logit_coef"] = pd.DataFrame({"feature": CLF_FEATURES, "effect": logit.coef_[0]}).sort_values("effect")
    out["forest_importance"] = pd.DataFrame({"feature": CLF_FEATURES, "importance": forest.feature_importances_}).sort_values("importance")
    return out


def clustering(df, k=4):
    Z = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    rows = []
    for kk in range(2, 9):
        km = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit(Z)
        rows.append({"k": kk, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Z, km.labels_, sample_size=1500, random_state=SEED)})
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    labels = pd.Series(km.labels_, index=df.index)
    prof = df.assign(cluster=labels).groupby("cluster").agg(
        respondents=("respondent_id", "size"), age=("age", "mean"), monthly_income=("monthly_income", "mean"),
        household_size=("household_size", "mean"), orders_per_month=("orders_per_month", "mean"),
        avg_basket_value=("avg_basket_value", "mean"), monthly_spend=("monthly_spend", "mean"),
        speed_importance=("speed_importance", "mean"), price_sensitivity=("price_sensitivity", "mean"),
        membership_rate=("membership_yes", "mean"), retained_rate=("retained_yes", "mean"))
    prof["share_of_respondents"] = prof["respondents"] / len(df)
    prof["share_of_spend"] = prof["respondents"] * prof["monthly_spend"] / df["monthly_spend"].sum()
    prof["name"] = name_clusters(prof)
    return {"elbow": pd.DataFrame(rows), "labels": labels.map(prof["name"]), "profile": prof.reset_index()}


def name_clusters(prof):
    """Plain-language labels from how the clusters rank against each other."""
    left = list(prof.index)
    names = {}

    def take(col, lowest, label):
        if not left:
            return
        pick = prof.loc[left, col].idxmin() if lowest else prof.loc[left, col].idxmax()
        names[pick] = label
        left.remove(pick)

    take("orders_per_month", True, "Occasional top-up shoppers")
    take("age", True, "Young small-basket snackers")
    take("household_size", False, "Value-seeking families")
    take("monthly_spend", False, "Convenience-first professionals")
    for i, c in enumerate(left, 1):
        names[c] = f"Other segment {i}"
    return prof.index.map(names)


def association(df, min_support=0.08, min_lift=1.1):
    cols = [c for c in df.columns if c.startswith("cat_")]
    B = df[cols].astype(bool)
    B.columns = [c[4:].replace("_", " ").title() for c in cols]
    freq = apriori(B, min_support=min_support, use_colnames=True, max_len=2)
    rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    rules["if_bought"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    rules["then_also"] = rules["consequents"].apply(lambda s: " + ".join(sorted(s)))
    rules = rules[["if_bought", "then_also", "support", "confidence", "lift"]].sort_values("lift", ascending=False).reset_index(drop=True)
    return {"rules": rules, "item_share": B.mean().sort_values(), "co": B.astype(int).T.dot(B.astype(int)) / len(B)}
