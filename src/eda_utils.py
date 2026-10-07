"""Helper functions for the per-person data exploration notebooks."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

FEATURE_GROUPS = {
    "account_billing": ["MonthlyRevenue", "TotalRecurringCharge", "OverageMinutes", "MonthsInService",
                        "CurrentEquipmentDays", "Handsets", "HandsetModels", "UniqueSubs", "ActiveSubs",
                        "AdjustmentsToCreditRating"],
    "usage": ["MonthlyMinutes", "DirectorAssistedCalls", "RoamingCalls", "ThreewayCalls", "ReceivedCalls",
              "OutboundCalls", "InboundCalls", "PeakCallsInOut", "OffPeakCallsInOut", "CallForwardingCalls",
              "CallWaitingCalls"],
    "usage_change_retention": ["PercChangeMinutes", "PercChangeRevenues", "RetentionCalls",
                               "RetentionOffersAccepted", "MadeCallToRetentionTeam",
                               "ReferralsMadeBySubscriber", "NewCellphoneUser", "NotNewCellphoneUser"],
    "service_quality": ["DroppedCalls", "BlockedCalls", "UnansweredCalls", "DroppedBlockedCalls",
                        "CustomerCareCalls"],
    "demographics_region": ["AgeHH1", "AgeHH2", "ChildrenInHH", "IncomeGroup", "Occupation", "MaritalStatus",
                            "Homeownership", "CreditRating", "PrizmCode", "ServiceArea"],
    "handset_lifestyle": ["HandsetPrice", "HandsetRefurbished", "HandsetWebCapable", "TruckOwner", "RVOwner",
                          "OwnsMotorcycle", "OwnsComputer", "HasCreditCard", "BuysViaMailOrder",
                          "RespondsToMailOffers", "OptOutMailings", "NonUSTravel"],
}

# Values that most likely mean "unknown" rather than a real value
PLACEHOLDERS = {"AgeHH1": 0, "AgeHH2": 0, "IncomeGroup": 0, "HandsetPrice": "Unknown",
                "MaritalStatus": "Unknown", "Homeownership": "Unknown"}


def is_numeric(s):
    return pd.api.types.is_numeric_dtype(s)


def overview(df, cols):
    """One row per column: type, missing, placeholder share, zeros, range, cardinality."""
    rows = []
    for c in cols:
        s = df[c]
        num = is_numeric(s)
        rows.append({
            "column": c,
            "type": "numeric" if num else "categorical",
            "n_unique": s.nunique(),
            "missing_%": round(100 * s.isna().mean(), 2),
            "placeholder_%": round(100 * (s == PLACEHOLDERS[c]).mean(), 2) if c in PLACEHOLDERS else None,
            "zero_%": round(100 * (s == 0).mean(), 2) if num else None,
            "negative_%": round(100 * (s < 0).mean(), 2) if num else None,
            "min": s.min() if num else None,
            "median": s.median() if num else None,
            "p99": s.quantile(0.99) if num else None,
            "max": s.max() if num else None,
            "top_value": s.mode().iloc[0] if not num else None,
        })
    return pd.DataFrame(rows).set_index("column")


def churn_signal(df, cols, target="churn"):
    """Univariate ROC-AUC for numeric columns; churn-rate spread across categories for the rest."""
    rows = []
    for c in cols:
        s = df[c]
        if is_numeric(s):
            m = s.notna()
            auc = roc_auc_score(df.loc[m, target], s[m])
            rows.append({"column": c, "auc": round(auc, 3), "signal": round(abs(auc - 0.5), 3)})
        else:
            rate = df.groupby(c)[target].mean()
            rows.append({"column": c, "churn_rate_min": round(rate.min(), 3),
                         "churn_rate_max": round(rate.max(), 3)})
    return pd.DataFrame(rows).set_index("column")


def churn_rate_by(df, col, bins=10, target="churn"):
    """Churn rate and size per category (categorical) or per quantile bin (numeric)."""
    key = pd.qcut(df[col], bins, duplicates="drop") if is_numeric(df[col]) and df[col].nunique() > bins else df[col]
    g = df.groupby(key, observed=True)[target]
    return pd.DataFrame({"n": g.size(), "churn_rate": g.mean().round(3)})


def plot_column(df, col, target="churn", bins=10):
    """Left: distribution. Right: churn rate per bin/category, dashed line = overall churn rate."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    s = df[col]
    if is_numeric(s):
        axes[0].hist(s.dropna().clip(upper=s.quantile(0.99)), bins=40)
        axes[0].set_title(f"{col} (clipped at p99)")
    else:
        s.value_counts().head(20).plot.bar(ax=axes[0])
        axes[0].set_title(col)
    rate = churn_rate_by(df, col, bins, target)
    axes[1].bar(range(len(rate)), rate["churn_rate"])
    axes[1].set_xticks(range(len(rate)), [str(i) for i in rate.index], rotation=45, ha="right", fontsize=7)
    axes[1].axhline(df[target].mean(), ls="--", c="grey")
    axes[1].set_title(f"churn rate by {col}")
    plt.tight_layout()
    plt.show()


def corr_heatmap(df, cols):
    """Spearman correlation between the numeric columns of a group."""
    num = [c for c in cols if is_numeric(df[c])]
    corr = df[num].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.6 * len(num) + 3, 0.6 * len(num) + 2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(num)), num, rotation=90)
    ax.set_yticks(range(len(num)), num)
    for i in range(len(num)):
        for j in range(len(num)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()
    return corr


def summary_skeleton():
    """Empty decision table with one row per feature; fill it in at the end of the notebook."""
    rows = [{"group": g, "column": c} for g, cols in FEATURE_GROUPS.items() for c in cols]
    table = pd.DataFrame(rows)
    for field in ["issues", "relation_to_churn", "keep_drop", "missing_handling", "transformation",
                  "encoding", "feature_ideas"]:
        table[field] = ""
    return table.set_index("column")
