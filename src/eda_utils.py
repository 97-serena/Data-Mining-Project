"""feature groups and helpers for exploring customer churn"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

FEATURE_GROUPS = {
    "account_billing": [
        "MonthlyRevenue",
        "TotalRecurringCharge",
        "OverageMinutes",
        "MonthsInService",
        "CurrentEquipmentDays",
        "Handsets",
        "HandsetModels",
        "UniqueSubs",
        "ActiveSubs",
        "AdjustmentsToCreditRating",
    ],
    "usage": [
        "MonthlyMinutes",
        "DirectorAssistedCalls",
        "RoamingCalls",
        "ThreewayCalls",
        "ReceivedCalls",
        "OutboundCalls",
        "InboundCalls",
        "PeakCallsInOut",
        "OffPeakCallsInOut",
        "CallForwardingCalls",
        "CallWaitingCalls",
    ],
    "usage_change_retention": [
        "PercChangeMinutes",
        "PercChangeRevenues",
        "RetentionCalls",
        "RetentionOffersAccepted",
        "MadeCallToRetentionTeam",
        "ReferralsMadeBySubscriber",
        "NewCellphoneUser",
        "NotNewCellphoneUser",
    ],
    "service_quality": [
        "DroppedCalls",
        "BlockedCalls",
        "UnansweredCalls",
        "DroppedBlockedCalls",
        "CustomerCareCalls",
    ],
    "demographics_region": [
        "AgeHH1",
        "AgeHH2",
        "ChildrenInHH",
        "IncomeGroup",
        "Occupation",
        "MaritalStatus",
        "Homeownership",
        "CreditRating",
        "PrizmCode",
        "ServiceArea",
    ],
    "handset_lifestyle": [
        "HandsetPrice",
        "HandsetRefurbished",
        "HandsetWebCapable",
        "TruckOwner",
        "RVOwner",
        "OwnsMotorcycle",
        "OwnsComputer",
        "HasCreditCard",
        "BuysViaMailOrder",
        "RespondsToMailOffers",
        "OptOutMailings",
        "NonUSTravel",
    ],
}

# values that most likely mean "unknown" rather than a real value
PLACEHOLDERS = {
    "AgeHH1": 0,
    "AgeHH2": 0,
    "IncomeGroup": 0,
    "HandsetPrice": "Unknown",
    "MaritalStatus": "Unknown",
    "Homeownership": "Unknown",
}


def is_numeric(s):
    """check whether a series has a numeric dtype"""
    return pd.api.types.is_numeric_dtype(s)


def overview(df, cols):
    """summarize types, missing values, ranges and cardinality for each column"""
    rows = []
    for column in cols:
        values = df[column]
        numeric = is_numeric(values)
        rows.append({
            "column": column,
            "type": "numeric" if numeric else "categorical",
            "n_unique": values.nunique(),
            "missing_%": round(100 * values.isna().mean(), 2),
            "placeholder_%": (
                round(100 * (values == PLACEHOLDERS[column]).mean(), 2)
                if column in PLACEHOLDERS else None
            ),
            "zero_%": round(100 * (values == 0).mean(), 2) if numeric else None,
            "negative_%": round(100 * (values < 0).mean(), 2) if numeric else None,
            "min": values.min() if numeric else None,
            "median": values.median() if numeric else None,
            "p99": values.quantile(0.99) if numeric else None,
            "max": values.max() if numeric else None,
            "top_value": values.mode().iloc[0] if not numeric else None,
        })
    table = pd.DataFrame(rows)
    return table.set_index("column")


def churn_signal(df, cols, target="churn"):
    """measure numeric columns with roc-auc and categorical columns by churn rate spread"""
    rows = []
    for column in cols:
        values = df[column]
        if is_numeric(values):
            observed = values.notna()
            area_under_curve = roc_auc_score(df.loc[observed, target], values[observed])
            rows.append({
                "column": column,
                "auc": round(area_under_curve, 3),
                "signal": round(abs(area_under_curve - 0.5), 3),
            })
        else:
            churn_rates = df.groupby(column)[target].mean()
            rows.append({
                "column": column,
                "churn_rate_min": round(churn_rates.min(), 3),
                "churn_rate_max": round(churn_rates.max(), 3),
            })
    table = pd.DataFrame(rows)
    return table.set_index("column")


def churn_rate_by(df, col, bins=10, target="churn"):
    """calculate the size and churn rate of each category or quantile bin"""
    values = df[col]
    if is_numeric(values) and values.nunique() > bins:
        groups = pd.qcut(values, bins, duplicates="drop")
    else:
        groups = values
    grouped_target = df.groupby(groups, observed=True)[target]
    counts = grouped_target.size()
    churn_rates = grouped_target.mean()
    churn_rates = churn_rates.round(3)
    return pd.DataFrame({"n": counts, "churn_rate": churn_rates})


def plot_column(df, col, target="churn", bins=10):
    """plot the distribution and churn rate, with a line for the overall rate"""
    figure, axes = plt.subplots(1, 2, figsize=(11, 3))
    values = df[col]
    if is_numeric(values):
        observed = values.dropna()
        clipped = observed.clip(upper=values.quantile(0.99))
        axes[0].hist(clipped, bins=40)
        axes[0].set_title(f"{col} (clipped at p99)")
    else:
        counts = values.value_counts()
        top_counts = counts.head(20)
        top_counts.plot.bar(ax=axes[0])
        axes[0].set_title(col)
    rates = churn_rate_by(df, col, bins, target)
    axes[1].bar(range(len(rates)), rates["churn_rate"])
    axes[1].set_xticks(
        range(len(rates)), [str(i) for i in rates.index],
        rotation=45, ha="right", fontsize=7,
    )
    axes[1].axhline(df[target].mean(), ls="--", c="grey")
    axes[1].set_title(f"churn rate by {col}")
    plt.tight_layout()
    plt.show()


def corr_heatmap(df, cols):
    """calculate and plot spearman correlations for numeric columns"""
    numeric_columns = [c for c in cols if is_numeric(df[c])]
    correlations = df[numeric_columns].corr(method="spearman")
    figure, axis = plt.subplots(
        figsize=(0.6 * len(numeric_columns) + 3, 0.6 * len(numeric_columns) + 2),
    )
    heatmap = axis.imshow(correlations, vmin=-1, vmax=1, cmap="RdBu_r")
    axis.set_xticks(range(len(numeric_columns)), numeric_columns, rotation=90)
    axis.set_yticks(range(len(numeric_columns)), numeric_columns)
    for row in range(len(numeric_columns)):
        for column_index in range(len(numeric_columns)):
            axis.text(
                column_index, row, f"{correlations.iloc[row, column_index]:.2f}",
                ha="center", va="center", fontsize=7,
            )
    figure.colorbar(heatmap)
    plt.tight_layout()
    plt.show()
    return correlations


def summary_skeleton():
    """create an empty decision table to fill in during exploration"""
    rows = [
        {"group": group, "column": column}
        for group, columns in FEATURE_GROUPS.items()
        for column in columns
    ]
    table = pd.DataFrame(rows)
    fields = [
        "issues",
        "relation_to_churn",
        "keep_drop",
        "missing_handling",
        "transformation",
        "encoding",
        "feature_ideas",
    ]
    for field in fields:
        table[field] = ""
    return table.set_index("column")
