"""Descriptive summaries and plots used in the data exploration notebook."""

from collections.abc import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

# Recorded values that may represent missing information in Cell2Cell.
PLACEHOLDERS: dict[str, int | str] = {
    "AgeHH1": 0,
    "AgeHH2": 0,
    "IncomeGroup": 0,
    "HandsetPrice": "Unknown",
    "MaritalStatus": "Unknown",
    "Homeownership": "Unknown",
}


def is_numeric(s: pd.Series) -> bool:
    """Return whether pandas stores the series as a numeric type."""
    return pd.api.types.is_numeric_dtype(s)


def overview(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Summarize the type, missingness, distinct values and distribution of each column.

    The result has one row per input column. Numeric columns show ranges and
    percentiles, while categorical columns show the prevailing observed value.
    Recorded placeholders and actual missing values are counted separately.
    """
    rows = []
    for column in cols:
        values = df[column]
        numeric = is_numeric(values)
        modes = values.mode() if not numeric else None
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
            "top_value": modes.iloc[0] if modes is not None and not modes.empty else None,
        })
    return pd.DataFrame(rows).set_index("column")


def churn_signal(
    df: pd.DataFrame, cols: Iterable[str], target: str = "churn",
) -> pd.DataFrame:
    """Describe each column's association with the binary churn label.

    For numeric columns, report ROC AUC on observed rows and its distance from
    0.5. For categorical columns, report the lowest and highest churn rates.
    """
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
    return pd.DataFrame(rows).set_index("column")


def churn_rate_by(
    df: pd.DataFrame, col: str, bins: int = 10, target: str = "churn",
) -> pd.DataFrame:
    """Return customer counts and mean churn labels by observed value or quantile bin.

    Numeric columns with more than `bins` distinct values use quantile bins.
    Rows missing the selected value are excluded.
    """
    values = df[col]
    if is_numeric(values) and values.nunique() > bins:
        groups = pd.qcut(values, bins, duplicates="drop")
    else:
        groups = values
    grouped_target = df.groupby(groups, observed=True)[target]
    return pd.DataFrame({
        "n": grouped_target.size(),
        "churn_rate": grouped_target.mean().round(3),
    })


def plot_column(
    df: pd.DataFrame, col: str, target: str = "churn", bins: int = 10,
) -> None:
    """Plot the input distribution and its churn rates against the overall rate.

    The distribution view clips numeric values at the 99th percentile and shows
    the 20 most common categorical values. Churn rates use the original values.
    """
    figure, (distribution_axis, churn_axis) = plt.subplots(1, 2, figsize=(11, 3))
    values = df[col]
    if is_numeric(values):
        observed = values.dropna()
        clipped = observed.clip(upper=values.quantile(0.99))
        distribution_axis.hist(clipped, bins=40)
        distribution_axis.set_title(f"{col} (clipped at p99)")
    else:
        top_counts = values.value_counts().head(20)
        top_counts.plot.bar(ax=distribution_axis)
        distribution_axis.set_title(col)
    rates = churn_rate_by(df, col, bins, target)
    positions = range(len(rates))
    churn_axis.bar(positions, rates["churn_rate"])
    churn_axis.set_xticks(
        positions, [str(label) for label in rates.index],
        rotation=45, ha="right", fontsize=7,
    )
    churn_axis.axhline(df[target].mean(), ls="--", c="grey")
    churn_axis.set_title(f"churn rate by {col}")
    figure.tight_layout()
    plt.show()


def corr_heatmap(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Plot and return the Spearman correlation matrix for the numeric columns."""
    numeric_columns = [column for column in cols if is_numeric(df[column])]
    correlations = df[numeric_columns].corr(method="spearman")
    column_count = len(numeric_columns)
    figure, axis = plt.subplots(
        figsize=(0.6 * column_count + 3, 0.6 * column_count + 2),
    )
    heatmap = axis.imshow(correlations, vmin=-1, vmax=1, cmap="RdBu_r")
    positions = range(column_count)
    axis.set_xticks(positions, numeric_columns, rotation=90)
    axis.set_yticks(positions, numeric_columns)
    for row in positions:
        for column_index in positions:
            axis.text(
                column_index, row, f"{correlations.iloc[row, column_index]:.2f}",
                ha="center", va="center", fontsize=7,
            )
    figure.colorbar(heatmap)
    figure.tight_layout()
    plt.show()
    return correlations
