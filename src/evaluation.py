"""Measure a ranked outreach list at a fixed contact budget."""

import numpy as np
import pandas as pd


def contact_top(churn_probability, share: float) -> np.ndarray:
    """Flag the given share of customers with the highest churn probability."""
    churn_probability = np.asarray(churn_probability)
    n_contact = round(share * len(churn_probability))
    contacted = np.zeros(len(churn_probability), dtype=bool)
    contacted[np.argsort(-churn_probability, kind="stable")[:n_contact]] = True
    return contacted


def outreach_scores(y_true, churn_probability, share: float) -> dict:
    """Return precision, recall, lift and outcome counts for the top share."""
    churned = np.asarray(y_true) == 1
    contacted = contact_top(churn_probability, share)
    churners_contacted = int((contacted & churned).sum())
    precision = churners_contacted / contacted.sum()
    return {
        "Precision": precision,
        "Recall": churners_contacted / churned.sum(),
        "Lift": precision / churned.mean(),
        "Churners contacted": churners_contacted,
        "Stayers contacted": int((contacted & ~churned).sum()),
        "Churners missed": int((~contacted & churned).sum()),
        "Customers": len(churned),
    }


def cost_matrix(contact_cost: float, churn_loss: float, save_rate: float) -> pd.DataFrame:
    """Return the cost per customer for each outcome of an outreach campaign.

    A contacted churner stays with probability ``save_rate``. Customers who
    stay without a contact cost nothing.
    """
    return pd.DataFrame(
        {
            "Contacted": [contact_cost + (1 - save_rate) * churn_loss, contact_cost],
            "Not contacted": [churn_loss, 0.0],
        },
        index=pd.Index(["Churned", "Stayed"], name="Actual"),
    )


def savings_per_1000(scores, costs: pd.DataFrame):
    """Return savings per 1,000 customers compared with no campaign.

    ``scores`` holds the counts from ``outreach_scores``, as a dict or a
    DataFrame with one row per evaluation.
    """
    campaign_cost = (
        scores["Churners contacted"] * costs.loc["Churned", "Contacted"]
        + scores["Stayers contacted"] * costs.loc["Stayed", "Contacted"]
        + scores["Churners missed"] * costs.loc["Churned", "Not contacted"]
    )
    no_campaign_cost = (
        scores["Churners contacted"] + scores["Churners missed"]
    ) * costs.loc["Churned", "Not contacted"]
    return (no_campaign_cost - campaign_cost) / scores["Customers"] * 1000
