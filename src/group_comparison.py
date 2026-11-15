"""Compare group contributions using cached scores for every feature subset."""
from itertools import permutations

import numpy as np
import pandas as pd


def group_contributions(fold_results, configurations, baseline_groups, optional_groups):
    """Return per-fold gains and gains in every possible group addition order.

    The average over orders is a Shapley allocation of the full-minus-baseline AP
    difference. These are refitted-model comparisons, not SHAP prediction values.
    Background groups stay present and are not ranked by this calculation.
    """
    baseline = frozenset(baseline_groups)
    optional = tuple(optional_groups)
    if not optional or len(set(optional)) != len(optional) or baseline.intersection(optional):
        raise ValueError('Optional groups must be nonempty, unique and outside the baseline')
    keys = {}
    for name, groups in configurations.items():
        group_set = frozenset(groups)
        if len(group_set) != len(groups) or not baseline.issubset(group_set):
            raise ValueError('Every configuration must contain the complete baseline')
        subset = group_set - baseline
        if not subset.issubset(optional) or subset in keys:
            raise ValueError('Configurations must contain distinct optional-group subsets')
        keys[subset] = name
    if len(keys) != 2 ** len(optional):
        raise ValueError('Supply every optional-group subset, including the baseline')
    scores = fold_results.pivot(index='fold', columns='model', values='average_precision')
    if set(scores.columns) != set(configurations) or not np.isfinite(scores.to_numpy()).all():
        raise ValueError('Every configuration needs a finite AP score on the same folds')

    full = frozenset(optional)
    order_rows = []
    for fold, row in scores.iterrows():
        for order in permutations(optional):
            previous = frozenset()
            for group in order:
                current = previous | {group}
                order_rows.append({
                    'fold': fold, 'order': ' -> '.join(order), 'group': group,
                    'ap_gain': row[keys[current]] - row[keys[previous]],
                })
                previous = current
    order_gains = pd.DataFrame(order_rows)
    rows = []
    for fold, row in scores.iterrows():
        for group in optional:
            gains = order_gains.loc[
                (order_gains['fold'] == fold) & (order_gains['group'] == group), 'ap_gain',
            ]
            rows.append({
                'fold': fold, 'group': group,
                'add_to_baseline': row[keys[frozenset([group])]] - row[keys[frozenset()]],
                'remove_from_full': row[keys[full]] - row[keys[full - {group}]],
                'mean_over_orders': gains.mean(),
                'min_order_gain': gains.min(), 'max_order_gain': gains.max(),
                'orders_improved': int((gains > 0).sum()), 'orders': len(gains),
            })
    return pd.DataFrame(rows), order_gains


def summarize_contributions(contributions):
    """Summarize paired gains; fold standard deviations are not confidence intervals."""
    return contributions.groupby('group', sort=False).agg(
        add_to_baseline=('add_to_baseline', 'mean'),
        remove_from_full=('remove_from_full', 'mean'),
        mean_over_orders=('mean_over_orders', 'mean'),
        std_over_folds=('mean_over_orders', 'std'),
        folds_positive=('mean_over_orders', lambda values: int((values > 0).sum())),
        folds=('fold', 'count'),
        min_order_gain=('min_order_gain', 'min'),
        max_order_gain=('max_order_gain', 'max'),
    )
