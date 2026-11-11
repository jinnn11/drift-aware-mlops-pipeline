"""What a credit model is worth in dollars under a simple lending rule.

Each month the lender turns down the riskiest applications by model score and funds the rest.
Two versions of the rule, for each decline rate:

- dollar budget (`usd`): turn down the riskiest applications until `rate` of the month's
  requested dollars is declined. Every model declines the same dollars, so a better model
  shifts declined dollars from loans that would have been repaid to loans that later charged off.
  This is the headline comparison.
- application count (`apps`): turn down the riskiest `rate` share of applications. Simpler,
  but a model that ranks bigger loans as riskier declines more dollars of both kinds.

For each rule we record the declined dollars of loans that later charged off
(`defaulted_declined_*`, what the rule keeps off the books) and of loans that were repaid in
full (`good_declined_*`, business given up). Amounts are requested loan amounts (`loan_amnt`):
the dataset has no repayment or recovery amounts, so these are not realized losses.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

DECLINE_RATES = (0.05, 0.10, 0.20)


def rate_tag(rate: float) -> str:
    return str(round(rate * 100))


def decline_rule_dollars(amounts, defaults, scores, rates: Iterable[float] = DECLINE_RATES) -> Dict[str, float]:
    amounts = np.asarray(amounts, dtype=float)
    defaults = np.asarray(defaults).astype(bool)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if not len(amounts) == len(defaults) == n:
        raise ValueError("amounts, defaults and scores must have the same length")

    order = np.argsort(-scores, kind="stable")  # riskiest first; ties keep row order
    cumulative = np.cumsum(amounts[order])
    total = float(amounts.sum())
    out = {"loan_dollars": total, "defaulted_dollars": float(amounts[defaults].sum())}
    for rate in rates:
        if not 0 < rate < 1:
            raise ValueError(f"decline rate must be between 0 and 1, got {rate}")
        tag = rate_tag(rate)
        cutoffs = {
            "apps": int(round(rate * n)),
            # smallest riskiest-first prefix whose requested dollars reach the budget
            "usd": min(n, int(np.searchsorted(cumulative, rate * total, side="left")) + 1),
        }
        for rule, k in cutoffs.items():
            declined = np.zeros(n, dtype=bool)
            declined[order[:k]] = True
            out[f"defaulted_declined_{rule}_{tag}"] = float(amounts[declined & defaults].sum())
            out[f"good_declined_{rule}_{tag}"] = float(amounts[declined & ~defaults].sum())
    return out
