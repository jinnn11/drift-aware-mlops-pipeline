import numpy as np
import pytest

from src.lending_value import decline_rule_dollars

AMOUNTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
DEFAULTS = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0]
# riskiest first: rows 0, 1, 8, 7, 6, 5, 4, 3, 2, 9
SCORES = [0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05]


def test_totals():
    out = decline_rule_dollars(AMOUNTS, DEFAULTS, SCORES, rates=(0.1,))
    assert out["loan_dollars"] == 55000
    assert out["defaulted_dollars"] == 1000 + 3000 + 9000


def test_application_count_rule_declines_the_riskiest_share_of_applications():
    out = decline_rule_dollars(AMOUNTS, DEFAULTS, SCORES, rates=(0.1, 0.2, 0.3))
    assert (out["defaulted_declined_apps_10"], out["good_declined_apps_10"]) == (1000, 0)
    assert (out["defaulted_declined_apps_20"], out["good_declined_apps_20"]) == (1000, 2000)
    assert (out["defaulted_declined_apps_30"], out["good_declined_apps_30"]) == (10000, 2000)


def test_dollar_budget_rule_declines_until_the_budget_is_reached():
    out = decline_rule_dollars(AMOUNTS, DEFAULTS, SCORES, rates=(0.05, 0.2))
    # 5% of $55,000 = $2,750: rows 0 and 1 ($3,000) reach it
    assert (out["defaulted_declined_usd_5"], out["good_declined_usd_5"]) == (1000, 2000)
    # 20% = $11,000: rows 0, 1 and 8 ($12,000)
    assert (out["defaulted_declined_usd_20"], out["good_declined_usd_20"]) == (10000, 2000)


def test_better_ranking_shifts_a_fixed_dollar_budget_toward_defaulted_loans():
    rng = np.random.default_rng(0)
    defaults = rng.random(5000) < 0.2
    amounts = rng.uniform(1000, 35000, 5000)
    informative = defaults + rng.normal(0, 0.5, 5000)
    random = rng.normal(0, 1, 5000)

    good = decline_rule_dollars(amounts, defaults, informative, rates=(0.1,))
    bad = decline_rule_dollars(amounts, defaults, random, rates=(0.1,))

    budget = 0.1 * amounts.sum()
    for out in (good, bad):
        declined = out["defaulted_declined_usd_10"] + out["good_declined_usd_10"]
        assert budget <= declined < budget + 35000  # overshoots by less than one loan
    assert good["defaulted_declined_usd_10"] > bad["defaulted_declined_usd_10"]
    assert good["good_declined_usd_10"] < bad["good_declined_usd_10"]


def test_rejects_mismatched_lengths_and_bad_rates():
    with pytest.raises(ValueError):
        decline_rule_dollars([1, 2], [0, 1], [0.5])
    with pytest.raises(ValueError):
        decline_rule_dollars([1, 2], [0, 1], [0.5, 0.4], rates=(1.0,))
