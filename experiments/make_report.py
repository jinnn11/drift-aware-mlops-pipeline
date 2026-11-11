"""Aggregate experiment outputs into tables (markdown) and figures (PNG).

    python -m experiments.make_report --results results --figures docs/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference categorical palette, fixed order; colors follow the entity across all figures.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL, INK, MUTED, GRID = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"

POLICY_ORDER = ["never", "scheduled_monthly", "scheduled_quarterly", "drift_ks_chi2", "drift_psi",
                "drift_domain_classifier", "drift_domain_classifier_auc55", "drift_calibration"]
POLICY_LABEL = {
    "never": "Never retrain",
    "scheduled_monthly": "Scheduled, monthly",
    "scheduled_quarterly": "Scheduled, quarterly",
    "drift_ks_chi2": "Drift: KS / chi-square",
    "drift_psi": "Drift: PSI >= 0.1",
    "drift_domain_classifier": "Drift: domain classifier (p < 0.05)",
    "drift_domain_classifier_auc55": "Drift: domain classifier (AUC >= 0.55)",
    "drift_calibration": "Labels: calibration test",
}

DETECTOR_ORDER = ["ks_chi2", "psi", "domain_classifier", "domain_classifier_auc55", "calibration",
                  "calibration (labels)"]
DETECTOR_LABEL = {
    "ks_chi2": "KS / chi-square",
    "psi": "PSI >= 0.1",
    "domain_classifier": "Domain classifier",
    "domain_classifier_auc55": "Domain classifier, AUC >= 0.55",
    "calibration": "Calibration (needs labels)",
    "calibration (labels)": "Calibration (needs labels)",
}
DETECTOR_COLOR = {"ks_chi2": SLOTS[0], "psi": SLOTS[1], "domain_classifier": SLOTS[2],
                  "domain_classifier_auc55": SLOTS[4], "calibration": SLOTS[3], "calibration (labels)": SLOTS[3]}
# A drift policy wears its detector's color, so one detector looks the same in every figure.
POLICY_COLOR = {"never": SLOTS[7], "scheduled_monthly": SLOTS[6], "scheduled_quarterly": SLOTS[5],
                **{f"drift_{d}": c for d, c in DETECTOR_COLOR.items() if d != "calibration (labels)"}}
SCENARIO_LABEL = {
    "no_drift": "No drift",
    "covariate_sudden": "Input-only, sudden",
    "covariate_gradual": "Input-only, gradual",
    "concept_sudden": "Concept-only, sudden",
    "concept_gradual": "Concept-only, gradual",
}


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=11, color=INK, fontweight="semibold")


def load_policy_runs(results: Path) -> dict:
    out = {}
    for delay_dir in sorted((results / "policy_comparison").glob("delay_*")):
        delay = int(delay_dir.name.split("_")[1])
        parts = {name: [] for name in ["summary", "monthly", "hindsight", "shadow", "events"]}
        for seed_dir in sorted(delay_dir.glob("seed_*")):
            seed = int(seed_dir.name.split("_")[1])
            for name in parts:
                path = seed_dir / f"{name}.csv"
                if path.exists() and path.stat().st_size > 1:
                    parts[name].append(pd.read_csv(path).assign(seed=seed))
        out[delay] = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in parts.items()}
    return out


def policy_table(runs: dict) -> str:
    lines = []
    for delay, data in runs.items():
        s = data["summary"]
        if s.empty:
            continue
        agg = s.groupby("policy").agg(
            seeds=("seed", "nunique"),
            pr_auc=("mean_pr_auc", "mean"), pr_auc_sd=("mean_pr_auc", "std"),
            gap=("mean_gap_to_hindsight", "mean"), gap_sd=("mean_gap_to_hindsight", "std"),
            retrains=("retrains", "mean"), promotions=("promotions", "mean"),
            train_s=("train_seconds", "mean"), alarms=("alarms", "mean"),
            blocked=("blocked_cooldown", "mean"),
        ).reindex([p for p in POLICY_ORDER if p in set(s["policy"])])
        lines.append(f"\n**Label delay {delay} month(s)**, mean over {int(agg['seeds'].max())} seeds (± sd)\n")
        lines.append("| Policy | Mean PR-AUC | Gap to hindsight | Retrains | Promoted | Train s | Alarms | Blocked by cooldown |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for policy, r in agg.iterrows():
            lines.append(
                f"| {POLICY_LABEL[policy]} | {r.pr_auc:.4f} ± {r.pr_auc_sd:.4f} | {r.gap:.4f} ± {r.gap_sd:.4f} | "
                f"{r.retrains:.1f} | {r.promotions:.1f} | {r.train_s:.1f} | {r.alarms:.1f} | {r.blocked:.1f} |")
    return "\n".join(lines)


def paired_table(runs: dict, baseline: str = "scheduled_monthly") -> str:
    """Per-seed paired difference in mean PR-AUC vs a baseline policy (same stream, same seed)."""
    lines = [f"| Delay | Policy | PR-AUC minus {POLICY_LABEL[baseline].lower()} (mean ± sd over seeds) | Seeds better |",
             "|---:|---|---:|---:|"]
    for delay, data in runs.items():
        s = data["summary"]
        if s.empty:
            continue
        wide = s.pivot(index="seed", columns="policy", values="mean_pr_auc")
        for policy in [p for p in POLICY_ORDER if p in wide.columns and p != baseline]:
            diff = wide[policy] - wide[baseline]
            lines.append(f"| {delay} | {POLICY_LABEL[policy]} | {diff.mean():+.4f} ± {diff.std():.4f} | "
                         f"{int((diff > 0).sum())}/{len(diff)} |")
    return "\n".join(lines)


def dollar_table(runs: dict, rate: int = 10, baseline: str = "never") -> str:
    """Lending rule with a dollar budget: each month, turn down the riskiest applications by the deployed
    model's score until `rate`% of requested dollars is declined (src/lending_value.py).

    Every policy declines the same dollars, so the comparison is how much of that budget lands on loans that
    later charged off. Dollars are requested loan amounts (the data has no repayment or recovery amounts),
    scaled to $1B of applications and paired by seed against `baseline` (same stream, same seed).
    """
    bad_col, good_col = f"defaulted_declined_usd_{rate}", f"good_declined_usd_{rate}"
    lines = [f"| Delay | Policy | Retrains | Declined $ that later defaulted | Defaulted $ declined vs "
             f"{POLICY_LABEL[baseline].lower()}, per $1B of applications | Seeds better |",
             "|---:|---|---:|---:|---:|---:|"]
    for delay, data in runs.items():
        s = data["summary"]
        if s.empty or bad_col not in s.columns:
            continue
        s = s.assign(precision=s[bad_col] / (s[bad_col] + s[good_col]), bad=s[bad_col] * 1e9 / s["loan_dollars"])
        wide = {k: s.pivot(index="seed", columns="policy", values=k) for k in ["precision", "bad", "retrains"]}
        for policy in [p for p in POLICY_ORDER if p in wide["bad"].columns]:
            head = (f"| {delay} | {POLICY_LABEL[policy]} | {wide['retrains'][policy].mean():.1f} | "
                    f"{wide['precision'][policy].mean():.2%} |")
            if policy == baseline:
                lines.append(head + " – | – |")
                continue
            diff = wide["bad"][policy] - wide["bad"][baseline]
            lines.append(head + f" {diff.mean() / 1e6:+.2f}M ± {diff.std() / 1e6:.2f}M | "
                         f"{int((diff > 0).sum())}/{len(diff)} |")
    return "\n".join(lines)


def gate_table(runs: dict) -> str:
    lines = ["| Delay | Policy | Retrains | Promoted | Rejected: no gain on recent | Rejected: worse on older |",
             "|---:|---|---:|---:|---:|---:|"]
    for delay, data in runs.items():
        e = data["events"]
        if e.empty:
            continue
        n_seeds = data["summary"]["seed"].nunique()
        counts = e.groupby(["policy", "gate_reason"]).size().unstack(fill_value=0) / n_seeds
        for policy in [p for p in POLICY_ORDER if p in counts.index]:
            r = counts.loc[policy]
            lines.append(f"| {delay} | {POLICY_LABEL[policy]} | {r.sum():.1f} | {r.get('promoted', 0):.1f} | "
                         f"{r.get('no_gain_on_recent', 0):.1f} | {r.get('regressed_on_older', 0):.1f} |")
    return "\n".join(lines)


def shadow_table(runs: dict) -> str:
    lines = ["| Delay | Detector | Alarm months | Harmful months | Alarms followed by harm (precision) | "
             "Harmful months with an alarm (recall) | First alarm | First harmful month |",
             "|---:|---|---:|---:|---:|---:|---|---|"]
    for delay, data in runs.items():
        sh = data["shadow"]
        if sh.empty:
            continue
        for det in [d for d in DETECTOR_ORDER if d in set(sh["detector"])]:
            g = sh[sh["detector"] == det]
            per_seed = g.groupby("seed").apply(lambda x: pd.Series({
                "alarms": x["alarm"].sum(), "harm": x["harm_ahead"].sum(),
                "precision": x.loc[x["alarm"], "harm_ahead"].mean() if x["alarm"].any() else np.nan,
                "recall": x.loc[x["harm_ahead"], "alarm"].mean() if x["harm_ahead"].any() else np.nan,
            }), include_groups=False)
            seed0 = g[g["seed"] == g["seed"].min()]
            first_alarm = seed0.loc[seed0["alarm"], "month"].min() if seed0["alarm"].any() else "never"
            first_harm = seed0.loc[seed0["harm_ahead"], "month"].min() if seed0["harm_ahead"].any() else "never"
            lines.append(f"| {delay} | {DETECTOR_LABEL[det]} | {per_seed['alarms'].mean():.1f} | "
                         f"{per_seed['harm'].mean():.1f} | {per_seed['precision'].mean():.2f} | "
                         f"{per_seed['recall'].mean():.2f} | {first_alarm} | {first_harm} |")
    return "\n".join(lines)


def fig_policy_timeline(data: dict, delay: int, policies: list, path: Path) -> None:
    monthly, hindsight = data["monthly"], data["hindsight"]
    months = sorted(monthly["month"].unique())
    x = pd.PeriodIndex(months, freq="M").to_timestamp()
    hs = hindsight.groupby("month")["hindsight_pr_auc"].mean().reindex(months)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    top.plot(x, hs.to_numpy(), color=NEUTRAL, linewidth=2, label="Hindsight reference (no delay, monthly)")
    for policy in policies:
        series = monthly[monthly["policy"] == policy].groupby("month")["pr_auc"].mean().reindex(months)
        top.plot(x, series.to_numpy(), color=POLICY_COLOR[policy], linewidth=2, label=POLICY_LABEL[policy])
        gap = (hs - series).rolling(3, min_periods=1).mean()
        bottom.plot(x, gap.to_numpy(), color=POLICY_COLOR[policy], linewidth=2, label=POLICY_LABEL[policy])
    _title(top, f"PR-AUC by issue month, label delay {delay} months (mean of seeds)")
    top.set_ylabel("PR-AUC")
    top.legend(frameon=False, fontsize=9, loc="upper left")
    _title(bottom, "Shortfall vs hindsight reference (3-month rolling mean, lower is better)")
    bottom.set_ylabel("PR-AUC points behind")
    bottom.axhline(0, color=MUTED, linewidth=1)
    for ax in (top, bottom):
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cost_quality(runs: dict, path: Path) -> None:
    delays = [d for d in runs if not runs[d]["summary"].empty]
    fig, axes = plt.subplots(1, len(delays), figsize=(5 * len(delays), 4.6), sharey=True, squeeze=False)
    for ax, delay in zip(axes[0], delays):
        agg = runs[delay]["summary"].groupby("policy")[["retrains", "mean_gap_to_hindsight"]].mean()
        for policy in [p for p in POLICY_ORDER if p in agg.index]:
            r = agg.loc[policy]
            ax.scatter(r.retrains, r.mean_gap_to_hindsight, s=70, color=POLICY_COLOR[policy],
                       edgecolor="white", linewidth=2, zorder=3, label=POLICY_LABEL[policy])
        _title(ax, f"Label delay {delay} month{'s' if delay > 1 else ''}")
        ax.set_xlabel("Retrains over 36 months (compute cost)")
        _style(ax)
    axes[0][0].set_ylabel("Mean PR-AUC behind hindsight (lower is better)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_shadow(data: dict, delay: int, path: Path) -> None:
    sh = data["shadow"]
    seed = sh["seed"].min()
    g = sh[sh["seed"] == seed]
    months = sorted(g["month"].unique())
    x = pd.PeriodIndex(months, freq="M").to_timestamp()
    regret = g.groupby("month")["regret_now"].first().reindex(months)
    harm = g.groupby("month")["harm_ahead"].first().reindex(months)
    detectors = [d for d in DETECTOR_ORDER if d in set(g["detector"])]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 3]})
    top.plot(x, regret.to_numpy(), color=NEUTRAL, linewidth=2)
    top.axhline(0.02, color=MUTED, linewidth=1)
    top.text(x[0], 0.021, "harm margin 0.02", fontsize=8, color=MUTED, va="bottom")
    _title(top, "Never-retrained model: PR-AUC behind the hindsight model")
    top.set_ylabel("PR-AUC points")

    for i, det in enumerate(detectors):
        d = g[g["detector"] == det].set_index("month").reindex(months)
        on = d["alarm"].fillna(False).astype(bool).to_numpy()
        bottom.scatter(x[on], np.full(on.sum(), i), s=40, color=DETECTOR_COLOR[det], edgecolor="white",
                       linewidth=1.5, zorder=3)
    for xi, h in zip(x, harm.to_numpy()):
        if h:
            bottom.axvspan(xi - pd.Timedelta(days=15), xi + pd.Timedelta(days=15), color="#f0efec", zorder=0)
    bottom.set_yticks(range(len(detectors)), [DETECTOR_LABEL[d] for d in detectors])
    bottom.set_ylim(-0.6, len(detectors) - 0.4)
    _title(bottom, "Monthly alarms (dots); shaded months are followed by >0.02 PR-AUC shortfall")
    for ax in (top, bottom):
        _style(ax)
    bottom.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_benchmark(batches: pd.DataFrame, config: dict, path: Path) -> None:
    scenarios = list(SCENARIO_LABEL)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(17, 3.8), sharey=True)
    for ax, scenario in zip(axes, scenarios):
        g = batches[batches["scenario"] == scenario]
        for det in [d for d in DETECTOR_ORDER if d in set(g["detector"])]:
            rate = g[g["detector"] == det].groupby("batch")["alarm"].mean()
            ax.plot(rate.index, rate.to_numpy(), color=DETECTOR_COLOR[det], linewidth=2, label=DETECTOR_LABEL[det])
        if scenario != "no_drift":
            ax.axvline(config["onset"], color=MUTED, linewidth=1)
            if "gradual" in scenario:
                ax.axvspan(config["onset"], config["onset"] + config["ramp"] - 1, color="#f0efec", zorder=0)
        _title(ax, SCENARIO_LABEL[scenario])
        ax.set_xlabel("Batch (month)")
        _style(ax)
    axes[0].set_ylabel(f"Share of {batches['seed'].nunique()} seeds alarming")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def benchmark_tables(results: Path) -> str:
    metrics = pd.read_csv(results / "detector_benchmark" / "metrics.csv")
    regret = pd.read_csv(results / "detector_benchmark" / "regret.csv")
    lines = ["| Scenario | Detector | False alarm rate | Detected (of seeds) | Detected by chance | "
             "Median delay (batches) | Alarms followed by harm |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in metrics.iterrows():
        detected = "n/a" if pd.isna(r.detection_rate) else f"{r.detection_rate:.0%}"
        chance = "n/a" if pd.isna(r.chance_detection_rate) else f"{r.chance_detection_rate:.0%}"
        delay = "n/a" if pd.isna(r.median_delay) else f"{r.median_delay:.0f}"
        precision = "n/a" if pd.isna(r.alarm_precision) else f"{r.alarm_precision:.0%}"
        lines.append(f"| {SCENARIO_LABEL[r.scenario]} | {DETECTOR_LABEL[r.detector]} | {r.false_alarm_rate:.1%} | "
                     f"{detected} | {chance} | {delay} | {precision} |")
    full = regret[regret["strength"].isin([0.0, 1.0])].groupby(["kind", "strength"])[
        ["champion_pr_auc", "oracle_pr_auc", "regret", "prevalence"]].mean()
    lines += ["", "Harm at full drift strength (mean over seeds):", "",
              "| Drift | Strength | Champion PR-AUC | Oracle PR-AUC | Regret | Default rate |",
              "|---|---:|---:|---:|---:|---:|"]
    for (kind, strength), r in full.iterrows():
        lines.append(f"| {kind} | {strength:.0f} | {r.champion_pr_auc:.4f} | {r.oracle_pr_auc:.4f} | "
                     f"{r.regret:+.4f} | {r.prevalence:.3f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="docs/figures")
    parser.add_argument("--timeline-delay", type=int, default=3)
    parser.add_argument("--timeline-policies", default="never,scheduled_monthly,drift_ks_chi2,drift_psi")
    args = parser.parse_args()
    results, figures = Path(args.results), Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    runs = load_policy_runs(results)
    sections = ["## Policy comparison", policy_table(runs), "", "## Paired differences", paired_table(runs)]
    for rate in (10, 5, 20):
        sections += ["", f"## Lending rule in dollars: decline the riskiest {rate}% of requested dollars each month",
                     dollar_table(runs, rate)]
    sections += ["", "## Promotion gate outcomes", gate_table(runs), "", "## Shadow monitoring", shadow_table(runs)]
    if args.timeline_delay in runs and not runs[args.timeline_delay]["monthly"].empty:
        fig_policy_timeline(runs[args.timeline_delay], args.timeline_delay, args.timeline_policies.split(","),
                            figures / "policy_pr_auc_over_time.png")
        fig_shadow(runs[args.timeline_delay], args.timeline_delay, figures / "shadow_alarms_vs_harm.png")
    fig_cost_quality(runs, figures / "cost_vs_quality.png")

    bench = results / "detector_benchmark"
    if (bench / "metrics.csv").exists():
        import json

        fig_benchmark(pd.read_csv(bench / "batches.csv"), json.loads((bench / "config.json").read_text()),
                      figures / "detector_benchmark.png")
        sections += ["", "## Detector benchmark", benchmark_tables(results)]
    report = "\n".join(sections)
    (results / "report_tables.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
