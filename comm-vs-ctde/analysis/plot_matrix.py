"""Pull 2x2 ablation matrix from wandb and produce blog-post figures.

Inputs: runs in wandb project `comm-vs-ctde-final` named like
    {algo}_{comm}_seed{N}     (e.g. "iql_no_comm_seed1")

Outputs:
    figures/learning_curves.png + .pdf
    figures/converged_bars.png + .pdf
    figures/summary_table.csv
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from scipy import stats

PROJECT = "robhpitkin-personal/comm-vs-ctde-final"
EVAL_KEY = "eval/average_episode_rewards"
NAME_RE = re.compile(r"^(iql|vdn)_(no_comm|with_comm)_seed(\d+)$")

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def fetch_runs() -> pd.DataFrame:
    """Return one row per run with parsed metadata + eval history dict."""
    api = wandb.Api()
    rows = []
    for run in api.runs(PROJECT):
        m = NAME_RE.match(run.name)
        if not m:
            print(f"  skipping {run.name!r} (doesn't match matrix naming)")
            continue
        algo, comm, seed = m.group(1), m.group(2), int(m.group(3))

        # Pull eval reward as a function of env step.
        hist = run.history(keys=[EVAL_KEY], pandas=True)
        if hist.empty:
            print(f"  warning: {run.name} has no eval history yet")
            continue

        rows.append({
            "name": run.name,
            "algo": algo,
            "comm": comm,
            "seed": seed,
            "state": run.state,
            "history": hist.set_index("_step")[EVAL_KEY].dropna(),
        })
    df = pd.DataFrame(rows)
    print(f"\nLoaded {len(df)} runs from {PROJECT}")
    if not df.empty:
        print(df.groupby(["algo", "comm"]).size().rename("n_seeds"))
    return df


def aggregate_seeds(runs: pd.DataFrame) -> pd.DataFrame:
    """For each (algo, comm) group, stack per-seed histories and return
    long-format df with columns [algo, comm, env_step, mean, sem, ci]."""
    out = []
    for (algo, comm), group in runs.groupby(["algo", "comm"]):
        # Align on env_step. Each series is indexed by step; outer-join across seeds.
        wide = pd.concat({r["seed"]: r["history"] for _, r in group.iterrows()}, axis=1)
        wide.columns.name = "seed"
        mean = wide.mean(axis=1)
        sem = wide.sem(axis=1)
        # t-distribution CI: t_{0.975, n-1} * sem (per-step n may vary if seeds aligned imperfectly)
        n_per_step = wide.notna().sum(axis=1)
        t_crit = n_per_step.apply(lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else np.nan)
        ci = sem * t_crit
        for step, mu in mean.items():
            out.append({
                "algo": algo,
                "comm": comm,
                "env_step": step,
                "mean": mu,
                "sem": sem.loc[step],
                "ci": ci.loc[step],
                "n_seeds": int(n_per_step.loc[step]),
            })
    return pd.DataFrame(out)


def plot_learning_curves(agg: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    palette = {"no_comm": "#888888", "with_comm": "#1f77b4"}
    label = {"no_comm": "no-comm", "with_comm": "with-comm"}

    for ax, algo in zip(axes, ["iql", "vdn"]):
        sub = agg[agg["algo"] == algo].sort_values("env_step")
        for comm, gdf in sub.groupby("comm"):
            ax.plot(gdf["env_step"], gdf["mean"],
                    color=palette[comm], label=label[comm], linewidth=2)
            ax.fill_between(gdf["env_step"],
                            gdf["mean"] - gdf["ci"],
                            gdf["mean"] + gdf["ci"],
                            color=palette[comm], alpha=0.2)
        ax.set_title(f"{algo.upper()}")
        ax.set_xlabel("env step")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("eval episode reward (mean ± 95% CI)")
    fig.suptitle("comm-vs-ctde: learning curves (5 seeds per cell)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"learning_curves.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR}/learning_curves.[png|pdf]")


def converged_stats(runs: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Per-seed converged number = mean of last `window` evals.
    Returns one row per (algo, comm, seed)."""
    rows = []
    for _, r in runs.iterrows():
        if len(r["history"]) < window:
            continue
        converged = r["history"].iloc[-window:].mean()
        peak = r["history"].min()  # most negative = best avg reward
        rows.append({
            "algo": r["algo"], "comm": r["comm"], "seed": r["seed"],
            "converged": converged, "peak": peak,
        })
    return pd.DataFrame(rows)


def plot_converged_bars(per_seed: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Custom error bar: SE × t_{0.975, n-1} per cell (matches summary_table semantics)
    def _t_ci(x):
        n = len(x)
        if n <= 1:
            return (x.mean(), x.mean())
        half = stats.sem(x, ddof=1) * stats.t.ppf(0.975, n - 1)
        return (x.mean() - half, x.mean() + half)

    sns.barplot(data=per_seed, x="algo", y="converged", hue="comm",
                errorbar=_t_ci, capsize=0.1, ax=ax,
                palette={"no_comm": "#888888", "with_comm": "#1f77b4"},
                order=["iql", "vdn"], hue_order=["no_comm", "with_comm"])
    sns.stripplot(data=per_seed, x="algo", y="converged", hue="comm",
                  dodge=True, ax=ax, palette="dark:black", alpha=0.6, size=4,
                  order=["iql", "vdn"], hue_order=["no_comm", "with_comm"],
                  legend=False)
    ax.set_ylabel("converged eval reward (mean of last 5 evals per seed)")
    ax.set_xlabel("")
    ax.set_title("comm-vs-ctde: converged performance (5 seeds, 95% t-CI)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"converged_bars.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG_DIR}/converged_bars.[png|pdf]")


def summary_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    g = per_seed.groupby(["algo", "comm"])["converged"]
    # t-distribution CI with n-1 dof per cell
    t_crit = g.size().apply(lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else np.nan)
    tbl = pd.DataFrame({
        "n_seeds": g.size(),
        "mean": g.mean().round(2),
        "sem": g.sem().round(2),
        "ci95_half": (g.sem() * t_crit).round(2),
        "min_seed": g.min().round(2),
        "max_seed": g.max().round(2),
    }).reset_index()
    print("\nConverged performance summary:")
    print(tbl.to_string(index=False))
    out_path = FIG_DIR / "summary_table.csv"
    tbl.to_csv(out_path, index=False)
    print(f"\n  wrote {out_path}")
    return tbl


def main():
    runs = fetch_runs()
    if runs.empty:
        print("No runs to analyze yet.")
        return
    agg = aggregate_seeds(runs)
    plot_learning_curves(agg)
    per_seed = converged_stats(runs)
    if not per_seed.empty:
        plot_converged_bars(per_seed)
        summary_table(per_seed)


if __name__ == "__main__":
    main()
