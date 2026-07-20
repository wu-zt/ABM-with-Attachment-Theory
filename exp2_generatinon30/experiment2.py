"""
Analyze multi-generation evolution results (disaster_duration_test, 30 generations)
Run this script in the same folder as the three exported CSV files:
    pip install pandas matplotlib
    python analyze_multigen.py

Note: this dataset's header does not include a "generation" column (the
BehaviorSpace reporter wasn't checked), but since epoch-duration is fixed
within each run, the generation number can be derived directly:
    generation = floor(step / epoch_duration) + 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TABLE_FILE = "AT_disaster_duration_test-table.csv"

df = pd.read_csv(TABLE_FILE, skiprows=6)
df.columns = ["run", "initial_pop", "epoch_duration", "step",
              "green", "yellow", "blue", "red"]

COLORS = {"green": "#639922", "yellow": "#eda100", "blue": "#2a78d6", "red": "#e34948"}
LABELS = {"green": "Secure", "yellow": "Dismissing-Avoidant",
          "blue": "Preoccupied", "red": "Fearful-Avoidant"}
DURATIONS = sorted(df.epoch_duration.unique())

# Derive generation number and "tick within the current epoch"
df["generation"] = (df["step"] // df["epoch_duration"]) + 1
df["local_tick"] = df["step"] % df["epoch_duration"]
df["total_pop"] = df[["green", "yellow", "blue", "red"]].sum(axis=1)


# =====================================================================
# Analysis 1: core question - how phenotype proportions evolve over generations
# =====================================================================
# Take the "end of epoch" snapshot for each generation (last tick before
# reproduction happens), i.e. rows where step is an exact multiple of epoch_duration
gen_end = df[df["step"] % df["epoch_duration"] == 0].copy()
gen_end = gen_end[gen_end["step"] > 0]  # drop step=0 (initial population, not an "end" snapshot)
gen_end["generation"] = gen_end["step"] // gen_end["epoch_duration"]

# Convert to percentage (reproduction resets total population, but survivor
# counts at epoch end differ, so percentage is the fairer comparison)
for c in COLORS:
    gen_end[f"{c}_pct"] = gen_end[c] / gen_end["total_pop"] * 100

# Group by epoch_duration + generation, average across the 10 repetitions
gen_summary = gen_end.groupby(["epoch_duration", "generation"])[
    [f"{c}_pct" for c in COLORS] + ["total_pop"]
].mean().reset_index()

fig1, axes1 = plt.subplots(1, len(DURATIONS), figsize=(6 * len(DURATIONS), 4.5), sharey=True)
for ax, ed in zip(axes1, DURATIONS):
    sub = gen_summary[gen_summary.epoch_duration == ed]
    for c in COLORS:
        ax.plot(sub["generation"], sub[f"{c}_pct"], color=COLORS[c],
                 marker="o", markersize=3, label=LABELS[c])
    ax.set_title(f"epoch-duration = {ed}")
    ax.set_xlabel("Generation")
axes1[0].set_ylabel("Proportion (%), averaged across 10 repetitions")
axes1[0].legend()
plt.tight_layout()
plt.savefig("evolution_trend_by_generation.png", dpi=150)
print("Saved evolution_trend_by_generation.png")

# Also check total surviving population at the end of each epoch
# (is population size itself stable across generations?)
fig1b, ax1b = plt.subplots(figsize=(7, 4.5))
for ed in DURATIONS:
    sub = gen_summary[gen_summary.epoch_duration == ed]
    ax1b.plot(sub["generation"], sub["total_pop"], marker="o", markersize=3, label=f"duration={ed}")
ax1b.set_xlabel("Generation")
ax1b.set_ylabel("Total survivors at end of epoch (avg across 10 repetitions)")
ax1b.legend()
plt.tight_layout()
plt.savefig("total_population_by_generation.png", dpi=150)
print("Saved total_population_by_generation.png")

print("\n=== Generation 1 vs. final generation: phenotype proportions ===")
for ed in DURATIONS:
    sub = gen_summary[gen_summary.epoch_duration == ed]
    first = sub[sub.generation == sub.generation.min()]
    last = sub[sub.generation == sub.generation.max()]
    print(f"\n--- epoch_duration={ed} ---")
    print("Generation 1:", {c: round(first[f"{c}_pct"].values[0], 1) for c in COLORS})
    print(f"Generation {int(last.generation.values[0])}:",
          {c: round(last[f"{c}_pct"].values[0], 1) for c in COLORS})


# =====================================================================
# Analysis 2: within-epoch survival curve shape, early vs. late generations
# =====================================================================
# Compare a few representative generations: gen 1 (no selection pressure yet),
# a mid-point, and the final generation (selection has acted for a long time)
GEN_SAMPLES = [1, 15, 30]

fig2, axes2 = plt.subplots(1, len(DURATIONS), figsize=(6 * len(DURATIONS), 4.5), sharey=False)
for ax, ed in zip(axes2, DURATIONS):
    sub_ed = df[df.epoch_duration == ed]
    for g in GEN_SAMPLES:
        sub_gen = sub_ed[sub_ed.generation == g]
        # average across the 10 runs, grouped by tick-within-epoch
        curve = sub_gen.groupby("local_tick")["total_pop"].mean()
        ax.plot(curve.index, curve.values, label=f"Generation {g}", linewidth=2)
    ax.set_title(f"epoch-duration = {ed}")
    ax.set_xlabel("Tick within epoch (local_tick)")
axes2[0].set_ylabel("Total survivors (avg across 10 repetitions)")
axes2[0].legend()
plt.tight_layout()
plt.savefig("intra_epoch_curve_early_vs_late_gen.png", dpi=150)
print("\nSaved intra_epoch_curve_early_vs_late_gen.png")

# Same comparison, but broken down by phenotype for epoch_duration=400 only,
# since that was the condition where blue/red went extinct within a single
# generation in the earlier (single-generation) experiment - most worth
# checking whether evolution has changed that outcome
ed_focus = max(DURATIONS)
fig3, axes3 = plt.subplots(1, len(GEN_SAMPLES), figsize=(6 * len(GEN_SAMPLES), 4.5), sharey=True)
sub_focus = df[df.epoch_duration == ed_focus]
for ax, g in zip(axes3, GEN_SAMPLES):
    sub_gen = sub_focus[sub_focus.generation == g]
    for c in COLORS:
        curve = sub_gen.groupby("local_tick")[c].mean()
        ax.plot(curve.index, curve.values, color=COLORS[c], label=LABELS[c])
    ax.set_title(f"Generation {g} (duration={ed_focus})")
    ax.set_xlabel("local_tick")
axes3[0].set_ylabel("Population")
axes3[0].legend()
plt.tight_layout()
plt.savefig(f"phenotype_curves_early_vs_late_gen_duration{ed_focus}.png", dpi=150)
print(f"Saved phenotype_curves_early_vs_late_gen_duration{ed_focus}.png")

plt.show()
print(gen_summary.groupby("epoch_duration")["generation"].nunique())