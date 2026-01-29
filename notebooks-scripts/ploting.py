"""
Determine clusters present post predator interaction (multiple timesteps)
Visualise cluster distribution using box plots

Input CSV columns:
nearest_x,run,step,agent_id,x,y,z
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ------------------- SETTINGS -------------------
INPUT_CSV = INPUT_CSV = r"C:\Users\vilke\OneDrive\Dokumentai\GitHub\Project-Computational-Science\final_positions_multi_step.csv"
OUTPUT_CSV = "dbscan_cluster_distributions_multi_step.csv"

EPS = 0.25
MIN_SAMPLES = 3

# If you want to restrict to only some steps, set e.g. [301,311,321]
STEPS_TO_USE = None  # or [301, 311, 321]
# ------------------------------------------------


df = pd.read_csv(INPUT_CSV)

# make sure these are ints (your file may store them as floats)
df["nearest_x"] = df["nearest_x"].astype(int)
df["run"] = df["run"].astype(int)
df["step"] = df["step"].astype(int)

if STEPS_TO_USE is not None:
    df = df[df["step"].isin(STEPS_TO_USE)].copy()

results = []

for nearest_x in sorted(df["nearest_x"].unique()):
    print(f"Processing nearest_x = {nearest_x}")
    for run in sorted(df["run"].unique()):
        for step in sorted(df["step"].unique()):
            subset = df[
                (df["nearest_x"] == nearest_x) &
                (df["run"] == run) &
                (df["step"] == step)
            ]

            if len(subset) < MIN_SAMPLES:
                continue

            X = subset[["x", "y", "z"]].values
            X = StandardScaler().fit_transform(X)

            db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)
            labels = db.labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))

            results.append({
                "nearest_x": nearest_x,
                "run": run,
                "step": step,
                "n_clusters": n_clusters,
                "n_noise": n_noise
            })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print("\nSaved:", OUTPUT_CSV)
print(results_df.head())

# ------------------- BOXPLOTS -------------------
# Recommended: separate plot for each step
for step in sorted(results_df["step"].unique()):
    step_df = results_df[results_df["step"] == step]

    plt.figure(figsize=(8, 5))
    step_df.boxplot(column="n_clusters", by="nearest_x", grid=False)

    plt.xlabel("Nearest N (nearest_x)")
    plt.ylabel("Number of clusters")
    plt.title(f"DBSCAN cluster counts post-predator (step={step})")
    plt.suptitle("")

    out_png = f"cluster_boxplot_step_{step}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print("Saved:", out_png)
    plt.show()

# Optional: combined plot (all steps pooled together)
# (Interpretation: mixes timepoints, so use only if you really want one figure)
plt.figure(figsize=(8, 5))
results_df.boxplot(column="n_clusters", by="nearest_x", grid=False)
plt.xlabel("Nearest N (nearest_x)")
plt.ylabel("Number of clusters")
plt.title("DBSCAN cluster counts post-predator (all steps pooled)")
plt.suptitle("")
plt.tight_layout()
plt.savefig("cluster_boxplot_all_steps_pooled.png", dpi=300)
print("Saved: cluster_boxplot_all_steps_pooled.png")
plt.show()
