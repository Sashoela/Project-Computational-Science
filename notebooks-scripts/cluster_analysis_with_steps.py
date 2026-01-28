import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("final_positions_multi_step.csv")

# Identify the three snapshot steps
steps = sorted(df["step"].unique())
print("Snapshot steps:", steps)

# ----------------------------
# Run DBSCAN clustering
# ----------------------------
results = []

for step in steps:
    for x in sorted(df["nearest_x"].unique()):
        for run in sorted(df["run"].unique()):

            subset = df[
                (df["step"] == step) &
                (df["nearest_x"] == x) &
                (df["run"] == run)
            ]

            if len(subset) < 3:
                continue

            X = subset[["x", "y", "z"]].values
            X = StandardScaler().fit_transform(X)

            db = DBSCAN(eps=0.152, min_samples=3).fit(X)
            labels = db.labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            results.append({
                "step": step,
                "nearest_x": x,
                "run": run,
                "n_clusters": n_clusters
            })

results_df = pd.DataFrame(results)

# ----------------------------
# Box plots: one per snapshot
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, step in zip(axes, steps):
    subset = results_df[results_df["step"] == step]

    subset.boxplot(
        column="n_clusters",
        by="nearest_x",
        ax=ax,
        grid=False
    )

    ax.set_title(f"Step {int(step)}")
    ax.set_xlabel("Neighbourhood size (nearest_x)")
    ax.set_ylabel("Number of clusters")

plt.suptitle("Cluster distribution vs neighbourhood size at different snapshots")
plt.tight_layout()
plt.show()

# ----------------------------
# Global trendline (all snapshots combined)
# ----------------------------
trend_data = (
    results_df
    .groupby("nearest_x")["n_clusters"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(7, 5))

plt.scatter(
    trend_data["nearest_x"],
    trend_data["n_clusters"],
    label="Mean cluster count (all snapshots)"
)

coeffs = np.polyfit(
    trend_data["nearest_x"],
    trend_data["n_clusters"],
    1
)
trend = np.poly1d(coeffs)

x_vals = np.linspace(
    trend_data["nearest_x"].min(),
    trend_data["nearest_x"].max(),
    100
)

plt.plot(
    x_vals,
    trend(x_vals),
    linestyle="--",
    label="Linear trend"
)

plt.xlabel("Neighbourhood size (nearest_x)")
plt.ylabel("Mean number of clusters")
plt.title("Overall effect of neighbourhood size on clustering")
plt.legend()
plt.tight_layout()
plt.show()
