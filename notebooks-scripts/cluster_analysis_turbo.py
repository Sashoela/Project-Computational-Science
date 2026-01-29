import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("final_positions_multi_step_sasha_1.csv")
df.columns = df.columns.str.strip()  # fixes hidden spaces in headers

results = []

# looping over steps 
for step in sorted(df["step"].unique()):
    print(f"\n=== Processing step = {step} ===")
    df_step = df[df["step"] == step]

    for x in sorted(df_step["nearest_x"].unique()):
        print(f"  Processing nearest_x = {x}")
        for run in sorted(df_step["run"].unique()):

            subset = df_step[
                (df_step["nearest_x"] == x) &
                (df_step["run"] == run)
            ]

            if len(subset) < 3:
                continue

            X = subset[["x", "y", "z"]].values
            X = StandardScaler().fit_transform(X)

            labels = DBSCAN(eps=0.8, min_samples=3).fit(X).labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))

            results.append({
                "step": step,
                "nearest_x": x,
                "run": run,
                "n_clusters": n_clusters,
                "n_noise": n_noise
            })

results_df = pd.DataFrame(results)
results_df.to_csv("dbscan_cluster_distributions_multi_step_sasha_1.csv", index=False)
print(results_df.head())

# ---- boxplot per step (same style as your original) ----
for step in sorted(results_df["step"].unique()):
    step_df = results_df[results_df["step"] == step]

    plt.figure(figsize=(8, 5))
    step_df.boxplot(
        column="n_clusters",
        by="nearest_x",
        grid=False
    )

    plt.xlabel("Nearest N")
    plt.ylabel("Number of clusters")
    plt.title(f"Cluster counts post-predator (step {step})")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"cluster_boxplot_step_{step}.png", dpi=300)
    plt.show()
