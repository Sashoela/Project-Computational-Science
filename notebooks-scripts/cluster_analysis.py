"""
Determine clusters present post predator interaction
Visualise cluster distribution using box plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("final_positions_multi_step.csv")

results = []

for x in sorted(df["nearest_x"].unique()):
    print(f"Processing nearest_x = {x}")
    for run in sorted(df["run"].unique()):
        subset = df[
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
        n_noise = np.sum(labels == -1)

        results.append({
            "nearest_x": x,
            "run": run,
            "n_clusters": n_clusters,
            "n_noise": n_noise
        })

results_df = pd.DataFrame(results)
results_df.to_csv("dbscan_cluster_distributions_multi_step.csv", index=False)

print(results_df.head())


# add box plot
plt.figure(figsize=(8, 5))

results_df.boxplot(
    column="n_clusters",
    by="nearest_x",
    grid=False
)

plt.savefig("cluster_boxplot_multi_step.png", dpi=300)

plt.xlabel("Nearest N")
plt.ylabel("Number of clusters")
plt.title("Cluster counts post-predator")
plt.suptitle("")

plt.savefig("cluster_boxplot.png", dpi=300)

plt.show()

#### Bar plot to check single run 

# plt.figure(figsize=(8, 5))

# plt.bar(
#     results_df["nearest_x"],
#     results_df["n_clusters"]
# )

# plt.xlabel("Neighbourhood size (nearest_x)")
# plt.ylabel("Number of clusters")
# plt.title("DBSCAN cluster count post-predator")

# plt.tight_layout()
# plt.show()
