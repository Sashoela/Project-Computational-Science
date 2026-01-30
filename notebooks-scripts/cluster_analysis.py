"""
Determine clusters present post predator interaction
Visualise cluster distribution using box plots
Supports mutiple box plot graphing options (for single run, over mutiple run with one snapshot, over multiple run AND snapshot)
Please comment out the graphing code accordingly 
Is defaulted to reproduced graph we used on the poster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("final_positions_multi_step.csv")

results = []
steps = [301, 311, 321]

for step in steps: 
    print(f"Processing step = {step}")
    for x in sorted(df["nearest_x"].unique()):
        print(f"Processing nearest_x = {x}")
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
            n_noise = np.sum(labels == -1)
    
            results.append({
                "nearest_x": x,
                "run": run,
                "step": step,
                "n_clusters": n_clusters,
                "n_noise": n_noise
            })

results_df = pd.DataFrame(results)
results_df.to_csv("dbscan_cluster_distributions_multi_step.csv", index=False)

print(results_df.head())

df = pd.read_csv("final_positions_multi_step.csv")
df.columns = df.columns.str.strip()

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

# To produced cluster analysis of three t= snapshots

### add box plot for multiple run, single snapshot
# plt.figure(figsize=(8, 5))

# results_df.boxplot(
#     column="n_clusters",
#     by="nearest_k",
#     grid=False
# )

# plt.xlabel("Nearest N")
# plt.ylabel("Number of clusters")
# plt.title("Cluster counts post-predator")
# plt.suptitle("")

# plt.savefig("cluster_boxplot.png", dpi=300)

# plt.show()

### for mutiple run, mutiple snapshots

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
    ax.set_xlabel("nearest_x")
    ax.set_ylabel("number of clusters")

plt.suptitle("Cluster distribution at step 301, 311, 321")
plt.savefig("cluster_boxplot.png", dpi=300)
plt.show()



