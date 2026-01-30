"""
Find linear association between cluster count and k (immediate neighbours responded to)
returns Pearson r and p-value
Plots decriptive! trendline
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

df = pd.read_csv("dbscan_cluster_distributions_multi_step.csv")

r, p = pearsonr(df["nearest_x"], df["n_clusters"])

# descriptive trendline
coeffs = np.polyfit(df["nearest_x"], df["n_clusters"], 1)
trendline = np.poly1d(coeffs)

# plot scatter and descriptive trendline
x = np.linspace(
    df["nearest_x"].min(),
    df["nearest_x"].max(),
    100

)

plt.figure(figsize=(7, 5))

plt.scatter(
    df["nearest_x"],
    df["n_clusters"],
    label="Mean clusters"
)

plt.plot(
    x,
    trendline(x),
    linestyle="--",
    label="descriptive trendline"
)

plt.xlabel("k (nearest_x)")
plt.ylabel("Average clusters across runs")

plt.text(
    0.05, 0.95,
    f"Pearson r = {r:.3f}\np = {p:.3e}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.legend()
plt.savefig("linear_association_plot.png", dpi=300)
plt.show()