import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from simulation_class import Simulation


def mean_nearest_neighbour_distance(positions: np.ndarray) -> float:
    """
    Mean nearest-neighbour distance r1:
    for each bird, distance to closest other bird, averaged over birds.
    positions: (N, 3)
    """
    N = positions.shape[0]
    nearest = np.empty(N, dtype=float)

    for i in range(N):
        dist = np.linalg.norm(positions - positions[i], axis=1)
        dist[i] = np.inf
        nearest[i] = dist.min()

    return float(nearest.mean())


def density_from_convex_hull(positions: np.ndarray) -> float:
    """
    Calculating density 
    """
    hull = ConvexHull(positions)
    V = hull.volume
    return float(len(positions) / V)


def get_positions(sim: Simulation) -> np.ndarray:
    """Return positions as an array """
    x, y, z = sim.dump()
    return np.column_stack([x, y, z])


def sample_r1_and_density(
    N_birds=200,
    nearest_x=7,
    coh=0.3,
    ali=0.3,
    sep=0.3,
    noise=0.1,
    steps=1200,
    settle_steps=300,
    sample_every=10,
):
   
    sim = Simulation(
        N_birds=N_birds,
        nearest_x=nearest_x,
        coh_vector_scale=coh,
        ali_vector_scale=ali,
        sep_vector_scale=sep,
        noise_vector_scale=noise,
        pred_intro_time=10**9,
        pred_exit_time=10**9,
    )

    r1_list, rho_list = [], []

    for t in range(steps):
        sim.step()

        if t < settle_steps:
            continue
        if t % sample_every != 0:
            continue

        positions = get_positions(sim)
        r1 = mean_nearest_neighbour_distance(positions)
        rho = density_from_convex_hull(positions)

        r1_list.append(r1)
        rho_list.append(rho)
        
    return np.array(r1_list, dtype=float), np.array(rho_list, dtype=float)


def plot_density_vs_r1_inv3(r1: np.ndarray, rho: np.ndarray, outpath="figure5a_style.png"):
    mask = np.isfinite(r1) & np.isfinite(rho) & (r1 > 0) & (rho > 0)
    r1 = r1[mask]
    rho = rho[mask]

    x = r1 ** (-3) 
    y = rho

    m, c = np.polyfit(x, y, 1)
    y_fit = m * x + c

    # R^2
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(x, y, s=55)  

    x_line = np.linspace(x.min(), x.max(), 300)
    ax.plot(x_line, m * x_line + c, linestyle="--", linewidth=2)

    ax.set_xlabel(r"average nearest neighbour distance$^{-3}$")
    ax.set_ylabel(r"density")
    ax.set_title(f"density vs NN distance$^{{-3}}$ (R$^2$={r2:.2f})")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.show()


def plot_r1_by_group(r1: np.ndarray, n_groups: int = 10, outpath="figure5b_style.png"):
    r1 = r1[np.isfinite(r1) & (r1 > 0)]
    chunks = np.array_split(r1, n_groups)
    means = np.array([ch.mean() if len(ch) else np.nan for ch in chunks], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar(np.arange(1, n_groups + 1), means)

    ax.set_xlabel("flock number")
    ax.set_ylabel("average nearest neighbour distance (m)")
    ax.set_title("average nearest neighbour distance (m)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.show()


def main():
    r1, rho = sample_r1_and_density(
        N_birds=200,
        nearest_x=7,
        coh=0.3,
        ali=0.3,
        sep=0.3,
        noise=0.1,
        steps=1200,
        settle_steps=300,
        sample_every=10,
    )

    plot_density_vs_r1_inv3(r1, rho, outpath="density_vs_r1_inv3.png")

    plot_r1_by_group(r1, n_groups=10, outpath="r1_by_flock_index.png")


if __name__ == "__main__":
    main()
