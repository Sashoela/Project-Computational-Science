import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from simulation_class import Simulation


def compute_r1(positions: np.ndarray):
    """
    Mean nearest-neighbour distance r1:
    for each bird, distance to closest other bird, averaged over birds.
    positions: (N, 3)
    """
    N = positions.shape[0]
    nearest = np.empty(N, dtype=float)

    for bird in range(N):
        dist = np.linalg.norm(positions - positions[bird], axis=1)
        dist[bird] = np.inf
        nearest[bird] = dist.min()

    return float(nearest.mean())


def compute_density_proxy(positions: np.ndarray):
    """
    Density proxy rho = N / V where V is convex hull volume.
    positions: (N, 3)
    """
    hull = ConvexHull(positions)
    V = hull.volume
    return float(len(positions) / V)


def dump_positions(sim: Simulation):
    """Return positions as array shape (N, 3)."""
    i, j, k = sim.dump()
    return np.column_stack([i, j, k])


def single_snapshot_check(sim: Simulation):

    for _ in range(400):
        sim.step()

    pos = dump_positions(sim)

    print("Ran steps:", sim.timestep)
    print("Number of birds:", pos.shape[0])
    print("Example bird 0 position:", pos[0, 0], pos[0, 1], pos[0, 2])
    print("Min/Max x:", pos[:, 0].min(), pos[:, 0].max())
    print("Min/Max y:", pos[:, 1].min(), pos[:, 1].max())
    print("Min/Max z:", pos[:, 2].min(), pos[:, 2].max())

    r1 = compute_r1(pos)
    rho = compute_density_proxy(pos)

    print("Mean nearest-neighbour distance r1:", r1)
    print("Density proxy rho = N/V:", rho)


def collect_time_series(
    N_birds=200,
    nearest_x=7,
    coh=0.3,
    ali=0.3,
    sep=0.3,
    noise=0.1,
    steps=1000,
    settle_steps=200,
    sample_every=10,
):
    """
    Run a simulation and collect (r1, rho) samples over time.
    Predator is disabled by setting pred_intro_time far in the future.
    Returns: r1_list, rho_list
    """
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

    r1_list = []
    rho_list = []

    for t in range(steps):
        sim.step()

        if t < settle_steps:
            continue
        if (t % sample_every) != 0:
            continue

        pos = dump_positions(sim)
        r1_list.append(compute_r1(pos))
        rho_list.append(compute_density_proxy(pos))

    return np.array(r1_list, dtype=float), np.array(rho_list, dtype=float)


def plot_and_fit(r1: np.ndarray, rho: np.ndarray):
    """Make log-log scatter, fit slope, print diagnostics."""
    mask = np.isfinite(r1) & np.isfinite(rho) & (r1 > 0) & (rho > 0)
    r1 = r1[mask]
    rho = rho[mask]

    x = np.log(r1)
    y = np.log(rho)

    slope, intercept = np.polyfit(x, y, 1)

    print("Collected", len(r1), "data points")
    print("r1 range:", float(r1.min()), float(r1.max()))
    print("rho range:", float(rho.min()), float(rho.max()))
    print("Fit: log(rho) = slope*log(r1) + intercept")
    print("slope =", float(slope), "(paper expects about -3)")

    const_check = rho * (r1 ** 3)
    print("rho * r1^3 mean:", float(const_check.mean()))
    print("rho * r1^3 min/max:", float(const_check.min()), float(const_check.max()))

    # Plot log-log scatter with fitted line
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=12)
    ax.set_xlabel("log(r1)")
    ax.set_ylabel("log(rho)")
    ax.set_title(f"Log–log density vs NN distance (slope = {slope:.2f})")

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, linewidth=2)

    fig.tight_layout()
    fig.savefig("loglog_density_vs_r1.png", dpi=200)
    print("Saved plot to loglog_density_vs_r1.png")

    plt.show(block=True)


def main():
    # --- Optional sanity check (uncomment if you want it) ---
    # sim_check = Simulation(200, 7, 0.3, 0.3, 0.3, 0.1, 10**9, 10**9)
    # single_snapshot_check(sim_check)

    # --- Collect time-series samples ---
    r1, rho = collect_time_series(
        N_birds=200,
        nearest_x=7,
        coh=0.3,
        ali=0.3,
        sep=0.3,
        noise=0.1,
        steps=1000,
        settle_steps=200,
        sample_every=10,
    )

    # --- Plot + fit ---
    plot_and_fit(r1, rho)


if __name__ == "__main__":
    main()
