import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from simulation_class import Simulation

def mean_nearest_neighbour_distance(positions):
    """
    Computes the mean nearest-neighbour distance r1 over all birds.

    :param positions: array of agent's 3D positions 
    
    Returns: 
        Mean nearest-neighbour distance r1, dtype='float
    """

    N = positions.shape[0]
    nearest = np.empty(N, dtype=float)

    for i in range(N):
        #Compuate distances from bird i to all other birds 
        dist = np.linalg.norm(positions - positions[i], axis=1)
        #ignore self 
        dist[i] = np.inf
        #Nearest neighbour distance for bird i 
        nearest[i] = dist.min()

    #Average nearest-neighbour distance over all birds 
    return float(nearest.mean())


def density_from_convex_hull(positions):
    """
    Estimate flock density using the convex hull volume. 

    Convex hull provides the smallest convex volume enclosing all birds. 
    This is ran without a predator, so assumes that the flock stays intact. 

    :param positions: array of agent's 3D positions 
 
    
    Returns:
        Estimated density rho = N/V, where V is the convex hull volume 
    """
    hull = ConvexHull(positions)
    V = hull.volume

    #Density defined as number of birds per unit volume 
    return float(len(positions) / V)


def get_positions(sim):
    """Return positions as an array """
    x, y, z = sim.dump()

    #stack arrays into columns to align each bird with its position
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
    """
    Run a flocking simulation and sample r1 and density over time. 

    Starting after step 200 to let the flock settle (in order to avoid issues with the convex hull).
    After the settling period, measurements are taken periodically. 

    Returns: 
        Arrays of measurements 
    """

    #Initialising a simulation without predator 
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

    #lists for the nearest neighbour distance and the density at the time 
    r1_list, rho_list = [], []

    for t in range(steps):
        sim.step()

        #do not record pre-settle steps 
        if t < settle_steps:
            continue
        #record every ten steps 
        if t % sample_every != 0:
            continue

        #retrieve positions
        positions = get_positions(sim)
        #record nearest neighbour distance
        r1 = mean_nearest_neighbour_distance(positions)
        #record density 
        rho = density_from_convex_hull(positions)

        #add information to lists 
        r1_list.append(r1)
        rho_list.append(rho)
        
    return np.array(r1_list), np.array(rho_list)


def plot_density_vs_r1_inv3(r1, rho):
    """
    Plot density as a function or r1^{-3} and fit a linear relationship 
    
    :param r1: Nearest-Neighbour distance
    :param rho: Density 
    """

    #Filtering out invalid or unusable values
    mask = np.isfinite(r1) & np.isfinite(rho) & (r1 > 0) & (rho > 0)
    r1 = r1[mask]
    rho = rho[mask]

    #mapping the scaling 
    x = r1 ** (-3) 
    y = rho

    #applying a linear fit 
    m, c = np.polyfit(x, y, 1)
    y_fit = m * x + c

    # Calculating the R2 
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
    fig.savefig("density_vs_r1_inv3.png")
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

    plot_density_vs_r1_inv3(r1, rho)

if __name__ == "__main__":
    main()
