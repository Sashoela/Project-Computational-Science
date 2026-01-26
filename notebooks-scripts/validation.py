from simulation_class import Simulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import alphashape
import trimesh
from sklearn.neighbors import NearestNeighbors

def auto_alpha(points, k=6, scale=1.5):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    dists, _ = nbrs.kneighbors(points)
    mean_nn_dist = dists[:,1:].mean()  # skip self distance
    return scale / mean_nn_dist

def inv_model(x, a):
    return a / x

#check distance to border and density for each agent for a number of simulations
dist_to_border = []
density = []

for i in range(10):
    #run simulation
    x, y, z = 0, 0, 0
    sim = Simulation(200, 7, 0.3, 0.35, 0.25, 0.1, 600, 600)
    for _ in range(150):
        sim.step()
    x, y, z = sim.dump()


    # # build hull
    points = np.column_stack((x, y, z))
    # hull = ConvexHull(points)

    alpha = auto_alpha(points)
    shape = alphashape.alphashape(points, alpha)
    mesh = trimesh.Trimesh(vertices=shape.vertices, faces=shape.faces)
    #distances to mesh
    distances = np.abs(mesh.nearest.signed_distance(points))

    for n in range(len(x)): # n is current bird
        #check against others
        dens = 0
        for m in range(len(x)):
            if n != m:
                d = np.sqrt(
                    (x[n] - x[m])**2 +
                    (y[n] - y[m])**2 +
                    (z[n] - z[m])**2
                )
                if d <= 2:
                    dens += 1
        density.append(dens)
        dist_to_border.append(distances[n])

#plot and fit
dist_to_border = np.array(dist_to_border)
density = np.array(density)
mask = dist_to_border > 0
params, cov = curve_fit(inv_model, dist_to_border[mask], density[mask])
a = params[0]
x_fit = np.linspace(dist_to_border.min(), dist_to_border.max(), 300)
y_fit = inv_model(x_fit, a)

# plt.scatter(dist_to_border, density, alpha = 0.2)
plt.hexbin(dist_to_border, density, gridsize=50, cmap="viridis")
plt.colorbar(label="count")
#plt.plot(dist_to_border, best_fit.slope * np.array(dist_to_border) + best_fit.intercept, 'r-', label=f"best fit, a = {best_fit.slope}")
plt.plot(x_fit, y_fit, color="red", lw=2, label=r"fit: $a/x$")
plt.xlabel("distance to border")
plt.ylabel("local density")
plt.ylim(0, max(density))
plt.legend(loc = "best")
#plt.title(f"r^2 = {best_fit.rvalue**2}")

plt.savefig("density_validation3.png")