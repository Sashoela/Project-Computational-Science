from simulation_class import Simulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

#run simulation
sim = Simulation(200, 7, 0.3, 0.35, 0.25, 0.1, 600, 600)
for _ in range(150):
    sim.step()
x, y, z = sim.dump()


# build hull
points = np.column_stack((x, y, z))
hull = ConvexHull(points)

# fuction that calculates distance to all hull planes and returnes smallest
def distance_to_hull(point, hull):
    return min(
        abs(eq[:3] @ point + eq[3]) / np.linalg.norm(eq[:3])
        for eq in hull.equations
    )


# #center of mass:
# avg_x = (max(x) + min(x)) / 2
# avg_y = (max(y) + min(y)) / 2
# avg_z = (max(z) + min(z)) / 2

#check distance to border and density for each agent for a number of simulations
dist_to_border = []
density = []


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
    #dist to border
    to_border = distance_to_hull(points[n], hull)

    dist_to_border.append(to_border)

#plot and fit
a, b = np.polyfit(dist_to_border, density, 1)
plt.scatter(dist_to_border, density)
plt.plot(dist_to_border, a * np.array(dist_to_border) + b, 'r-', label=f"best fit, a = {a}")
plt.xlabel("distance to border")
plt.ylabel("local density")
plt.legend(loc = "best")
plt.show()