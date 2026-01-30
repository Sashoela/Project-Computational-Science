# Project-Computational-Science

# Starling Murmuration Simulation

This project investigates collective behaviour in starling murmurations using
agent-based modelling and computational simulations.

The project is developed primarily using Jupyter notebooks and follows the
structure of a computational science workflow: model definition, simulation,
analysis, and results.

---
## For required libaries and version see "requirements.txt"

##### How to reproduce figure from poster:  #####

## File Structure
.
├── notebooks-scripts/
│ ├── agent_class.py
│ ├── cluster_analysis.py
│ ├── collect_data.py
│ ├── simulation_class_copy.py
│ ├── simulation_class.py
│ ├── turbo_data_collector.py
│ ├── validation_density_and_NN.py
│ ├── validation_dens_aggr.py
│ ├── validation_dimensions_ratios.py
│ ├── visualisation_arrows.py
│ ├── visualisation_spheres.py
│ └── wall.py
├── results/images/data
├── requirements.txt
└── README.md

---

### `agent_class.py`
Defines the bird and predator agents:
- agent class; saves location and speed of agent. has functions to output and change these.
- predator class; only saves location. functions to output and change these.
No full simulations are run in this notebook.

---

### `cluster_analysis.py`
This script identifies clusters after the predator interaction.
- It uses DBSCAN to count how many clusters form for each neighbourhood size.
- The results are compared across runs.
- It saves a CSV file with cluster counts and a box plot showing the cluster distribution for each nearest_x.

---

### `collect_data.py`
This file runs the simulation many times and print out the results athe last step.
- It changes nearest_x from 2 to 11 and repeats each setting 30 times.
- After 601 steps it saves the final  (x,y,z) position of every bird.
- It writes final_positions.csv with columns nearest_x, run, agent_id, x, y, z for DBSCAN analysis.
---

### `simulation_class.py` and `simulation_class_copy.py`
Implements and runs the simulation:
- Initialisation of bird populations
- Time-stepping loop
- Application of boid rules
- predator movement
- Visualisation of flock behaviour over time for test purposes
- copy adds red dot for predator in test visualization
---

### `turbo_data_collector`
This file runs the flocking simulation multiple times.
- It changes the interaction range (nearest_x) and repeats each run several times.
- At selected timesteps, it saves the positions of all birds.
- The output is a CSV file containing bird positions (x,y,z) for each run, timestep, and number of nearest neighbours.
---

### `validation_density_and_NN.py`
XX
- x

---

### `validation_dens_aggr.py`
Validation checks if local density is the highest at the edge of the flock
- runs x simulations for 150 steps, enough to settle
- for each simulation calculate concave hull, then calculates distance to hull and local density for each agent
- scatter plot of all data points is made with a fit of function a/x

---

### `validation_dimensions_ratios.py`
This is validation of flocks dimensions and it is done by finding out their ratios confidence intervals.
- Simulations are run and the flock shape is described by three dimensions I1<I2<I3.
- These are computed from the spatial extent of the flock along its principal axes.
- The ratios I3/I1 and I2/I1 measure how thin the flock is.
- The ratios are calculated only when the flock is not turning, and the code outputs their mean values with 95% confidence intervals.
---

### `visualisation_arrows.py`
XX
- x

---

### `visualisation_spheres.py`
XX
- x

---

### `wall.py`
Returnes vector away from "wall" when closer then 10
- input = coordinate and output = vector (can be 0, 0, 0)

---

### results/images/data

### `cluster_boxplot_step (301,311,321).png` 
These box plots show the distribution of the number of clusters across runs.
- Each box corresponds to a different nearest_x value.
- Steps 301, 311, and 321 are plotted separately to compare cluster stability over time.

---

### `dbscan_cluster_distributions_multi_step.csv` / `dbscan_cluster_distributions_multi_step_sasha_1.csv`
This file stores the results of the DBSCAN clustering analysis.
- Each row corresponds to one simulation run at a given timestep and neighbourhood size.

---

### `density_validation.png`
XX
- x

---

### `density_vs_r1_inv3.png`
XX
- x

---

### `final_positions_multi_step.csv` / `final_positions_multi_step_sasha_1.csv`
This file stores bird positions at selected timesteps.
- Each row corresponds to one bird in one simulation run.
- The data is used to analyse flock structure and clustering over time.

---

### `final_positions.csv`
XX
- x

---