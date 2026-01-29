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
XX
- x

---

### `collect_data.py`
XX
- x

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
XX
- x

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
XX
- x

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