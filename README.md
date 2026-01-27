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
├── pyvisa/
├── notebooks-scripts/
│ ├── agent_class.py
│ ├── simulation_class.py
│ ├── validation_dens_aggr.py
│ └── wall.py
├── results/images
├── requirements.txt
└── README.md

---

### `agent_class.py`
Defines the bird and predator agents:
- agent class; saves location and speed of agent. has functions to output and change these.
- predator class; only saves location. functions to output and change these.
No full simulations are run in this notebook.

---

### `02_simulation.ipynb`
Implements and runs the simulation:
- Initialisation of bird populations
- Time-stepping loop
- Application of interaction rules
- Visualisation of flock behaviour over time

---

### `04_analysis.ipynb`
Analyses simulation outputs using statistical methods:
*add later*

Produces quantitative results and plots.

---

### `05_results.ipynb`
Presents and interprets the results:
- Key figures and visualisations
- Summary of observed behaviours
- Discussion of findings in relation to literature

This notebook focuses on interpretation rather than implementation.

---

EVERYTHING ELSE TBA !

