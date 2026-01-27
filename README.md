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

### `simulation_class.py`
Implements and runs the simulation:
- initializes N birds
- uses 3 boids rules and small noise component to calculate new position for all bird agents in step function
- predator agent movement towards nearest sizeable cluster of bird agents in step function
- includes dump function to drop all bird agents coordinates
- simple show function only used for testing

---

### `validation_dens_aggr.py`
program to validate density compared to distance from flock edge:
- runs n simulations, at end of each simulation a concave hull is calculated (edge of flock).
- calculates distance to edge of the flock and local density for each bird
- fits this to a inverse relation
- plots in a 2D hexbin (large numer of datapoint)

---

### `wall.py`
returns vector away from "wall" if near wall.
- location input
- returns vector away from wall when near wall

---

EVERYTHING ELSE TBA !

