# import libraries
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
# import our own functions
from agent_class_01 import Agent
from predator_class_02 import Predator
from wall_03 import wall_vec
from pyvista_class_04 import *
from simulation_class_00 import *

# Initialize simulation
sim = Simulation(
    N_birds=500,
    nearest_neighbors=7,
    cohesion_scale=1,
    alignment_scale = 1,
    separation_scale = 1,
    noise_scale=0.3
)

viewer = PyVistaViewer(sim)

for _ in range(500):
    sim.step()
    viewer.update()