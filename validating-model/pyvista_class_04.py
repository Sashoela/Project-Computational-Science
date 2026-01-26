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



class PyVistaViewer: 
    def __init__(self, sim):
        self.sim = sim 

        # --- Birds --- 

        #initial positions and velocities 
        positions = np.array([agent.output_last()[:3] for agent in sim.agents])
        #Create a point cloud
        self.cloud = pv.PolyData(positions)

        # --- Plotter ---

        # Birds 
        self.plotter = pv.Plotter()
        self.plotter.add_axes()
        self.plotter.set_background("black")

        self.actor = self.plotter.add_points(
            self.cloud,
            render_points_as_spheres=True,
            point_size=6,
            color="white"
        )
        self.plotter.show(interactive_update=True)

    def update(self):
        #Update positions 
        new_positions = np.array([agent.output_last()[:3] for agent in self.sim.agents])

        self.cloud.points = new_positions

        self.plotter.update()