# import libraries
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import our own functions
from agent_class import Agent
class Simulation():
    def __init__(self, N_birds, nearest_x):
        # needed variables
        self.timestep = 0
        self.N_birds = N_birds
        self.nearest_x = nearest_x

        # initialize birds in a list
        self.agents = [Agent(i) for i in range(N_birds)]

        #random normalized speed
        v = np.random.normal(size=3)
        unit_v = v / np.linalg.norm(v)
        # give random location and speed
        for agent in self.agents:
            agent.setup(random.uniform(40, 60), random.uniform(40, 60), random.uniform(40, 60), unit_v[0], unit_v[1], unit_v[2])

    def step(self):
        return
    def show(self):
        i, j, k = [], [], []
        for agent in self.agents:
            x, y, z, vx, vy, vz = agent.output_last()
            i.append(x)
            j.append(y)
            k.append(z)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")
        ax.scatter(i, j, k)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 100)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        plt.show()
    def dump():
        return
    

sim = Simulation(100, 7)
sim.show()