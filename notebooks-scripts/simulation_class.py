# import libraries
import math
import random
# import our own functions
from agent_class import agent

class Simulation():
    def __init__(self, N_birds):
        # needed variables
        self.timestep = 0
        self.N_birds = N_birds

        # initialize birds in a list
        self.agents = [agent(i) for i in range(N_birds)]

        #random normalized speed
        v = [random.gauss(0, 1) for _ in range(3)]
        norm = math.sqrt(sum(x*x for x in v))
        unit_v = [x / norm for x in v]
        # give random location and speed
        for agent in self.agents:
            agent.setup(random.uniform(45, 55), random.uniform(45, 55), random.uniform(45, 55), unit_v[0], unit_v[1], unit_v[2])

    def step():
        return
    def show():
        return
    def dump():
        return