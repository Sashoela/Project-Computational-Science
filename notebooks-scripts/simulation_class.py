# import libraries
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import our own functions
from agent_class import Agent, Predator
from wall import wall_vec
class Simulation():
    def __init__(self, N_birds, nearest_x, loc_vector_scale, dir_vector_scale, noise_vector_scale):
        # needed variables
        self.timestep = 0
        self.N_birds = N_birds
        self.nearest_x = nearest_x
        self.loc_vector_scale = loc_vector_scale
        self.dir_vector_scale = dir_vector_scale
        self.noise_vector_scale = noise_vector_scale

        # initialize birds in a list
        self.agents = [Agent(i) for i in range(N_birds)]

        # give random location and speed
        for agent in self.agents:
            #random normalized speed
            v = np.random.normal(size=3)
            unit_v = v / np.linalg.norm(v)
            agent.setup(random.uniform(40, 60), random.uniform(40, 60), random.uniform(40, 60), unit_v[0], unit_v[1], unit_v[2])

    def step(self):
        # variables used to find nearest birds
        i, j, k, ids = [], [], [], []
        for agent in self.agents:
            x, y, z, vx, vy, vz, d = agent.output_last()
            i.append(x)
            j.append(y)
            k.append(z)
            ids.append(d)

        # make a loop to update all agents (loop over all birds)
        for agent in self.agents:
            # data of current bird
            x, y, z, vx, vy, vz, id = agent.output_last()
            #find nearest birds
            nearest_ids = self.nearest_x_ids(i,j,k,ids,self.nearest_x,id)

            # three components of speed vector: location, direction, noise ; n = neighbour
            total_loc_vector = np.array([0, 0, 0], dtype=np.float64) #x, y, z
            total_direction_vector = np.array([0, 0, 0], dtype=np.float64) #x, y, z
            for id in nearest_ids:
                nx, ny, nz, nvx, nvy, nvz, nid = self.agents[id].output_last()
                total_loc_vector[0] += nx - x
                total_loc_vector[1] += ny - y
                total_loc_vector[2] += nz - z
                total_direction_vector[0] += nvx
                total_direction_vector[1] += nvy
                total_direction_vector[2] += nvz
            loc_vec = total_loc_vector / self.nearest_x
            direction_vec = total_direction_vector / self.nearest_x

            # these vectors need to be normalised and given their required scale afctor
            scaled_loc_vec = (loc_vec / np.linalg.norm(loc_vec)) * self.loc_vector_scale 
            scaled_dir_vec = (direction_vec / np.linalg.norm(direction_vec)) * self.dir_vector_scale
            # noise vector
            noise = np.random.normal(size=3)
            scaled_noise = noise / np.linalg.norm(noise) * self.noise_vector_scale

            #total movement :
            total_vec = scaled_loc_vec + scaled_dir_vec + scaled_noise
            #print(total_vec, type(total_vec[0]))
            ##### all other influences on movement (wall and wind)
            wall = np.array(wall_vec(x, y, z, 5))

            total_vec += wall 

            # assign new loc and speed to agent
            agent.set_current(x + total_vec[0], y + total_vec[1], z + total_vec[2], total_vec[0], total_vec[1], total_vec[2])
        
        # all agents now calculated a new location and speed -> step calculations done -> "current" to "last" for next step
        for agent in self.agents:
            agent.current_to_last()


    def show(self):
        i, j, k = [], [], []
        for agent in self.agents:
            x, y, z, vx, vy, vz, id = agent.output_last()
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

        plt.show(block=False)   # non-blocking
        plt.pause(0.2)          # keep the window open for 0.2s
        plt.close()

    def dump():
        return
    

    def nearest_x_ids(self, i, j, k, ids, near_x, initial_bird):
        n = initial_bird
        nr = len(i)
        items = []  # (distance, id)
        
        for m in range(nr):
            if m == n:
                continue
            d = np.sqrt(
                (i[n] - i[m])**2 +
                (j[n] - j[m])**2 +
                (k[n] - k[m])**2
            )
            items.append((d, ids[m]))
        smallest_near_x = sorted(items, key=lambda x: x[0])[:near_x]

        return [id_ for _, id_ in smallest_near_x]

# test code
sim = Simulation(100, 7, 0.45, 0.45, 0.1)
for i in range(100):
    sim.step()
    sim.show()