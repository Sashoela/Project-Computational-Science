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
    def __init__(self, N_birds, nearest_x, coh_vector_scale, ali_vector_scale, sep_vector_scale, noise_vector_scale):
        # needed variables
        self.timestep = 0
        self.N_birds = N_birds
        self.nearest_x = nearest_x
        self.coh_vector_scale = coh_vector_scale
        self.ali_vector_scale = ali_vector_scale
        self.sep_vector_scale = sep_vector_scale
        self.noise_vector_scale = noise_vector_scale
        self.predator_area = 50
        self.pred_intro = 100

        # initialize birds in a list
        self.agents = [Agent(i) for i in range(N_birds)]

        # give random location and speed
        for agent in self.agents:
            #random normalized speed
            v = np.random.normal(size=3)
            unit_v = v / np.linalg.norm(v)
            agent.setup(random.uniform(40, 60), random.uniform(40, 60), random.uniform(40, 60), unit_v[0], unit_v[1], unit_v[2])

    # --- Boids Rules ---
    def cohesion(self, agent_index, neighbor_ids):
        if not neighbor_ids:
            return np.zeros(3)
        agent_pos = np.array(self.agents[agent_index].output_last()[:3])
        neighbor_positions = np.array([self.agents[nid].output_last()[:3] for nid in neighbor_ids])
        return neighbor_positions.mean(axis=0) - agent_pos

    def alignment(self, agent_index, neighbor_ids):
        if not neighbor_ids:
            return np.zeros(3)
        neighbor_vels = np.array([self.agents[nid].output_last()[3:6] for nid in neighbor_ids])
        return neighbor_vels.mean(axis=0)

    def separation(self, agent_index, neighbor_ids, separation_distance=5.0):
        if not neighbor_ids:
            return np.zeros(3)
        agent_pos = np.array(self.agents[agent_index].output_last()[:3])
        sep_vec = np.zeros(3)
        for nid in neighbor_ids:
            neighbor_pos = np.array(self.agents[nid].output_last()[:3])
            diff = agent_pos - neighbor_pos
            dist = np.linalg.norm(diff)
            if 0 < dist < separation_distance:
                sep_vec += diff / dist
        return sep_vec

    def step(self):
        # variables used to find nearest birds
        i, j, k, ids = [], [], [], []
        for agent in self.agents:
            x, y, z, vx, vy, vz, d = agent.output_last()
            i.append(x)
            j.append(y)
            k.append(z)
            ids.append(d)

        # predator movement
        if self.timestep == self.pred_intro:
            self.predator = Predator()
        if self.timestep > self.pred_intro:
            predx, predy, predz = self.predator.info()
            # use i, j and k list with all bird info from earlier in the step function
            vector = np.array([0, 0, 0], dtype = np.float64)
            for a in range(len(i)):
                dist = np.sqrt(((predx - i[a]) ** 2) + ((predy - j[a]) ** 2) + ((predz - k[a]) ** 2))
                # if within "range" add vector to this bird to total
                if dist < self.predator_area:
                    vec = np.array([i[a] - predx, j[a] - predy, k[a] - predz], dtype = np.float64)
                    vec = vec / np.linalg.norm(vec)
                    vector += vec
            #normalize and scale vector to speed = 2
            if any(vector):
                movement = vector / np.linalg.norm(vector) * np.sqrt(2)
                #calc new x, y, z and update predator
                self.predator.update(predx + movement[0], predy + movement[1], predz + movement[2])
        self.timestep += 1

        # make a loop to update all agents (loop over all birds)
        for agent in self.agents:
            # data of current bird
            x, y, z, vx, vy, vz, id = agent.output_last()

            #find nearest birds
            nearest_ids = self.nearest_x_ids(i,j,k,ids,self.nearest_x,id)

            #bird reaction to predator vector
            bird_loc=(x,y,z)
            effective_dist=20 #can change (have a look at the papers)
            if self.timestep > self.pred_intro:
                pred_loc = self.predator.info()
                react_pred_vec= self.bird_react_to_predator(bird_loc,pred_loc,effective_dist)
            else : 
                react_pred_vec = np.array([0, 0, 0])

            # four components of speed vector: location, direction, noise ; n = neighbour
            cohesion_vec = self.cohesion(id, nearest_ids)
            cohesion_vec = (cohesion_vec / np.linalg.norm(cohesion_vec)) * self.coh_vector_scale

            alignment_vec = self.alignment(id, nearest_ids)
            alignment_vec = (alignment_vec / np.linalg.norm(alignment_vec)) * self.ali_vector_scale

            seperation_vec = self.separation(id, nearest_ids)
            if any(seperation_vec):seperation_vec = (seperation_vec / np.linalg.norm(seperation_vec)) * self.sep_vector_scale

            noise = np.random.normal(size=3)
            scaled_noise = noise / np.linalg.norm(noise) * self.noise_vector_scale

            #total movement :
            if any(react_pred_vec): total_vec = react_pred_vec + scaled_noise
            else: 
                total_vec = cohesion_vec + alignment_vec + seperation_vec + scaled_noise
                total_vec = total_vec / np.linalg.norm(total_vec)
            ##### all other influences on movement (wall and wind)
            wall = np.array(wall_vec(x, y, z, 5), dtype = np.float64)

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

        if self.timestep > self.pred_intro:
            x, y, z = self.predator.info()
            i.append(x)
            j.append(y)
            k.append(z)

        scat._offsets3d = (i, j, k)
        fig.canvas.draw_idle()
        plt.pause(0.05)           
        

    def dump(self):
        i, j, k = [], [], []
        for agent in self.agents:
            x, y, z, vx, vy, vz, id = agent.output_last()
            i.append(x)
            j.append(y)
            k.append(z)
        return i, j, k
    

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

    def bird_react_to_predator(self,bird_loc,pred_loc,effective_dist):
        x,y,z=bird_loc
        i,j,k=pred_loc
        dist=np.sqrt(
            (x - i)**2 +
            (y - j)**2 +
            (z - k)**2
        )
        if dist==0: return np.array([0,0,0]) # can be also changed to bird dying but for now i just ignore
        dx, dy, dz = x - i, y - j, z - k
        direction = np.array([dx, dy, dz]) / dist   # unit vector
        closeness = 1.0 - (dist / effective_dist)
        beta=5 #we can change this on how wild do we want reaction to be 
        strength = (np.exp(beta * closeness) - 1.0) / (np.exp(beta) - 1.0)
        if dist <= effective_dist:
            min_speed, max_speed = 1.0, 2.0
            speed = min_speed + (max_speed - min_speed) * strength
            return speed * direction
        else: return np.array([0,0,0])
            


# test code
sim = Simulation(400, 7, 0.3, 0.3, 0.3, 0.1)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
scat = ax.scatter([], [], [], s = 5)
for i in range(200):
    sim.step()
    sim.show()
plt.ioff()
plt.show()