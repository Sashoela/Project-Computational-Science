# --- Imports ---
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv 

from agent_class import Agent, Predator 
from wall_fix import wall_vec

"""
Fixing Problems: 

Issue with the wall, imported from wall_fix. Keeping the original wall script. 

Issue with cohesion. Birds did not previously want to be in one big flock. Fix: forward vision. 
Forward vision added in the nearest_x_ids function

Issue with the cohesion function: when added forward vision, the birds with the old cohesion function would just go in one line, which was weird. 
By normalising the vector, we can get proper flocks. 

Added functions:

Predator Toggle: now you can easily toggle on and off the predator in the sim part. Helps with testing. I had to mainly do it so I could actually make the birds flock. 

"""

# --- PyVista Viewer ---
class PyVistaViewer:
    def __init__(self, sim):
        self.sim = sim
        self.plotter = pv.Plotter()
        self.plotter.add_axes()
        self.plotter.set_background("black")

        # Birds
        positions = np.array([agent.output_last()[:3] for agent in sim.agents])
        self.cloud = pv.PolyData(positions)
        self.actor = self.plotter.add_points(
            self.cloud, render_points_as_spheres=True, point_size=6, color="white"
        )

        # Predator
        self.predator_mesh = pv.Sphere(radius=2.0)
        self.predator_actor = self.plotter.add_mesh(self.predator_mesh, color="red")
        self.predator_actor.SetVisibility(False)

        self.plotter.show(interactive_update=True)

    def update(self):
        positions = np.array([agent.output_last()[:3] for agent in self.sim.agents])
        self.cloud.points = positions

        if self.sim.predator_active():
            x, y, z = self.sim.predator.info()
            self.predator_actor.SetPosition(x, y, z)
            self.predator_actor.SetVisibility(True)
        else:
            self.predator_actor.SetVisibility(False)

        self.plotter.update()


# --- Simulation ---
class Simulation:
    def __init__(
        self,
        N_birds=200,
        nearest_neighbors=7,
        cohesion_scale=1.0,
        alignment_scale=1.0,
        separation_scale=1.5,
        noise_scale=0.3,
        predator_area=50,
        pred_intro=50, 
        predator_enabled = True
    ):
        self.timestep = 0
        self.N_birds = N_birds
        self.nearest_neighbors = nearest_neighbors
        self.cohesion_scale = cohesion_scale
        self.alignment_scale = alignment_scale
        self.separation_scale = separation_scale
        self.noise_scale = noise_scale
        self.predator_area = predator_area
        self.pred_intro = pred_intro
        self.predator_enabled = predator_enabled
        self.predator = None

        # Initialize agents
        self.agents = [Agent(i) for i in range(N_birds)]
        for agent in self.agents:
            pos = np.random.uniform(0, 100, size=3)
            vel = np.random.randn(3)
            vel /= np.linalg.norm(vel)
            agent.setup(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])

    # --- Nearest neighbors with forward-facing vision ---
    def nearest_x_ids(self, positions, agent_ids, num_neighbors, current_index, fov_cos=0.5):
        """
        Find nearest neighbors in front of the bird.
        fov_cos: cosine of the field of view angle (e.g., 0.5 ~ 60° forward cone)
        """
        current_pos = positions[current_index]
        current_vel = self.agents[current_index].output_last()[3:6]
        current_vel_norm = current_vel / np.linalg.norm(current_vel)

        distances = []
        for i, pos in enumerate(positions):
            if i == current_index:
                continue
            vec_to_neighbor = pos - current_pos
            dist = np.linalg.norm(vec_to_neighbor)
            if dist == 0:
                continue
            vec_to_neighbor /= dist  # normalize
            # Only include neighbor if it’s roughly in front
            if np.dot(current_vel_norm, vec_to_neighbor) >= fov_cos:
                distances.append((dist, agent_ids[i]))

        # sort and pick closest num_neighbors
        return [agent_id for _, agent_id in sorted(distances, key=lambda x: x[0])[:num_neighbors]]


    # --- Boids Rules ---
    def cohesion(self, agent_index, neighbor_ids):
        """
        FIX: 
        - before the birds would just go into a straight line, once the forward vision is applied 
        - before did not normalise the vector

        NOW:
        - helps to merge them into a flock
        """
        if not neighbor_ids:
            return np.zeros(3)
        agent_pos = np.array(self.agents[agent_index].output_last()[:3])
        neighbor_positions = np.array([self.agents[nid].output_last()[:3] for nid in neighbor_ids])
        center = neighbor_positions.mean(axis=0)
        dist_to_center = np.linalg.norm(center - agent_pos)
        vec = center - agent_pos
        cohesion_vec = vec / np.linalg.norm(vec)

        return (cohesion_vec)

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
                sep_vec += diff / (dist ** 2)  # stronger at close range
        return sep_vec


    
    # --- Predator Active Toggle --- 
    def predator_active(self):
        return (
            self.predator_enabled
            and self.predator is not None
            and self.timestep >= self.pred_intro
        )

    # --- Predator reaction ---
    def bird_react_to_predator(self, bird_pos, pred_pos, effective_dist):
        vec = bird_pos - pred_pos
        dist = np.linalg.norm(vec)
        if dist == 0:
            return np.zeros(3)
        closeness = 1.0 - (dist / effective_dist)
        beta = 5
        strength = (np.exp(beta * closeness) - 1.0) / (np.exp(beta) - 1.0)
        if dist <= effective_dist:
            return (vec / dist) * strength
        else:
            return np.zeros(3)

    # --- Step ---
    def step(self):
        positions = np.array([a.output_last()[:3] for a in self.agents])
        velocities = np.array([a.output_last()[3:6] for a in self.agents])
        agent_ids = [a.get_id() for a in self.agents]

        predator_active = self.predator_active()
        predator_pos = self.predator.info() if predator_active else None

        for idx, agent in enumerate(self.agents):
            bird_pos = positions[idx]
            neighbors = self.nearest_x_ids(positions, agent_ids, self.nearest_neighbors, idx)

            # --- Boids forces ---
            cohesion_vec = self.cohesion(idx, neighbors) * self.cohesion_scale
            alignment_vec = self.alignment(idx, neighbors) * self.alignment_scale
            separation_vec = self.separation(idx, neighbors) * self.separation_scale


            # --- Noise burst ---
            noise_vec = np.random.randn(3)
            noise_vec /= np.linalg.norm(noise_vec)
            noise_vec *= self.noise_scale

            # --- Wall ---
            wall_vec_3d = wall_vec(bird_pos[0], bird_pos[1], bird_pos[2], effective_distance=10)

            # --- Predator-adaptive behavior ---
            if predator_active:
                bird_to_pred = np.linalg.norm(bird_pos - predator_pos)
                # Reduce cohesion near predator → allows split
                if bird_to_pred < self.predator_area:
                    predator_factor = bird_to_pred / self.predator_area
                    cohesion_vec *= predator_factor  # weaker cohesion when close
                    # Boost separation to escape
                    separation_vec += self.bird_react_to_predator(bird_pos, predator_pos, self.predator_area) * 5.0

            # --- Combine forces with inertia ---
            current_vel = velocities[idx]
            total_vec = cohesion_vec + alignment_vec + separation_vec + noise_vec + wall_vec_3d
            new_vel = current_vel + total_vec
            new_vel /= np.linalg.norm(new_vel)  # normalize
            new_pos = bird_pos + new_vel

            agent.set_current(new_pos[0], new_pos[1], new_pos[2],
                              new_vel[0], new_vel[1], new_vel[2])

        # Commit bird updates
        for agent in self.agents:
            agent.current_to_last()

        # --- Predator logic ---
        if self.predator_enabled and self.timestep == self.pred_intro:
            self.predator = Predator()
            start_pos = np.random.uniform(0, 100, size=3)
            self.predator.update(*start_pos)

        if predator_active:
            px, py, pz = self.predator.info()
            predator_force = np.zeros(3)
            for b in range(len(positions)):
                vec = positions[b] - np.array([px, py, pz])
                dist = np.linalg.norm(vec)
                if 0 < dist < self.predator_area:
                    predator_force += vec / dist
            if np.linalg.norm(predator_force) > 0:
                predator_force /= np.linalg.norm(predator_force)
                predator_speed = 1.5
                self.predator.update(px + predator_force[0] * predator_speed,
                                     py + predator_force[1] * predator_speed,
                                     pz + predator_force[2] * predator_speed)

        self.timestep += 1


# --- Run Simulation ---
sim = Simulation(
    N_birds=200,
    nearest_neighbors=7,
    cohesion_scale=2.0,
    alignment_scale=2.0,
    separation_scale=1.0,
    noise_scale=0.1,
    predator_enabled=True, 
    predator_area=50,
    pred_intro=50
)

viewer = PyVistaViewer(sim)

for _ in range(500):
    sim.step()
    viewer.update()
