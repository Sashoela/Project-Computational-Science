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
        predator_enabled=True,
        perception_radius=100
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
        self.perception_radius = perception_radius

        # Initialize agents
        self.agents = [Agent(i) for i in range(N_birds)]
        for agent in self.agents:
            pos = np.random.uniform(0, 100, size=3)
            vel = np.random.randn(3)
            vel /= np.linalg.norm(vel)
            agent.setup(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])

    # --- Nearest neighbors with forward-facing vision ---
    def nearest_x_ids(self, positions, agent_ids, num_neighbors, current_index, fov_cos=0.5):
        current_pos = positions[current_index]
        current_vel = self.agents[current_index].output_last()[3:6]
        current_vel_norm = current_vel / np.linalg.norm(current_vel)

        distances = []
        for i, pos in enumerate(positions):
            if i == current_index:
                continue
            vec_to_neighbor = pos - current_pos
            dist = np.linalg.norm(vec_to_neighbor)
            if dist == 0 or dist > self.perception_radius:
                continue
            vec_to_neighbor /= dist
            if np.dot(current_vel_norm, vec_to_neighbor) >= fov_cos:
                distances.append((dist, agent_ids[i]))

        neighbors = [agent_id for _, agent_id in sorted(distances, key=lambda x: x[0])[:num_neighbors]]

        if not neighbors:
            fallback_ids = [i for i, pos in enumerate(positions)
                            if i != current_index and np.linalg.norm(pos - current_pos) <= self.perception_radius]
            if fallback_ids:
                neighbors = random.sample(fallback_ids, min(num_neighbors, len(fallback_ids)))

        return neighbors

    # --- Boids Rules ---
    def cohesion(self, agent_index, neighbor_ids):
        if not neighbor_ids:
            return np.zeros(3)
        agent_pos = np.array(self.agents[agent_index].output_last()[:3])
        neighbor_positions = np.array([self.agents[nid].output_last()[:3] for nid in neighbor_ids])
        vec = neighbor_positions.mean(axis=0) - agent_pos
        if np.linalg.norm(vec) == 0:
            return np.zeros(3)
        return vec / np.linalg.norm(vec)

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
                sep_vec += diff / (dist ** 2)
        return sep_vec

    # --- Predator Toggle ---
    def predator_active(self):
        return self.predator_enabled and self.predator is not None and self.timestep >= self.pred_intro

    # --- Bird Reaction to Predator ---
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
        return np.zeros(3)

    # --- Step ---
    def step(self):
        positions = np.array([a.output_last()[:3] for a in self.agents])
        velocities = np.array([a.output_last()[3:6] for a in self.agents])
        agent_ids = [a.get_id() for a in self.agents]

        predator_active = self.predator_active()

        # --- Update birds ---
        for idx, agent in enumerate(self.agents):
            bird_pos = positions[idx]
            neighbors = self.nearest_x_ids(positions, agent_ids, self.nearest_neighbors, idx)

            cohesion_vec = self.cohesion(idx, neighbors) * self.cohesion_scale
            alignment_vec = self.alignment(idx, neighbors) * self.alignment_scale
            separation_vec = self.separation(idx, neighbors) * self.separation_scale
            noise_vec = np.random.randn(3)
            noise_vec /= np.linalg.norm(noise_vec)
            noise_vec *= self.noise_scale
            wall_vec_3d = wall_vec(bird_pos[0], bird_pos[1], bird_pos[2], effective_distance=10)

            if predator_active:
                bird_to_pred = np.linalg.norm(bird_pos - np.array(self.predator.info()))
                if bird_to_pred < self.predator_area:
                    predator_factor = bird_to_pred / self.predator_area
                    cohesion_vec *= predator_factor
                    separation_vec += self.bird_react_to_predator(bird_pos, np.array(self.predator.info()), self.predator_area) * 5.0

            total_vec = cohesion_vec + alignment_vec + separation_vec + noise_vec + wall_vec_3d
            new_vel = velocities[idx] + total_vec
            new_vel /= np.linalg.norm(new_vel)
            new_pos = bird_pos + new_vel
            agent.set_current(new_pos[0], new_pos[1], new_pos[2], new_vel[0], new_vel[1], new_vel[2])

        for agent in self.agents:
            agent.current_to_last()

        # --- Predator logic ---
        if self.predator_enabled and self.timestep == self.pred_intro:
            self.predator = Predator()
            start_pos = np.random.uniform(0, 100, size=3)
            self.predator.update(*start_pos)

        if predator_active:
            predator_pos = np.array(self.predator.info())
            flock_threshold = 20
            unassigned = set(range(len(positions)))
            flocks = []

            while unassigned:
                idx = unassigned.pop()
                flock = [idx]
                for j in list(unassigned):
                    if np.linalg.norm(positions[j] - positions[idx]) <= flock_threshold:
                        flock.append(j)
                        unassigned.remove(j)
                flocks.append(flock)

            max_size = max(len(f) for f in flocks)
            candidate_flocks = [f for f in flocks if len(f) == max_size]

            flock_centers = [positions[f].mean(axis=0) for f in candidate_flocks]
            distances = [np.linalg.norm(center - predator_pos) for center in flock_centers]
            target_center = flock_centers[np.argmin(distances)]

            vec_to_flock = target_center - predator_pos
            if np.linalg.norm(vec_to_flock) > 0:
                vec_to_flock /= np.linalg.norm(vec_to_flock)
                predator_speed = 1.5
                new_pred_pos = predator_pos + vec_to_flock * predator_speed
                self.predator.update(*new_pred_pos)

        self.timestep += 1




# --- Run Simulation ---
sim = Simulation(
    N_birds=200,
    nearest_neighbors=7,
    cohesion_scale=2.0,
    alignment_scale=2.0,
    separation_scale=1,
    noise_scale=0.1,
    predator_enabled=True, 
    predator_area=50,
    pred_intro=50,
    perception_radius=100
)

viewer = PyVistaViewer(sim)

for _ in range(1000):
    sim.step()
    viewer.update()
