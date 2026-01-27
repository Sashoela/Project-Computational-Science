# --- Imports ---
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv 

from agent_class import Agent, Predator 
from wall import wall_vec
from pyvista_class import *


class Simulation:
    """
    Boids-style flocking simulation with optional predator.
    Responsible ONLY for simulation state and time evolution.
    """

    # ============================================================
    # INITIALISATION
    # ============================================================
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
        predator_enabled=True
    ):
        # --- Time ---
        self.timestep = 0

        # --- Flock parameters ---
        self.N_birds = N_birds
        self.nearest_neighbors = nearest_neighbors
        self.cohesion_scale = cohesion_scale
        self.alignment_scale = alignment_scale
        self.separation_scale = separation_scale
        self.noise_scale = noise_scale

        # --- Predator parameters ---
        self.predator_area = predator_area
        self.pred_intro = pred_intro
        self.predator_enabled = predator_enabled
        self.predator = None

        # --- Initialise birds ---
        self.agents = [Agent(i) for i in range(N_birds)]

        for agent in self.agents:
            pos = np.random.uniform(0, 100, size=3)
            vel = np.random.randn(3)
            vel /= np.linalg.norm(vel)

            agent.setup(
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2]
            )

    # ============================================================
    # NEIGHBOUR FINDING
    # ============================================================
    def nearest_x_ids(
        self,
        positions,
        agent_ids,
        num_neighbors,
        current_index,
        fov_cos=0.5
    ):
        """
        Return IDs of nearest neighbours within forward-facing cone.
        """
        current_pos = positions[current_index]

        current_vel = self.agents[current_index].output_last()[3:6]
        current_vel /= np.linalg.norm(current_vel)

        distances = []

        for i, pos in enumerate(positions):
            if i == current_index:
                continue

            vec = pos - current_pos
            dist = np.linalg.norm(vec)

            if dist == 0:
                continue

            vec /= dist

            # Forward-facing vision constraint
            if np.dot(current_vel, vec) >= fov_cos:
                distances.append((dist, agent_ids[i]))

        distances.sort(key=lambda x: x[0])
        return [agent_id for _, agent_id in distances[:num_neighbors]]

    # ============================================================
    # BOIDS FORCES
    # ============================================================
    def cohesion(self, agent_index, neighbor_ids):
        """Steer toward neighbour centre of mass."""
        if not neighbor_ids:
            return np.zeros(3)

        agent_pos = self.agents[agent_index].output_last()[:3]
        neighbor_positions = np.array(
            [self.agents[nid].output_last()[:3] for nid in neighbor_ids]
        )

        centre = neighbor_positions.mean(axis=0)
        vec = centre - agent_pos

        return vec / np.linalg.norm(vec)

    def alignment(self, agent_index, neighbor_ids):
        """Align velocity with neighbours."""
        if not neighbor_ids:
            return np.zeros(3)

        neighbor_vels = np.array(
            [self.agents[nid].output_last()[3:6] for nid in neighbor_ids]
        )

        return neighbor_vels.mean(axis=0)

    def separation(self, agent_index, neighbor_ids, separation_distance=5.0):
        """Repel close neighbours."""
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

    # ============================================================
    # PREDATOR HELPERS
    # ============================================================
    def predator_active(self):
        return (
            self.predator_enabled
            and self.predator is not None
            and self.timestep >= self.pred_intro
        )

    def bird_react_to_predator(self, bird_pos, pred_pos, effective_dist):
        """Escape force away from predator."""
        vec = bird_pos - pred_pos
        dist = np.linalg.norm(vec)

        if dist == 0:
            return np.zeros(3)

        closeness = 1.0 - dist / effective_dist
        beta = 5

        strength = (np.exp(beta * closeness) - 1) / (np.exp(beta) - 1)

        if dist <= effective_dist:
            return (vec / dist) * strength

        return np.zeros(3)

    # ============================================================
    # SINGLE TIMESTEP UPDATE
    # ============================================================
    def step(self):
        """
        Advance simulation by one timestep.
        """
        positions = np.array([a.output_last()[:3] for a in self.agents])
        velocities = np.array([a.output_last()[3:6] for a in self.agents])
        agent_ids = [a.get_id() for a in self.agents]

        predator_active = self.predator_active()
        predator_pos = self.predator.info() if predator_active else None

        # --- Update birds ---
        for idx, agent in enumerate(self.agents):
            bird_pos = positions[idx]

            neighbors = self.nearest_x_ids(
                positions,
                agent_ids,
                self.nearest_neighbors,
                idx
            )

            # Boids forces
            cohesion_vec = self.cohesion(idx, neighbors) * self.cohesion_scale
            alignment_vec = self.alignment(idx, neighbors) * self.alignment_scale
            separation_vec = self.separation(idx, neighbors) * self.separation_scale

            # Noise
            noise_vec = np.random.randn(3)
            noise_vec /= np.linalg.norm(noise_vec)
            noise_vec *= self.noise_scale

            # Walls
            wall_vec_3d = wall_vec(*bird_pos, effective_distance=10)

            # Predator influence
            if predator_active:
                bird_to_pred = np.linalg.norm(bird_pos - predator_pos)
                if bird_to_pred < self.predator_area:
                    factor = bird_to_pred / self.predator_area
                    cohesion_vec *= factor
                    separation_vec += (
                        self.bird_react_to_predator(
                            bird_pos,
                            predator_pos,
                            self.predator_area
                        ) * 5.0
                    )

            # Velocity + position update
            new_vel = velocities[idx] + (
                cohesion_vec
                + alignment_vec
                + separation_vec
                + noise_vec
                + wall_vec_3d
            )

            new_vel /= np.linalg.norm(new_vel)
            new_pos = bird_pos + new_vel

            agent.set_current(
                new_pos[0], new_pos[1], new_pos[2],
                new_vel[0], new_vel[1], new_vel[2]
            )

        # Commit updates
        for agent in self.agents:
            agent.current_to_last()

        # --- Predator logic ---
        if self.predator_enabled and self.timestep == self.pred_intro:
            self.predator = Predator()
            self.predator.update(*np.random.uniform(0, 100, size=3))

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
                self.predator.update(
                    px + predator_force[0] * predator_speed,
                    py + predator_force[1] * predator_speed,
                    pz + predator_force[2] * predator_speed
                )

        self.timestep += 1


# --- Run Simulation ---
sim = Simulation(
    N_birds=200,
    nearest_neighbors=7,
    cohesion_scale=2.0,
    alignment_scale=2.0,
    separation_scale=0.75,
    noise_scale=0.1,
    predator_enabled=False, 
    predator_area=50,
    pred_intro=50
)

viewer = PyVistaViewer(sim)

for _ in range(1000):
    sim.step()
    viewer.update()
