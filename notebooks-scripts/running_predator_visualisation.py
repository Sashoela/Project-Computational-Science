# --- Imports ---
import math
import random
import numpy as np
import pyvista as pv

from agent_class import Agent  # Predator will be separate
from wall import wall_vec


# --- Predator Class ---
class Predator():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0

    def info(self):
        return np.array([self.x, self.y, self.z])

    def velocity(self):
        return np.array([self.vx, self.vy, self.vz])

    def update(self, new_x, new_y, new_z):
        self.vx = new_x - self.x
        self.vy = new_y - self.y
        self.vz = new_z - self.z
        self.x = new_x
        self.y = new_y
        self.z = new_z


# --- Simulation Class ---
class Simulation:
    def __init__(self, N_birds, nearest_neighbors, cohesion_scale, alignment_scale, separation_scale, noise_scale):
        self.timestep = 0
        self.N_birds = N_birds
        self.nearest_neighbors = nearest_neighbors
        self.cohesion_scale = cohesion_scale
        self.alignment_scale = alignment_scale
        self.separation_scale = separation_scale
        self.noise_scale = noise_scale

        # Initialize birds
        self.agents = [Agent(i) for i in range(N_birds)]
        for agent in self.agents:
            pos = np.random.uniform(0, 100, size=3)
            vel = np.random.randn(3)
            vel /= np.linalg.norm(vel)
            agent.setup(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])

        # Predator
        self.predator = None
        self.pred_intro = 50
        self.predator_area = 30

    # --- Boids rules ---
    def nearest_x_ids(self, x, y, z, agent_ids, num_neighbors, idx):
        distances = []
        for i, aid in enumerate(agent_ids):
            if i == idx:
                continue
            dx, dy, dz = x[idx] - x[i], y[idx] - y[i], z[idx] - z[i]
            distances.append((np.linalg.norm([dx, dy, dz]), aid))
        return [aid for _, aid in sorted(distances)[:num_neighbors]]

    def cohesion(self, idx, neighbors):
        if not neighbors:
            return np.zeros(3)
        agent_pos = np.array(self.agents[idx].output_last()[:3])
        neighbor_pos = np.array([self.agents[nid].output_last()[:3] for nid in neighbors])
        return neighbor_pos.mean(axis=0) - agent_pos

    def alignment(self, idx, neighbors):
        if not neighbors:
            return np.zeros(3)
        neighbor_vel = np.array([self.agents[nid].output_last()[3:6] for nid in neighbors])
        return neighbor_vel.mean(axis=0)

    def separation(self, idx, neighbors, separation_distance=5.0):
        if not neighbors:
            return np.zeros(3)
        agent_pos = np.array(self.agents[idx].output_last()[:3])
        sep_vec = np.zeros(3)
        for nid in neighbors:
            neighbor_pos = np.array(self.agents[nid].output_last()[:3])
            diff = agent_pos - neighbor_pos
            dist = np.linalg.norm(diff)
            if 0 < dist < separation_distance:
                sep_vec += diff / dist
        return sep_vec

    # --- Step Function ---
    def step(self):
        positions = np.array([a.output_last()[:3] for a in self.agents])
        velocities = np.array([a.output_last()[3:6] for a in self.agents])
        agent_ids = [a.get_id() for a in self.agents]

        # --- Update birds ---
        for idx, agent in enumerate(self.agents):
            agent_pos = positions[idx]

            neighbors = self.nearest_x_ids(positions[:,0], positions[:,1], positions[:,2], agent_ids, self.nearest_neighbors, idx)
            cohesion_vec = self.cohesion(idx, neighbors) * self.cohesion_scale
            alignment_vec = self.alignment(idx, neighbors) * self.alignment_scale
            separation_vec = self.separation(idx, neighbors) * self.separation_scale

            # Predator avoidance
            if self.predator is not None:
                pred_pos = self.predator.info()
                diff = agent_pos - pred_pos
                dist = np.linalg.norm(diff)
                if dist < self.predator_area:
                    separation_vec += (diff / dist) * 2.0  # flee weight

            # Noise
            noise_vec = np.random.randn(3)
            noise_vec /= np.linalg.norm(noise_vec)
            noise_vec *= self.noise_scale

            # Wall
            wall_vec_3d = np.array(wall_vec(agent_pos[0], agent_pos[1], agent_pos[2], 10))

            total_vec = cohesion_vec + alignment_vec + separation_vec + noise_vec + wall_vec_3d

            agent.set_current(agent_pos[0]+total_vec[0], agent_pos[1]+total_vec[1], agent_pos[2]+total_vec[2],
                              total_vec[0], total_vec[1], total_vec[2])

        # --- Predator introduction ---
        if self.timestep == self.pred_intro:
            self.predator = Predator()
            pos = np.random.uniform(0, 100, size=3)
            self.predator.update(pos[0], pos[1], pos[2])

        # --- Predator movement ---
        if self.predator is not None:
            pred_pos = self.predator.info()
            vector = np.zeros(3)
            for a in range(len(positions)):
                bird_pos = positions[a]
                dist = np.linalg.norm(pred_pos - bird_pos)
                if dist < self.predator_area:
                    vec = bird_pos - pred_pos
                    vec /= np.linalg.norm(vec)
                    vector += vec
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector) * np.sqrt(2)
                new_pos = pred_pos + vector
                self.predator.update(*new_pos)

        # Save bird positions
        for agent in self.agents:
            agent.current_to_last()

        self.timestep += 1


# --- PyVista Viewer ---
class PyVistaViewer:
    def __init__(self, sim, arrow_scale=2.0):
        self.sim = sim
        self.arrow_scale = arrow_scale

        # Bird positions & velocities
        positions = np.array([a.output_last()[:3] for a in sim.agents])
        velocities = np.array([a.output_last()[3:6] for a in sim.agents])
        self.cloud = pv.PolyData(positions)
        self.cloud["velocity"] = velocities

        # Predator
        if sim.predator is not None:
            pred_pos = np.array([sim.predator.info()])
            pred_vel = np.array([sim.predator.velocity()])
        else:
            pred_pos = np.empty((0,3))
            pred_vel = np.empty((0,3))
        self.predator_cloud = pv.PolyData(pred_pos)
        self.predator_cloud["velocity"] = pred_vel

        # Plotter
        self.plotter = pv.Plotter()
        self.plotter.add_axes()
        self.plotter.set_background("black")

        # Bird arrows
        arrows = self.cloud.glyph(orient="velocity", scale="velocity", factor=self.arrow_scale)
        self.actor = self.plotter.add_mesh(arrows, color="white")

        # Predator arrow
        pred_arrows = self.predator_cloud.glyph(orient="velocity", scale="velocity", factor=self.arrow_scale)
        self.predator_actor = self.plotter.add_mesh(pred_arrows, color="red")

        self.plotter.show(interactive_update=True)

    def update(self):
        # Birds
        new_positions = np.array([a.output_last()[:3] for a in self.sim.agents])
        new_velocities = np.array([a.output_last()[3:6] for a in self.sim.agents])
        self.cloud.points = new_positions
        self.cloud["velocity"] = new_velocities
        arrows = self.cloud.glyph(orient="velocity", scale="velocity", factor=self.arrow_scale)
        self.actor.mapper.SetInputData(arrows)

        # Predator
        if self.sim.predator is not None:
            pred_pos = np.array([self.sim.predator.info()])
            pred_vel = np.array([self.sim.predator.velocity()])
            self.predator_cloud.points = pred_pos
            self.predator_cloud["velocity"] = pred_vel
            pred_arrows = self.predator_cloud.glyph(orient="velocity", scale="velocity", factor=self.arrow_scale)
            self.predator_actor.mapper.SetInputData(pred_arrows)

        self.plotter.update()


# --- Initialize and run ---
sim = Simulation(
    N_birds=200,
    nearest_neighbors=7,
    cohesion_scale=1,
    alignment_scale=1,
    separation_scale=1,
    noise_scale=0.3
)

viewer = PyVistaViewer(sim, arrow_scale=2.0)

for _ in range(500):
    sim.step()
    viewer.update()
