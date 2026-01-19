# --- Imports ---
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv 

from agent_class import Agent, Predator 
from wall import wall_vec

# --- Pyvista Visualisation --- 

class PyVistaViewer: 
    def __init__(self, sim):
        self.sim = sim 

        # --- Plotter ---
        self.plotter = pv.Plotter()
        self.plotter.add_axes()
        self.plotter.set_background("black")

        # --- Birds ---
        positions = np.array([agent.output_last()[:3] for agent in sim.agents])
        self.cloud = pv.PolyData(positions)

        self.actor = self.plotter.add_points(
            self.cloud,
            render_points_as_spheres=True,
            point_size=6,
            color="white"
        )

        # --- Predator ---
        self.predator_mesh = pv.Sphere(radius=2.0)
        self.predator_actor = self.plotter.add_mesh(
            self.predator_mesh,
            color="red"
        )
        self.predator_actor.SetVisibility(False)

        self.plotter.show(interactive_update=True)


    def update(self):
        # --- Update birds ---
        new_positions = np.array([agent.output_last()[:3] for agent in self.sim.agents])
        self.cloud.points = new_positions

        # --- Update predator ---
        if self.sim.predator is not None:
            x, y, z = self.sim.predator.info()
            self.predator_actor.SetPosition(x, y, z)
            self.predator_actor.SetVisibility(True)

        self.plotter.update()

# --- Simulation Class ---
class Simulation: 
    def __init__(
        self,
        N_birds,
        nearest_neighbors,
        cohesion_scale,
        alignment_scale,
        separation_scale,
        noise_scale,
        predator_area, 
        pred_intro
    ):
        # --- Simulation parameters ---
        self.timestep = 0
        self.N_birds = N_birds
        self.nearest_neighbors = nearest_neighbors
        
        # --- Boids rule scales ---
        self.cohesion_scale = cohesion_scale
        self.alignment_scale = alignment_scale
        self.separation_scale = separation_scale
        self.noise_scale = noise_scale

        # --- Predator parameters ---
        self.predator_area = predator_area
        self.pred_intro = pred_intro
        self.predator = None

        # --- Initialize agents ---
        self.agents = [Agent(i) for i in range(N_birds)]
        for agent in self.agents:
            pos = np.random.uniform(0, 100, size=3)   # random position
            vel = np.random.randn(3)
            vel /= np.linalg.norm(vel)               # normalize velocity
            agent.setup(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])

    # --- Nearest neighbors ---
    def nearest_x_ids(self, x_positions, y_positions, z_positions, agent_ids, num_neighbors, current_index):
        distances = []
        for i, agent_id in enumerate(agent_ids):
            if i == current_index:
                continue
            dx = x_positions[current_index] - x_positions[i]
            dy = y_positions[current_index] - y_positions[i]
            dz = z_positions[current_index] - z_positions[i]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            distances.append((dist, agent_id))
        # sort and pick closest
        return [agent_id for _, agent_id in sorted(distances, key=lambda x: x[0])[:num_neighbors]]

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
    
    # --- Wall --- 
    def wall_vec(x, y, z, effective_distance):
        def wall_distance(v):
            return v - 100 if v >= 50 else v

        dx = wall_distance(x)
        dy = wall_distance(y)
        dz = wall_distance(z)

        ax = 1 if abs(dx) <= effective_distance else 0
        ay = 1 if abs(dy) <= effective_distance else 0
        az = 1 if abs(dz) <= effective_distance else 0

        return (
            effective_distance * ax / dx,
            effective_distance * ay / dy,
            effective_distance * az / dz
    )

    def bird_react_to_predator(self,bird_loc,pred_loc,effective_dist):

        x,y,z=bird_loc
        i,j,k=pred_loc

        dist=np.sqrt(
            (x - i)**2 +
            (y - j)**2 +
            (z - k)**2
        )

        if dist==0: 
            return np.array([0,0,0]) # can be also changed to bird dying but for now i just ignore
        
        closeness = 1.0 - (dist / effective_dist)
        beta=5 #we can change this on how wild do we want reaction to be 
        strength = (np.exp(beta * closeness) - 1.0) / (np.exp(beta) - 1.0)

        if dist <= effective_dist:
            a=((x-i)/dist)*strength
            b=((y-j)/dist)*strength
            c=((z-k)/dist)*strength
            return np.array([a,b,c])
        else: 
            return np.array([0,0,0])
            


    # --- Step Function ---
    def step(self):
        # --- Cache bird state ---
        positions = np.array([a.output_last()[:3] for a in self.agents])
        velocities = np.array([a.output_last()[3:6] for a in self.agents])
        agent_ids = [a.get_id() for a in self.agents]

        predator_active = (
            self.predator is not None and
            self.timestep >= self.pred_intro
        )

        predator_pos = None
        if predator_active:
            predator_pos = self.predator.info()

        # =========================
        #        UPDATE BIRDS
        # =========================
        for idx, agent in enumerate(self.agents):
            bird_pos = positions[idx]

            # --- Neighbors ---
            neighbors = self.nearest_x_ids(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                agent_ids,
                self.nearest_neighbors,
                idx
            )

            # --- Boids forces ---
            cohesion_vec = self.cohesion(idx, neighbors) * self.cohesion_scale
            alignment_vec = self.alignment(idx, neighbors) * self.alignment_scale
            separation_vec = self.separation(idx, neighbors) * self.separation_scale

            # --- Noise ---
            noise_vec = np.random.randn(3)
            noise_vec /= np.linalg.norm(noise_vec)
            noise_vec *= self.noise_scale

            # --- Wall avoidance ---
            wall_vec_3d = np.array(
                wall_vec(bird_pos[0], bird_pos[1], bird_pos[2], effective_distance=10)
            )

            # --- Predator avoidance (THIS WAS MISSING) ---
            predator_vec = np.zeros(3)
            if predator_active:
                predator_vec = self.bird_react_to_predator(
                    bird_loc=bird_pos,
                    pred_loc=predator_pos,
                    effective_dist=self.predator_area
                )

            # --- Combine all forces ---
            total_vec = (
                cohesion_vec +
                alignment_vec +
                separation_vec +
                noise_vec +
                wall_vec_3d +
                predator_vec * 8.0   # 🔥 critical scaling
            )

            # --- Update bird ---
            agent.set_current(
                bird_pos[0] + total_vec[0],
                bird_pos[1] + total_vec[1],
                bird_pos[2] + total_vec[2],
                total_vec[0],
                total_vec[1],
                total_vec[2]
            )

        # --- Commit bird updates ---
        for agent in self.agents:
            agent.current_to_last()

        # =========================
        #      PREDATOR LOGIC
        # =========================
        if self.timestep == self.pred_intro:
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
                self.predator.update(
                    px + predator_force[0] * predator_speed,
                    py + predator_force[1] * predator_speed,
                    pz + predator_force[2] * predator_speed
                )

        # --- Advance time ---
        self.timestep += 1

    # --- Visualization ---
    def show(self, scatter_plot, fig):
        x, y, z = zip(*[agent.output_last()[:3] for agent in self.agents])
        scatter_plot._offsets3d = (x, y, z)
        fig.canvas.draw_idle()
        plt.pause(0.05)

# Initialize simulation
sim = Simulation(
    N_birds=200,
    nearest_neighbors=7,
    cohesion_scale=1,
    alignment_scale=1,
    separation_scale=1,
    noise_scale=0.3,
    predator_area=25, 
    pred_intro=50
)

viewer = PyVistaViewer(sim)

for _ in range(500):
    sim.step()
    viewer.update()
