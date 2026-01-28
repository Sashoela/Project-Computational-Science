# import libraries
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# import our own functions
from agent_class import Agent, Predator
from wall import wall_vec


class Simulation():
    def __init__(self, N_birds, nearest_x, coh_vector_scale, ali_vector_scale, sep_vector_scale, noise_vector_scale, pred_intro_time, pred_exit_time):
        # needed variables
        self.timestep = 0
        self.N_birds = N_birds
        self.nearest_x = nearest_x
        self.coh_vector_scale = coh_vector_scale
        self.ali_vector_scale = ali_vector_scale
        self.sep_vector_scale = sep_vector_scale
        self.noise_vector_scale = noise_vector_scale
        self.predator_area = 50
        self.pred_intro = pred_intro_time
        self.pred_exit_time = pred_exit_time

        # initialize birds in a list
        self.agents = [Agent(i) for i in range(N_birds)]

        # give random location and speed
        for agent in self.agents:
            v = np.random.normal(size=3)
            unit_v = v / np.linalg.norm(v)
            agent.setup(
                random.uniform(40, 60),
                random.uniform(40, 60),
                random.uniform(40, 60),
                unit_v[0], unit_v[1], unit_v[2]
            )

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

    # ---- validation helpers (NEW) ----
    @staticmethod
    def flock_dimensions_I_from_positions(P):
        """
        P: Nx3 positions array.
        Returns I1<I2<I3 as PCA-axis extents (max-min).
        """
        P = np.asarray(P, dtype=float)
        P = P - P.mean(axis=0)

        C = np.cov(P.T, bias=True)
        w, V = np.linalg.eigh(C)
        V = V[:, np.argsort(w)]  # thin -> long axes

        proj = P @ V
        extents = [proj[:, a].max() - proj[:, a].min() for a in range(3)]
        I1, I2, I3 = sorted(extents)
        return I1, I2, I3

    def validate_ratios(self, start_step=20, stride=5):
        """
        Uses ALL birds (single flock assumption).
        Samples timesteps t >= start_step and every `stride` steps.
        Stores ratios in self.validation_samples and returns summary dict.
        """
        xs, ys, zs = [], [], []
        for agent in self.agents:
            x, y, z, vx, vy, vz, _id = agent.output_last()
            xs.append(x); ys.append(y); zs.append(z)

        P = np.column_stack([xs, ys, zs])
        I1, I2, I3 = self.flock_dimensions_I_from_positions(P)

        r21 = (I2 / I1) if I1 > 1e-12 else np.nan
        r31 = (I3 / I1) if I1 > 1e-12 else np.nan

        if not hasattr(self, "validation_samples"):
            self.validation_samples = []  # list of dicts

        self.validation_samples.append({
            "t": self.timestep,
            "I1": I1, "I2": I2, "I3": I3,
            "r21": r21, "r31": r31
        })

        return {"t": self.timestep, "r21": r21, "r31": r31}

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
            vector = np.array([0.0, 0.0, 0.0], dtype=np.float64)

            listed = []
            bin = 10
            for a in range(0, 9):
                for b in range(0, 9):
                    for c in range(0, 9):
                        amount = 0
                        for d in range(0, len(i)):
                            if (bin * a <= i[d] <= bin * (a + 1) and
                                bin * b <= j[d] <= bin * (b + 1) and
                                bin * c <= k[d] <= bin * (c + 1)):
                                amount += 1
                        listed.append((amount, bin * a + bin / 2, bin * b + bin / 2, bin * c + bin / 2))

            listed_sorted = sorted(listed, key=lambda x: x[0], reverse=True)
            top3 = listed_sorted[:3]
            distances = []
            for m in range(0, 3):
                g, a, b, c = top3[m]
                d = np.sqrt((predx - a)**2 + (predy - b)**2 + (predz - c)**2)
                distances.append((d, a, b, c))
            best_dist, best_a, best_b, best_c = min(distances, key=lambda x: x[0])
            coordinates = (best_a, best_b, best_c)

            dist = np.linalg.norm(np.array([predx, predy, predz]) - np.array(coordinates))
            if dist < self.predator_area:
                vec = np.array([coordinates[0] - predx, coordinates[1] - predy, coordinates[2] - predz], dtype=np.float64)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vector = vec / norm
            else:
                vec = np.array([i[0] - predx, j[0] - predy, k[0] - predz], dtype=np.float64)
                vector = vec / np.linalg.norm(vec)

            if any(vector):
                movement = vector / np.linalg.norm(vector) * np.sqrt(2)
                self.predator.update(predx + movement[0], predy + movement[1], predz + movement[2])

        self.timestep += 1

        # make a loop to update all agents (loop over all birds)
        for agent in self.agents:
            x, y, z, vx, vy, vz, id = agent.output_last()

            nearest_ids = self.nearest_x_ids(i, j, k, ids, self.nearest_x, id)

            bird_loc = (x, y, z)
            effective_dist = 20
            if self.timestep > self.pred_intro and self.timestep <= self.pred_exit_time:
                pred_loc = self.predator.info()
                react_pred_vec = self.bird_react_to_predator(bird_loc, pred_loc, effective_dist)
            else:
                react_pred_vec = np.array([0.0, 0.0, 0.0])

            cohesion_vec = self.cohesion(id, nearest_ids)
            if any(cohesion_vec):
                cohesion_vec = (cohesion_vec / np.linalg.norm(cohesion_vec)) * self.coh_vector_scale

            alignment_vec = self.alignment(id, nearest_ids)
            if any(alignment_vec):
                alignment_vec = (alignment_vec / np.linalg.norm(alignment_vec)) * self.ali_vector_scale

            seperation_vec = self.separation(id, nearest_ids)
            if any(seperation_vec):
                seperation_vec = (seperation_vec / np.linalg.norm(seperation_vec)) * self.sep_vector_scale

            noise = np.random.normal(size=3)
            scaled_noise = noise / np.linalg.norm(noise) * self.noise_vector_scale

            # total movement :
            if any(react_pred_vec):
                total_vec = react_pred_vec + scaled_noise
            else:
                total_vec = cohesion_vec + alignment_vec + seperation_vec + scaled_noise
                total_vec = total_vec / np.linalg.norm(total_vec)

            wall = np.array(wall_vec(x, y, z, 5), dtype=np.float64)

            total_vec += wall
            agent.set_current(
                x + total_vec[0], y + total_vec[1], z + total_vec[2],
                total_vec[0], total_vec[1], total_vec[2]
            )

        for agent in self.agents:
            agent.current_to_last()

    def show(self):
        i, j, k = [], [], []
        for agent in self.agents:
            x, y, z, vx, vy, vz, id = agent.output_last()
            i.append(x); j.append(y); k.append(z)

        if self.timestep > self.pred_intro:
            x, y, z = self.predator.info()
            i.append(x); j.append(y); k.append(z)
            print(x, y, z)

        scat._offsets3d = (i, j, k)
        fig.canvas.draw_idle()
        plt.pause(0.05)

    def dump(self):
        i, j, k = [], [], []
        for agent in self.agents:
            x, y, z, vx, vy, vz, id = agent.output_last()
            i.append(x); j.append(y); k.append(z)
        return i, j, k

    def nearest_x_ids(self, i, j, k, ids, near_x, initial_bird, fov_cos=0.5):
        n = initial_bird
        nr = len(i)
        items = []  # (distance, id)
        self_vel = np.array(self.agents[n].output_last()[3:6])
        self_vel = self_vel / np.linalg.norm(self_vel)

        for m in range(nr):
            if m == n:
                continue
            d = np.sqrt((i[n] - i[m])**2 + (j[n] - j[m])**2 + (k[n] - k[m])**2)
            vec_to_neighbour = np.array([i[m] - i[n], j[m] - j[n], k[m] - k[n]])
            vec_to_neighbour /= np.linalg.norm(vec_to_neighbour)
            if np.dot(self_vel, vec_to_neighbour) >= fov_cos:
                items.append((d, ids[m]))

        smallest_near_x = sorted(items, key=lambda x: x[0])[:near_x]
        return [id_ for _, id_ in smallest_near_x]

    def bird_react_to_predator(self, bird_loc, pred_loc, effective_dist):
        x, y, z = bird_loc
        i, j, k = pred_loc
        dist = np.sqrt((x - i)**2 + (y - j)**2 + (z - k)**2)
        if dist == 0:
            return np.array([0.0, 0.0, 0.0])
        dx, dy, dz = x - i, y - j, z - k
        direction = np.array([dx, dy, dz]) / dist
        closeness = 1.0 - (dist / effective_dist)
        beta = 5
        strength = (np.exp(beta * closeness) - 1.0) / (np.exp(beta) - 1.0)
        if dist <= effective_dist:
            min_speed, max_speed = 1.0, 2.0
            speed = min_speed + (max_speed - min_speed) * strength
            return speed * direction
        else:
            return np.array([0.0, 0.0, 0.0])


# ----- test + validation sampling -----
sim = Simulation(400, 7, 0.3, 0.3, 0.3, 0.1, 600, 600)  # predator never really active

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
scat = ax.scatter([], [], [], s=5)

start_step = 20
stride = 5
steps = 400

for _ in range(steps):
    sim.step()
    sim.show()

    # sample ratios starting from step 20, every 5 steps
    if sim.timestep >= start_step and (sim.timestep - start_step) % stride == 0:
        sim.validate_ratios(start_step=start_step, stride=stride)

plt.ioff()
plt.show()

# ----- summary -----
r21 = [d["r21"] for d in getattr(sim, "validation_samples", []) if np.isfinite(d["r21"])]
r31 = [d["r31"] for d in getattr(sim, "validation_samples", []) if np.isfinite(d["r31"])]

def ci95(x):
    if len(x) < 2:
        return np.nan
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))

print("\n=== VALIDATION (single flock, samples from t>=20 every 5 steps) ===")
print("samples:", len(r21))
if len(r21) >= 2:
    print("I2/I1 = %.3f ± %.3f (95%% CI)" % (np.mean(r21), ci95(r21)))
else:
    print("I2/I1: not enough samples")

if len(r31) >= 2:
    print("I3/I1 = %.3f ± %.3f (95%% CI)" % (np.mean(r31), ci95(r31)))
else:
    print("I3/I1: not enough samples")

# optional: print the first few samples
for d in getattr(sim, "validation_samples", [])[:5]:
    print(f"t={d['t']} r21={d['r21']:.3f} r31={d['r31']:.3f}")
