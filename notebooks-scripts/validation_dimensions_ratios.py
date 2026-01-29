#validation on dimensions ratios I1/I3 and I2/I3 95% CI intervals
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque, defaultdict

from agent_class import Agent, Predator
from wall import wall_vec


class Simulation():
    def __init__(
        self,
        N_birds, nearest_x,
        coh_vector_scale, ali_vector_scale, sep_vector_scale, noise_vector_scale,
        pred_intro_time, pred_exit_time,
        *,
        # flocks + flattening params
        cluster_R=6.0,            # distance threshold to connect birds in same flock
        min_flock_size=25,        # ignore small components for flattening/validation
        flatten_strength=0.03,    # strength of pull-to-plane

        # validation gating (turning detector)
        validate_start_step=20,   # start measuring after transient
        turn_threshold_deg=6.0,   # max heading change allowed to count as not turning
        turn_smooth_window=4,     # heading history window per flock
        compute_I_every=1,        # compute ratios every N steps when not turning

        dump_not_turning=False,   # optional debug printing
        dump_every=50,            # debug print frequency
    ):
        # core state
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

        # flock/flatten config
        self.cluster_R = float(cluster_R)
        self.min_flock_size = int(min_flock_size)
        self.flatten_strength = float(flatten_strength)

        # validation config
        self.validate_start_step = int(validate_start_step)
        self.turn_threshold_rad = np.deg2rad(turn_threshold_deg)
        self.turn_smooth_window = int(turn_smooth_window)
        self.compute_I_every = int(compute_I_every)
        self.dump_not_turning = bool(dump_not_turning)
        self.dump_every = int(dump_every)

        # per-flock heading history for turning detection
        self.flock_heading_hist = {}

        # measurement store for later summary
        self.I_history = []  # entries hold I1,I2,I3 and ratios for not-turning flocks

        # create birds
        self.agents = [Agent(i) for i in range(N_birds)]

        # initialize positions and unit velocities
        for agent in self.agents:
            v = np.random.normal(size=3)
            unit_v = v / np.linalg.norm(v)
            agent.setup(
                random.uniform(40, 60),
                random.uniform(40, 60),
                random.uniform(40, 60),
                unit_v[0], unit_v[1], unit_v[2]
            )

    # utilities
    @staticmethod
    def safe_unit(v, eps=1e-12):
        # normalize vector safely (avoid divide by zero)
        n = np.linalg.norm(v)
        return v / n if n > eps else np.zeros_like(v)

    @staticmethod
    def angle_between(u, v, eps=1e-12):
        # compute angle between u and v in radians
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu < eps or nv < eps:
            return 0.0
        c = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
        return float(np.arccos(c))

    # boids rules
    def cohesion(self, agent_index, neighbor_ids):
        # steer toward neighbors' mean position
        if not neighbor_ids:
            return np.zeros(3)
        agent_pos = np.array(self.agents[agent_index].output_last()[:3], dtype=float)
        neighbor_positions = np.array([self.agents[nid].output_last()[:3] for nid in neighbor_ids], dtype=float)
        return neighbor_positions.mean(axis=0) - agent_pos

    def alignment(self, agent_index, neighbor_ids):
        # steer toward neighbors' mean velocity
        if not neighbor_ids:
            return np.zeros(3)
        neighbor_vels = np.array([self.agents[nid].output_last()[3:6] for nid in neighbor_ids], dtype=float)
        return neighbor_vels.mean(axis=0)

    def separation(self, agent_index, neighbor_ids, separation_distance=5.0):
        # steer away from neighbors closer than separation_distance
        if not neighbor_ids:
            return np.zeros(3)
        agent_pos = np.array(self.agents[agent_index].output_last()[:3], dtype=float)
        sep_vec = np.zeros(3, dtype=float)
        for nid in neighbor_ids:
            neighbor_pos = np.array(self.agents[nid].output_last()[:3], dtype=float)
            diff = agent_pos - neighbor_pos
            dist = np.linalg.norm(diff)
            if 0 < dist < separation_distance:
                sep_vec += diff / dist
        return sep_vec

    # neighbor selection with FOV
    def nearest_x_ids(self, i, j, k, ids, near_x, initial_bird, fov_cos=0.5):
        # pick nearest_x neighbors that also lie within the bird's field-of-view cone
        n = initial_bird
        nr = len(i)
        items = []

        # normalize self velocity for FOV check
        self_vel = np.array(self.agents[n].output_last()[3:6], dtype=float)
        nv = np.linalg.norm(self_vel)
        if nv < 1e-12:
            self_vel = np.array([1.0, 0.0, 0.0])
        else:
            self_vel /= nv

        for m in range(nr):
            if m == n:
                continue

            # relative vector to candidate neighbor
            dx = i[m] - i[n]
            dy = j[m] - j[n]
            dz = k[m] - k[n]
            d = math.sqrt(dx*dx + dy*dy + dz*dz)

            vec_to = np.array([dx, dy, dz], dtype=float)
            nn = np.linalg.norm(vec_to)
            if nn < 1e-12:
                continue
            vec_to /= nn

            # keep neighbor if within FOV
            if np.dot(self_vel, vec_to) >= fov_cos:
                items.append((d, ids[m]))

        # return ids of closest neighbors passing FOV
        smallest = sorted(items, key=lambda x: x[0])[:near_x]
        return [id_ for _, id_ in smallest]

    # predator reaction (kept but can be disabled)
    def bird_react_to_predator(self, bird_loc, pred_loc, effective_dist):
        # return avoidance vector if predator within effective_dist
        x, y, z = bird_loc
        i, j, k = pred_loc
        dist = np.sqrt((x - i)**2 + (y - j)**2 + (z - k)**2)
        if dist == 0:
            return np.array([0.0, 0.0, 0.0])
        direction = np.array([x - i, y - j, z - k], dtype=float) / dist
        closeness = 1.0 - (dist / effective_dist)
        beta = 5
        strength = (np.exp(beta * closeness) - 1.0) / (np.exp(beta) - 1.0)
        if dist <= effective_dist:
            min_speed, max_speed = 1.0, 2.0
            speed = min_speed + (max_speed - min_speed) * strength
            return speed * direction
        return np.array([0.0, 0.0, 0.0])

    # flock clustering helpers
    def get_positions_and_ids(self):
        # collect positions and ids for all birds in current order
        P = []
        ids = []
        for a in self.agents:
            x, y, z, vx, vy, vz, bird_id = a.output_last()
            P.append([x, y, z])
            ids.append(bird_id)
        return np.array(P, dtype=float), ids

    def get_components(self, P, R):
        # build connected components of a graph where edges exist if distance <= R
        """Connected components under distance threshold R (correct: compare to R^2)."""
        N = len(P)
        R2 = R * R
        adj = [[] for _ in range(N)]

        # pairwise neighbor graph construction
        for a in range(N):
            pa = P[a]
            for b in range(a + 1, N):
                d2 = np.sum((pa - P[b]) ** 2)
                if d2 <= R2:
                    adj[a].append(b)
                    adj[b].append(a)

        # DFS to extract components
        seen = np.zeros(N, dtype=bool)
        comps = []
        for start in range(N):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            comps.append(comp)
        return comps

    # PCA plane + flattening forces per flock
    def flock_plane_normal_and_com(self, P_sub):
        # compute COM and PCA normal (smallest eigenvector of covariance)
        com = P_sub.mean(axis=0)
        P0 = P_sub - com
        C = np.cov(P0.T, bias=True)
        w, V = np.linalg.eigh(C)
        normal = V[:, np.argmin(w)]
        normal = self.safe_unit(normal)
        return com, normal

    def flatten_forces_by_flock(self, P, ids, comps):
        # compute a pull toward the PCA plane for birds in each large component
        flat_forces = {}
        normals = {}

        for comp in comps:
            if len(comp) < self.min_flock_size:
                continue

            P_sub = P[comp]
            com, normal = self.flock_plane_normal_and_com(P_sub)

            for idx in comp:
                r = P[idx] - com
                dist = float(np.dot(r, normal))
                bird_id = ids[idx]
                normals[bird_id] = normal
                flat_forces[bird_id] = (-dist * normal) * self.flatten_strength

        return flat_forces, normals

    # compute I1<I2<I3 from extents in PCA basis
    def flock_dimensions_I_subset(self, P_sub):
        # project onto PCA axes, then take axis-aligned extents as I1,I2,I3
        P0 = P_sub - P_sub.mean(axis=0)
        C = np.cov(P0.T, bias=True)
        w, V = np.linalg.eigh(C)
        V = V[:, np.argsort(w)]
        proj = P0 @ V
        extents = [proj[:, a].max() - proj[:, a].min() for a in range(3)]
        I1, I2, I3 = sorted(extents)
        return I1, I2, I3

    # mean heading per flock
    def mean_heading_subset(self, idxs):
        # average velocities over birds and normalize
        V = np.array([self.agents[i].output_last()[3:6] for i in idxs], dtype=float)
        s = V.sum(axis=0)
        return self.safe_unit(s)

    # one simulation step
    def step(self):
        # snapshot positions/ids for neighbor search (same structure as original code)
        i, j, k, ids = [], [], [], []
        for agent in self.agents:
            x, y, z, vx, vy, vz, d = agent.output_last()
            i.append(x); j.append(y); k.append(z); ids.append(d)

        # predator movement (only active if timestep > pred_intro)
        if self.timestep == self.pred_intro:
            self.predator = Predator()
        if self.timestep > self.pred_intro:
            predx, predy, predz = self.predator.info()
            vec = np.array([i[0] - predx, j[0] - predy, k[0] - predz], dtype=float)
            vec = self.safe_unit(vec)
            if np.any(vec):
                movement = vec * math.sqrt(2)
                self.predator.update(predx + movement[0], predy + movement[1], predz + movement[2])

        # advance timestep counter
        self.timestep += 1

        # compute flock components and flattening forces
        P, ids2 = self.get_positions_and_ids()
        comps = self.get_components(P, self.cluster_R)
        flat_forces, _normals = self.flatten_forces_by_flock(P, ids2, comps)

        # record I-ratios only for non-turning flocks after validate_start_step
        if self.timestep >= self.validate_start_step and self.compute_I_every > 0 and (self.timestep % self.compute_I_every == 0):
            for comp in comps:
                if len(comp) < self.min_flock_size:
                    continue

                # use min index as a simple stable identifier for the component
                flock_key = min(comp)
                heading = self.mean_heading_subset(comp)

                # init deque for this flock if needed
                if flock_key not in self.flock_heading_hist:
                    self.flock_heading_hist[flock_key] = deque(maxlen=self.turn_smooth_window)

                hist = self.flock_heading_hist[flock_key]
                hist.append(heading)

                # wait until we have enough history to smooth
                if len(hist) < 3:
                    continue

                # compare latest heading to mean of previous headings
                prev = self.safe_unit(np.mean(list(hist)[:-1], axis=0))
                ang = self.angle_between(prev, hist[-1])

                # if turning too much, skip recording
                if ang > self.turn_threshold_rad:
                    continue

                # compute I1,I2,I3 and ratios for this flock
                P_sub = P[comp]
                I1, I2, I3 = self.flock_dimensions_I_subset(P_sub)
                r21 = (I2 / I1) if I1 > 1e-12 else np.nan
                r31 = (I3 / I1) if I1 > 1e-12 else np.nan

                # store measurement sample
                self.I_history.append({
                    "t": self.timestep,
                    "flock_key": flock_key,
                    "size": len(comp),
                    "I1": I1, "I2": I2, "I3": I3,
                    "r21": r21, "r31": r31,
                    "turn_angle_deg": float(np.rad2deg(ang)),
                })

                # optional debug printing for not-turning samples
                if self.dump_not_turning and (self.timestep % self.dump_every == 0):
                    print(f"[NOT TURNING] t={self.timestep} flock={flock_key} size={len(comp)} "
                          f"angle={np.rad2deg(ang):.2f}deg r21={r21:.3f} r31={r31:.3f}")

        # update birds using boids forces + wall + flattening
        for agent in self.agents:
            x, y, z, vx, vy, vz, bird_id = agent.output_last()

            # pick neighbors (FOV + nearest_x)
            nearest_ids = self.nearest_x_ids(i, j, k, ids, self.nearest_x, bird_id)

            # predator reaction term (usually zero if predator disabled)
            bird_loc = (x, y, z)
            effective_dist = 20
            if self.timestep > self.pred_intro and self.timestep <= self.pred_exit_time:
                pred_loc = self.predator.info()
                react_pred_vec = self.bird_react_to_predator(bird_loc, pred_loc, effective_dist)
            else:
                react_pred_vec = np.array([0.0, 0.0, 0.0])

            # compute cohesion/alignment/separation contributions
            cohesion_vec = self.cohesion(bird_id, nearest_ids)
            if np.any(cohesion_vec):
                cohesion_vec = (cohesion_vec / np.linalg.norm(cohesion_vec)) * self.coh_vector_scale

            alignment_vec = self.alignment(bird_id, nearest_ids)
            if np.any(alignment_vec):
                alignment_vec = (alignment_vec / np.linalg.norm(alignment_vec)) * self.ali_vector_scale

            seperation_vec = self.separation(bird_id, nearest_ids)
            if np.any(seperation_vec):
                seperation_vec = (seperation_vec / np.linalg.norm(seperation_vec)) * self.sep_vector_scale

            # add random noise term
            noise = np.random.normal(size=3)
            scaled_noise = noise / np.linalg.norm(noise) * self.noise_vector_scale

            # choose behavior: predator avoidance overrides boids terms
            if np.any(react_pred_vec):
                total_vec = react_pred_vec + scaled_noise
            else:
                total_vec = cohesion_vec + alignment_vec + seperation_vec + scaled_noise
                total_vec = total_vec / np.linalg.norm(total_vec)

            # add wall repulsion/containment
            wall = np.array(wall_vec(x, y, z, 5), dtype=np.float64)
            total_vec += wall

            # add flattening force for birds in large flocks
            total_vec += flat_forces.get(bird_id, np.array([0.0, 0.0, 0.0]))

            # renormalize direction
            total_vec = total_vec / np.linalg.norm(total_vec)

            # update agent position and velocity
            agent.set_current(
                x + total_vec[0],
                y + total_vec[1],
                z + total_vec[2],
                total_vec[0],
                total_vec[1],
                total_vec[2]
            )

        # commit current state to last state
        for agent in self.agents:
            agent.current_to_last()

    # visualization
    def show(self):
        # update scatter plot with current bird (and predator) positions
        xs, ys, zs = [], [], []
        for agent in self.agents:
            x, y, z, vx, vy, vz, bird_id = agent.output_last()
            xs.append(x); ys.append(y); zs.append(z)

        # include predator point if active
        if self.timestep > self.pred_intro:
            x, y, z = self.predator.info()
            xs.append(x); ys.append(y); zs.append(z)

        scat._offsets3d = (xs, ys, zs)
        fig.canvas.draw_idle()
        plt.pause(0.05)


# RUN (VALIDATION)
# predator disabled by setting pred_intro_time huge
sim = Simulation(
    N_birds=400,
    nearest_x=7,
    coh_vector_scale=0.3,
    ali_vector_scale=0.3,
    sep_vector_scale=0.3,
    noise_vector_scale=0.1,
    pred_intro_time=10_000,   # disable predator by never introducing it
    pred_exit_time=10_000,

    # flock flattening
    cluster_R=6.0,
    min_flock_size=25,
    flatten_strength=0.05,    # stronger flattener makes thinner flocks

    # validation gating
    validate_start_step=20,
    turn_threshold_deg=6.0,
    turn_smooth_window=4,
    compute_I_every=1,

    dump_not_turning=False,
    dump_every=50,
)

# set up interactive 3D plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_zlim(0, 100)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
scat = ax.scatter([], [], [], s=5)

# main loop
steps = 300
for _ in range(steps):
    sim.step()
    sim.show()

# close plot
plt.ioff()
plt.show()

# SUMMARY
# extract all not-turning ratio samples and compute 95% CI on the mean
r21 = [d["r21"] for d in sim.I_history if np.isfinite(d["r21"])]
r31 = [d["r31"] for d in sim.I_history if np.isfinite(d["r31"])]

def ci95(x):
    # 95% CI half-width for the mean using normal approximation
    if len(x) < 2:
        return np.nan
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))

print("\n=== ALL-FLOCK NOT-TURNING I-RATIO SUMMARY (WITH FLATTENER) ===")
print("samples:", len(r21), f"(t>={sim.validate_start_step}, only not turning)")

if len(r21) >= 2:
    print("I2/I1 = %.3f ± %.3f (95%% CI)" % (np.mean(r21), ci95(r21)))
else:
    print("I2/I1: not enough samples")

if len(r31) >= 2:
    print("I3/I1 = %.3f ± %.3f (95%% CI)" % (np.mean(r31), ci95(r31)))
else:
    print("I3/I1: not enough samples")

# compute per-flock averages grouped by flock_key
acc21 = defaultdict(list)
acc31 = defaultdict(list)
for d in sim.I_history:
    if np.isfinite(d["r21"]): acc21[d["flock_key"]].append(d["r21"])
    if np.isfinite(d["r31"]): acc31[d["flock_key"]].append(d["r31"])

print("\n=== PER-FLOCK AVERAGES (by flock_key) ===")
for k in sorted(acc21.keys()):
    if len(acc21[k]) >= 2:
        print(f"flock {k}: n={len(acc21[k])}  mean r21={np.mean(acc21[k]):.3f}  mean r31={np.mean(acc31[k]):.3f}")

'''results
=== ALL-FLOCK NOT-TURNING I-RATIO SUMMARY (WITH FLATTENER) ===
samples: 189 (t>=20, only not turning)
I2/I1 = 2.894 ± 0.088 (95% CI)
I3/I1 = 6.575 ± 0.319 (95% CI)

=== PER-FLOCK AVERAGES (by flock_key) ===
flock 0: n=165  mean r21=2.971  mean r31=6.893
flock 1: n=24  mean r21=2.368  mean r31=4.389
'''
