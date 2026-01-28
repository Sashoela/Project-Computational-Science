from simulation_class import Simulation
import numpy as np


def get_coordinates_at_steps(
    N_birds,
    nearest_x,
    coh_vector_scale,
    ali_vector_scale,
    sep_vector_scale,
    noise_vector_scale,
    pred_intro_time,
    pred_exit_time,
    sample_steps,
):
    """
    Run ONE simulation and return snapshots at the requested timesteps.
    Returns: dict {step: np.array([[id,x,y,z], ...])}
    """
    sim = Simulation(
        N_birds,
        nearest_x,
        coh_vector_scale,
        ali_vector_scale,
        sep_vector_scale,
        noise_vector_scale,
        pred_intro_time,
        pred_exit_time,
    )

    sample_steps = sorted(set(int(s) for s in sample_steps))
    snapshots = {s: None for s in sample_steps}
    max_step = max(sample_steps)

    # step index here matches what YOU want to label as "step"
    # (i.e., after calling sim.step() we treat it as that step count)
    for step in range(1, max_step + 1):
        sim.step()

        if step in snapshots:
            xs, ys, zs = sim.dump()
            ids = np.arange(len(xs), dtype=int)
            snapshots[step] = np.column_stack((ids, xs, ys, zs))

    return snapshots


"""
Run for a range of neighbourhood sizes, repeat runs,
save coordinate snapshots at multiple steps into ONE CSV with a step column.
"""
rows_save = []

N_RUNS = 12
SAMPLE_STEPS = [301, 311, 321]

# parameters you had
N_BIRDS = 200
COH = 0.3
ALI = 0.3
SEP = 0.3
NOISE = 0.1
PRED_INTRO = 15
PRED_EXIT = 300

for nearest_x in range(3, 11):
    for run in range(N_RUNS):
        print(f"run={run+1}, nearest_x={nearest_x}")

        snapshots = get_coordinates_at_steps(
            N_BIRDS,
            nearest_x,
            COH,
            ALI,
            SEP,
            NOISE,
            PRED_INTRO,
            PRED_EXIT,
            SAMPLE_STEPS,
        )

        for step in SAMPLE_STEPS:
            coords = snapshots[step]
            if coords is None:
                raise RuntimeError(f"Missing snapshot for step={step}")

            for row in coords:
                rows_save.append([
                    nearest_x,
                    run,
                    step,          # <-- timestep column
                    int(row[0]),    # agent_id
                    row[1],         # x
                    row[2],         # y
                    row[3],         # z
                ])

    print("-end-")

rows_save = np.array(rows_save, dtype=float)

np.savetxt(
    "final_positions_multi_step.csv",
    rows_save,
    delimiter=",",
    header="nearest_x,run,step,agent_id,x,y,z",
    comments=""
)
