from simulation_class import Simulation 
import numpy as np 

# save final position of all agents
def get_final_coordinates(
        N_birds,            # Flock size
        nearest_x,          # Effective neighbours
        coh_vector_scale, 
        ali_vector_scale, 
        sep_vector_scale, 
        noise_vector_scale, 
        pred_intro_time,    
        pred_exit_time, 
        n_steps
        ):
    
    sim = Simulation(N_birds, nearest_x, coh_vector_scale, ali_vector_scale, sep_vector_scale, noise_vector_scale, pred_intro_time, pred_exit_time)
    for step in range(n_steps):
        sim.step()

    xs, ys, zs= sim.dump()

    ids = np.arange(len(xs))   
    final_coordinates = np.column_stack((ids, xs, ys, zs))
    return final_coordinates

"""
Test for range of neighbourhood size 2-11, run for 601 steps in total, 
predator introduced at 100 steps and exits at 600, 
save final coordinate data at step 601 in csv for DBSCAN analysis
Repeat for total 30 runs each configuration
"""
rows_save = []

N_RUNS = 30

for nearest_x in range(2, 12):
    for run in range(N_RUNS):
        print(f"run={run+1}, nearest_x={nearest_x}")
        final_coords = get_final_coordinates(
            200,
            nearest_x,
            0.3,
            0.3,
            0.3,
            0.1,
            100,
            600,
            601
        )

        for row in final_coords:
            rows_save.append([
                nearest_x,
                run,
                int(row[0]),  # agent_id
                row[1],
                row[2],
                row[3]
            ])
    print("-end-")

rows_save = np.array(rows_save)

np.savetxt(
    "final_positions.csv",
    rows_save,
    delimiter=",",
    header="nearest_x,run,agent_id,x,y,z",
)