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
Test for range of neighbourhood size 2-7, run for 600 steps in total, 
predator introduced at 100 steps and exits at 500, 
save final coordinate data in for DBSCAN analysis
"""
final_positions = {}

for nearest_x in range(2, 8):
    final_coords = get_final_coordinates(300, 7, 0.3, 0.3, 0.3, 0.1, 100, 500, 600)

    final_positions[nearest_x] = final_coords

for nearest_x, positions in final_positions.items():
    np.save(f"neighbours_{nearest_x}.npy", positions)

# test 
coords = np.load("neighbours_3.npy")
print(coords.shape)  
print(coords[:5])     



