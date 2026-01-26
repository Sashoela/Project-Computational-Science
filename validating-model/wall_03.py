#3D 100x100x100
# if the bird is coming close to the wall the flying direction gets opposite
# lets say  the range is 10 to any wall, and then it gets scaled the closer bird gets scaling will be increasing from 0 to 1 the closer it gets
# will combine the vectors if corners 

# so basically check the location and see how close it is to the wall, calculate the vector to the wall, return the scaled opposite vextor from the wall

import numpy as np 

"""
Added comments:
"effective distance" = range of influence. At what point does the wall start to influence the direction of the bird 
The closer the bird is to the wall, the stronger the repuslive force 
Fixing wall issues: 

"""

def wall_vec(x, y, z, effective_distance=10, box_size=100):
    repulsion = np.zeros(3)

    # X walls
    if x < effective_distance:
        repulsion[0] = (effective_distance - x) / effective_distance
    elif x > box_size - effective_distance:
        repulsion[0] = -((x - (box_size - effective_distance)) / effective_distance)

    # Y walls
    if y < effective_distance:
        repulsion[1] = (effective_distance - y) / effective_distance
    elif y > box_size - effective_distance:
        repulsion[1] = -((y - (box_size - effective_distance)) / effective_distance)

    # Z walls
    if z < effective_distance:
        repulsion[2] = (effective_distance - z) / effective_distance
    elif z > box_size - effective_distance:
        repulsion[2] = -((z - (box_size - effective_distance)) / effective_distance)

    return repulsion
