import numpy as np

"""
Given an agent's position in the 3D simulation plane, returns a wind vector adding to agent's movement 
Simulates constant circular wind with a southwest bias (wind coming from southwest, towards northwest)
As to simulate typical wind conditions in Dutch winter weather, when murmurations are often observed
"""
def wind_vec(x, y, z):

    # Southwest bias -> wind blows to northeast
    southwest_bias = np.array([1.0, 1.0, 0.0])  
    southwest_bias_magnitude = 1.5  
    
    # Circular wind in anti-clockwise swirl
    angle_strength = 0.05  
    circ_x = -y * angle_strength  # Negative y affects x
    circ_y = x * angle_strength   # Positive x affects y
    circ_z = 0.0  # no 3D rotation 
    
    circular_wind = np.array([circ_x, circ_y, circ_z])

    # Combine the wind components
    total_wind = southwest_bias * southwest_bias_magnitude + circular_wind

    # Map the wind vector back to the simulation scale
    return total_wind

test = wind_vec(50, 50, 50)
print("Wind vector at (50, 50, 50):", test)