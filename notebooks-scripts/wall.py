#3D 100x100x100
# if the bird is coming close to the wall the flying direction gets opposite
# lets say  the range is 10 to any wall, and then it gets scaled the closer bird gets scaling will be increasing from 0 to 1 the closer it gets
# will combine the vectors if corners 

# so basically check the location and see how close it is to the wall, calculate the vector to the wall, return the scaled opposite vextor from the wall

def wall_vec(self, effective_distance):
    x, y, z = self.location

    def wall_distance(v):
        return v - 100 if v >= 50 else v

    dx = wall_distance(x)
    dy = wall_distance(y)
    dz = wall_distance(z)

    ax = 1 if abs(dx) <= effective_distance else 0
    ay = 1 if abs(dy) <= effective_distance else 0
    az = 1 if abs(dz) <= effective_distance else 0

    return (
        dx * ax / 10,
        dy * ay / 10,
        dz * az / 10
    )
