class Agent():
    def __init__(self, id):
        self.id = id
        # all variables for speed and location 
        self.last_x = 0
        self.last_y = 0
        self.last_z = 0

        self.current_x = 0
        self.current_y = 0
        self.current_z = 0

        self.last_vx = 0
        self.last_vy = 0
        self.last_vz = 0

        self.current_vx = 0
        self.current_vy = 0
        self.current_vz = 0

    def setup(self, x, y, z, vx, vy, vz):
        # function to call when initializing model with random values
        self.last_x = x
        self.last_y = y
        self.last_z = z
        self.last_vx = vx
        self.last_vy = vy
        self.last_vz = vz

    def output_last(self):
        # output all values to compute with
        return self.last_x, self.last_y, self.last_z, self.last_vx, self.last_vy, self.last_vz, self.id
    
    def get_id(self):
        return self.id
    
    def set_current(self, x, y, z, vx, vy ,vz):
        # save new data, made permanent in current_to_last
        self.current_x = x
        self.current_y = y
        self.current_z = z
        self.current_vx = vx
        self.current_vy = vy
        self.current_vz = vz

    def current_to_last(self):
        # save "new" data to "old" data
        self.last_x = self.current_x
        self.last_y = self.current_y
        self.last_x = self.current_z
        self.last_vx = self.current_vx
        self.last_vy = self.current_vy
        self.last_vx = self.current_vz



class Predator():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

    def info(self):
        return self.x, self.y, self.z
    
    def update(self, dx, dy, dz):
        self.x = dx
        self.y = dy
        self.z = dz