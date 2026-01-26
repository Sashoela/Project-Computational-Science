
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