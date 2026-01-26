import pyvista as pv
import numpy as np

# --- PyVista Viewer ---
class PyVistaViewer:
    def __init__(self, sim):
        self.sim = sim
        self.plotter = pv.Plotter()
        self.plotter.add_axes()
        self.plotter.set_background("black")

        # Birds
        positions = np.array([agent.output_last()[:3] for agent in sim.agents])
        self.cloud = pv.PolyData(positions)
        self.actor = self.plotter.add_points(
            self.cloud, render_points_as_spheres=True, point_size=6, color="white"
        )

        # Predator
        self.predator_mesh = pv.Sphere(radius=2.0)
        self.predator_actor = self.plotter.add_mesh(self.predator_mesh, color="red")
        self.predator_actor.SetVisibility(False)

        self.plotter.show(interactive_update=True)

    def update(self):
        positions = np.array([agent.output_last()[:3] for agent in self.sim.agents])
        self.cloud.points = positions

        if self.sim.predator_active():
            x, y, z = self.sim.predator.info()
            self.predator_actor.SetPosition(x, y, z)
            self.predator_actor.SetVisibility(True)
        else:
            self.predator_actor.SetVisibility(False)

        self.plotter.update()
