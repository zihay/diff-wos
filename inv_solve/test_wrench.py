from dataclasses import dataclass
from inv_solve.optimize import TrainRunner, Writer
from wos.fwd import *
from wos.io import read_2d_obj
from wos.scene import Polyline
from wos.solver import CustomSolver, Solver
from wos.utils import concat, rotate
from wos.wos import Baseline, WoSCUDA
from wos.wos_grad import OursCUDA


@dataclass
class Model:
    solver: Solver = None
    vertices: Array2 = None
    values: Float = None
    rotation: Float = None
    translation: Array2 = None
    optimizer: mi.ad.Adam = None
    npasses: int = 1
    test_res: tuple = (512, 512)
    res: int = 1024

    def make_points(self) -> Array2:
        '''
        make points on the boundary of the shape
        '''
        size = 1.
        x = np.linspace(-size, size, self.res)
        x, y = np.meshgrid(x, x)
        mask = np.abs(
            (np.max([np.abs(x), np.abs(y)], axis=0)) - size) < 1e-4
        x = x[mask]
        y = y[mask]
        return Array2(np.array([x, y]).T)

    def __post_init__(self):
        # initial vertices
        vertices, indices, values = read_2d_obj(basedir / 'data' / 'meshes' / 'wrench.obj',
                                                flip_orientation=True)
        self.values = Float(values)
        self.vertices = Array2(vertices)
        self.indices = Array2i(indices)
        # init optimizer
        self.optimizer = mi.ad.Adam(
            lr=0.06, params={'rotation': self.rotation})
        self.optimizer2 = mi.ad.Adam(
            lr=0.00001, params={'translation': self.translation})

    def parameters(self):
        return torch.concatenate([self.optimizer['rotation'].torch().reshape(-1),
                                  self.optimizer2['translation'].torch().reshape(-1)])

    def make_scene(self):
        cube_vertices = Array2(
            np.array([[-1., -1.], [-1., 1.], [1., 1.], [-1., 1.]]) * 5.)
        cube_indices = Array2i(np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
        cube_values = Float([0., 0., 0., 0.])
        shape_vertices = rotate(self.vertices, self.optimizer['rotation'])
        shape_vertices = shape_vertices + self.optimizer2['translation']
        vertices = concat(shape_vertices, cube_vertices)
        indices = concat(self.indices, cube_indices + dr.width(self.vertices))
        values = concat(self.values, cube_values)
        return Polyline(vertices=vertices,
                        indices=indices,
                        values=values)

    def forward(self, i=0):
        pts = self.make_points()
        scene = self.make_scene()
        image = Float(0.)
        for j in range(self.npasses):
            image += self.solver.walk(pts, scene, seed=1000 * i + j)
            dr.eval(image)
        image = image / self.npasses
        return image

    def step(self):
        self.optimizer.step()
        self.optimizer2.step()

    def render(self, i=0):
        # for testing
        x = dr.linspace(Float, -1., 1., self.test_res[0])
        y = dr.linspace(Float, 1., -1., self.test_res[1])
        pts = Array2(dr.meshgrid(x, y))
        image = self.solver.walk(pts, self.make_scene(), seed=i)
        image = Tensor(dr.ravel(image), shape=self.test_res)
        return image

    def report(self, i, writer: Writer):
        writer.add_tensor('rotation', self.optimizer['rotation'].numpy(), i)
        writer.add_tensor(
            'translation', self.optimizer2['translation'].numpy(), i)
        print('rotation: ', self.optimizer['rotation'])
        print('translation: ', self.optimizer2['translation'])
        image = self.render(i)
        writer.add_image('image', image.numpy().reshape(self.test_res), i)


def run(gradient_solver, out_dir):
    runner = TrainRunner(
        out_dir=out_dir,
        model_target=Model(rotation=Float(dr.pi / 2),
                           translation=Array2(0., 0.),
                           solver=WoSCUDA(nwalks=1000, nsteps=64,
                                          double_sided=True),
                           npasses=10),
        model=Model(rotation=Float(0.),
                    translation=Array2(0., 0.),
                    solver=CustomSolver(fwd_solver=WoSCUDA(nwalks=100, nsteps=64, double_sided=True),
                                        bwd_solver=gradient_solver)),
        niters=200,
        is_mask=False)
    runner.run()


if __name__ == '__main__':
    run(OursCUDA(nwalks=10, nsteps=64, double_sided=True,
                 epsilon2=5e-4, clamping=1e-1), 'out/wrench/ours')
    # run(Baseline(nwalks=5, nsteps=64, double_sided=True),
    #     'out/wrench/baseline')
