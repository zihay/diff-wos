from dataclasses import dataclass
from inv_solve.optimize import TrainRunner, Writer
from wos.fwd import *
from wos.io import read_3d_obj
from wos.scene3d import Detector3D, Scene3D
from wos.solver import CustomSolver, Solver
from wos.utils import concat, rotate_euler
from wos.wos3d import WoS3DCUDA
from wos.wos_grad_3d import Baseline3D, Ours3D


@dataclass
class Model:
    solver: Solver = None
    vertices: Array2 = None
    values: Float = None
    rotation: Float = None
    translation: Float = None
    optimizer: mi.ad.Adam = None
    npasses: int = 1
    test_detector: Detector3D = None

    def make_points(self) -> Array2:
        raise NotImplementedError

    def __post_init__(self):
        # initial vertices
        vertices, indices, values = read_3d_obj(
            basedir / 'data' / 'meshes' / 'bunny2.obj')
        self.values = Float(values)
        self.vertices = Array3(vertices)
        self.indices = Array3i(indices)
        # init optimizer
        self.optimizer = mi.ad.Adam(
            lr=0.02, params={'rotation': self.rotation})

    def parameters(self):
        return torch.concatenate([self.optimizer['rotation'].torch()])

    def make_scene(self):
        raise NotImplementedError

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

    def render(self, i=0):
        # for testing
        print("render test view")
        pts = self.test_detector.make_points()
        image = self.solver.walk(pts, self.make_scene(), seed=i)
        image = Tensor(dr.ravel(image), shape=self.test_detector.res)
        return image

    def report(self, i, writer: Writer):
        writer.add_tensor('rotation', self.optimizer['rotation'].numpy(), i)
        print('rotation: ', self.optimizer['rotation'])
        image = self.render(i)
        writer.add_image('image', image.numpy().reshape(
            self.test_detector.res), i)


@dataclass
class ModelExterior(Model):
    res: int = 128

    def make_points(self):
        '''
        make points on the boundary of the shape
        '''
        size = 1.
        x = np.linspace(-size, size, self.res)
        x, y, z = np.meshgrid(x, x, x)
        mask = np.abs(
            (np.max([np.abs(x), np.abs(y), np.abs(z)], axis=0)) - size) < 1e-4
        x = x[mask]
        y = y[mask]
        z = z[mask]
        return Array3(np.array([x, y, z]).T)

    def make_scene(self):
        # bounding cube
        cube_vertices = Array3(np.array([[1., -1., -1.],
                                         [1., -1., 1.],
                                         [-1., -1., 1.],
                                         [-1., -1., -1.],
                                         [1., 1., -1.],
                                         [1., 1., 1.],
                                         [-1., 1., 1.],
                                         [-1., 1., -1.]])) * 2.
        cube_indices = Array3i(np.array([[1, 2, 3],
                                         [7, 6, 5],
                                         [4, 5, 1],
                                         [5, 6, 2],
                                         [2, 6, 7],
                                         [0, 3, 7],
                                         [0, 1, 3],
                                         [4, 7, 5],
                                         [0, 4, 1],
                                         [1, 5, 2],
                                         [3, 2, 7],
                                         [4, 0, 7]]))
        cube_values = dr.repeat(Float(0.), dr.width(cube_vertices))
        shape_vertices = self.vertices * 0.8
        shape_vertices = rotate_euler(
            shape_vertices, self.optimizer['rotation'])
        vertices = concat(shape_vertices, cube_vertices)
        indices = concat(self.indices, cube_indices + dr.width(self.vertices))
        values = concat(self.values, cube_values)
        return Scene3D(vertices=vertices,
                       indices=indices,
                       values=values,
                       use_bvh=True)


def run(gradient_solver, out_dir):
    runner = TrainRunner(
        out_dir=out_dir,
        model_target=ModelExterior(
            rotation=Array3(0.),
            translation=Array3(0.),
            solver=WoS3DCUDA(nwalks=100, nsteps=64, double_sided=True),
            test_detector=Detector3D(res=(64, 64), z=Float(-0.05)),
            npasses=10),
        model=ModelExterior(
            rotation=Array3(dr.pi / 4),
            translation=Array3(0.),
            solver=CustomSolver(fwd_solver=WoS3DCUDA(nwalks=50, nsteps=64, double_sided=True),
                                bwd_solver=gradient_solver),
            test_detector=Detector3D(res=(64, 64), z=Float(-0.05))),
        niters=201,
        is_mask=False)
    runner.run()


if __name__ == '__main__':
    run(Ours3D(nwalks=2, nsteps=64, double_sided=True,
               epsilon2=5e-4, clamping=1e-1), 'out/bunny/ours')
    # run(Baseline3D(nwalks=1, nsteps=64, double_sided=True),
    #     'out/bunny/baseline')
