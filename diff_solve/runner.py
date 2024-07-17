from dataclasses import dataclass, field
from typing import List
from wos.fwd import *
from wos.scene import Detector
from wos.solver import Solver
from wos.io import write_exr
from wos.wos_boundary import WoSBoundary


@dataclass
class Task:
    solver: Solver = None
    out_file: str = None
    npasses: int = None


@dataclass
class RenderC(Task):
    out_file: str = 'renderC.exr'


@dataclass
class RenderD(Task):
    out_file: str = 'renderD.exr'


@dataclass
class RenderFD(Task):
    out_file: str = 'renderFD.exr'


@dataclass
class TestRunner:
    detector: Detector = field(default_factory=Detector)
    tasks: List[Task] = field(default_factory=list)
    out_dir: str = './out'
    delta: Float = Float(1e-2)
    npasses: int = 1
    exclude_boundary: bool = False
    wos_boundary: WoSBoundary = None
    pixel_sampling: bool = True

    def __post_init__(self):
        self.p = self.detector.make_points()
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def make_scene(self, delta=0.):
        '''
        This function returns a transformed scene by encoding both 
        the scene and its transformation within a single function.
        '''
        raise NotImplementedError

    def renderC(self, solver, out_file='renderC.exr', npasses=None):
        scene = self.make_scene()
        image = Float(0.)
        if npasses is None:
            npasses = self.npasses
        for i in range(npasses):
            print('pass: ', i)
            if self.pixel_sampling:
                image += solver.walk_detector(scene, self.detector, seed=i)
            else:
                image += solver.walk(self.p, scene, seed=i)
            _image = image / (i + 1)
            _image = _image.numpy().reshape(self.detector.res)
            self.detector.save(_image, Path(self.out_dir) / out_file)
        is_boundary = dr.abs(scene.sdf(self.p)) < self.delta + 1e-3
        if self.exclude_boundary:
            image[is_boundary] = 0.
        image = image.numpy().reshape(self.detector.res) / npasses
        self.detector.save(image, Path(self.out_dir) / out_file)
        return image

    def renderD(self, solver, out_file='renderD.exr', npasses=None):
        d_image = Float(0.)
        if npasses is None:
            npasses = self.npasses
        for i in range(npasses):
            print('pass: ', i)
            delta = Float(0.)
            dr.enable_grad(delta)
            dr.set_grad(delta, 1.)
            scene = self.make_scene(delta)
            if self.pixel_sampling:
                image = solver.walk_detector(scene, self.detector, seed=i)
            else:
                image = solver.walk(self.p, scene, seed=i)
            if self.wos_boundary is not None:
                image += self.wos_boundary.walk_detector(
                    scene, self.detector, seed=i)
            dr.forward_to(image)
            d_image += dr.grad(image)
            dr.eval(d_image)
            # save the image
            _image = d_image
            if self.exclude_boundary:
                is_boundary = dr.abs(scene.sdf(self.p)) < self.delta + 1e-3
                _image = dr.select(~is_boundary, _image, 0.)
            self.detector.save(_image.numpy().reshape(self.detector.res) / (i + 1),
                               Path(self.out_dir) / out_file)

        is_boundary = dr.abs(scene.sdf(self.p)) < self.delta + 1e-3
        if self.exclude_boundary:
            d_image[is_boundary] = 0.
        d_image = d_image.numpy().reshape(self.detector.res) / npasses
        self.detector.save(d_image,
                           Path(self.out_dir) / out_file)
        return d_image

    def renderFD(self, solver, out_file='renderFD.exr', npasses=None):
        scene1 = self.make_scene()
        scene2 = self.make_scene(self.delta)
        dimage = Float(0.)
        if npasses is None:
            npasses = self.npasses
        for i in range(npasses):
            print('pass: ', i)
            image1 = solver.walk_detector(scene1, self.detector, seed=i)
            print(image1.numpy().sum())
            image2 = solver.walk_detector(scene2, self.detector, seed=i)
            print(image2.numpy().sum())
            dimage += (image2 - image1) / self.delta
            # save image
            _image = dimage
            if self.exclude_boundary:
                is_boundary = dr.abs(scene1.sdf(self.p)) < self.delta + 1e-3
                _image = dr.select(~is_boundary, _image, 0.)
            write_exr(Path(self.out_dir) / out_file,
                      _image.numpy().reshape(self.detector.res) / (i + 1))
        is_boundary = dr.abs(scene1.sdf(self.p)) < self.delta + 1e-3
        if self.exclude_boundary:
            dimage[is_boundary] = 0.
        dimage = dimage.numpy().reshape(self.detector.res) / npasses
        write_exr(Path(self.out_dir) / out_file, dimage)
        return dimage

    def run_task(self, task):
        if isinstance(task, RenderC):
            return self.renderC(task.solver, task.out_file, task.npasses)
        elif isinstance(task, RenderD):
            return self.renderD(task.solver, task.out_file, task.npasses)
        elif isinstance(task, RenderFD):
            return self.renderFD(task.solver, task.out_file, task.npasses)
        else:
            raise NotImplementedError

    def run_task_idx(self, idx):
        self.run_task(self.tasks[idx])

    def run(self):
        for task in self.tasks:
            print('running task: ', task)
            self.run_task(task)
