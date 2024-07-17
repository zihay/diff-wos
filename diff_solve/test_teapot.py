from dataclasses import dataclass
from math import e
from select import epoll
from diff_solve.runner import TestRunner
from wos.fwd import *
from wos.io import read_2d_obj
from wos.scene import Detector, Polyline
from wos.wos import Baseline, WoS, WoSCUDA
from wos.wos_boundary import WoSBoundary
from wos.wos_grad import Baseline2, OursCUDA


@dataclass
class TestTeapot(TestRunner):
    def make_scene(self, delta=0):
        vertices, indices, values = read_2d_obj(
            basedir / 'data' / 'meshes' / 'teapot.obj', flip_orientation=False)
        vertices = Array2(vertices)
        vertices = vertices + Array2(0., 1.) * delta

        return Polyline(vertices=vertices,
                        indices=Array2i(indices),
                        values=Float(values))


if __name__ == '__main__':
    runner = TestTeapot(detector=Detector(vmin=(-1., -1.), vmax=(1., 1.), res=(512, 512)),
                        npasses=1,
                        # exclude_boundary=True,
                        wos_boundary=WoSBoundary(nwalks=100),
                        delta=Float(2e-3),
                        out_dir='./out/teapot')

    # runner.renderC(solver=WoS(nwalks=1000, nsteps=64),
    #                out_file="renderC.exr", npasses=10)
    runner.renderC(solver=WoSCUDA(nwalks=1000, nsteps=64),
                   out_file="renderC.exr", npasses=10)
    # runner.renderFD(solver=WoSCUDA(nwalks=1000, nsteps=64, epsilon=1e-4, prevent_fd_artifacts=True),
    #                 out_file="renderFD.exr", npasses=1000)
    # runner.renderD(solver=OursCUDA(nwalks=250, nsteps=64, epsilon=1e-3,
    #                                epsilon2=1e-4, clamping=1e-1),
    #                out_file="ours_high_spp.exr", npasses=1000)
    # runner.renderD(solver=Baseline(nwalks=250, nsteps=64),
    #                out_file="baseline.exr", npasses=5)
    # runner.renderD(solver=Baseline2(nwalks=250, nsteps=64, epolison=1e-3),
    #                out_file="baseline2.exr", npasses=10)
    # runner.renderD(solver=OursCUDA(nwalks=250, nsteps=64, epsilon=1e-3,
    #                                epsilon2=1e-4, clamping=1e-1),
    #                out_file="ours.exr", npasses=10)
