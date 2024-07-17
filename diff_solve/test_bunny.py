import argparse
from dataclasses import dataclass
from drjit import shape

from matplotlib import pyplot as plt
from diff_solve.runner import TestRunner
from wos.fwd import *
from wos.io import read_2d_obj, read_3d_obj
from wos.scene import Detector, Polyline
from wos.scene3d import Detector3D, Scene3D
from wos.solver import Solver
from wos.io import write_exr
from wos.utils import concat, rotate_euler
from wos.wos import WoSCUDA
from wos.wos3d import WoS3D, WoS3DCUDA
from wos.wos_boundary import WoSBoundary
from wos.wos_grad_3d import Baseline23D, Baseline3D, Ours3D


@dataclass
class TestBunny(TestRunner):
    def make_scene(self, delta=0):
        vertices, indices, values = read_3d_obj(
            basedir / 'data' / 'meshes' / 'bunny.obj')
        vertices = Array3(vertices)
        # values *= 3.
        vertices = vertices + Array3(1., 0., 0.) * delta
        scene = Scene3D(vertices=vertices,
                        indices=Array3i(indices),
                        values=Float(values),
                        use_bvh=False)
        return scene


if __name__ == '__main__':
    runner = TestBunny(detector=Detector3D(res=(512, 512), z=Float(-0.05)),
                       npasses=1,
                       # exclude_boundary=True,
                       wos_boundary=WoSBoundary(nwalks=100),
                       delta=Float(3e-3),
                       out_dir='./out/bunny')

    # runner.renderC(solver=WoS(nwalks=1000, nsteps=64),
    #                out_file="renderC.exr", npasses=10)
    runner.renderC(solver=WoS3DCUDA(nwalks=250, nsteps=64),
                   out_file="renderC.exr", npasses=10)
    # runner.renderFD(solver=WoS3DCUDA(nwalks=100, nsteps=64, epsilon=1e-4, #prevent_fd_artifacts=True
    #                              ),
    #                 out_file="renderFD.exr", npasses=1000)
    # runner.renderD(solver=Ours3D(nwalks=10, nsteps=64,
    #                              epsilon2=2e-4, clamping=5e-2),
    #                out_file="ours_high_spp.exr", npasses=1000)
    # runner.renderD(solver=Baseline3D(nwalks=10, nsteps=64),
    #                out_file="baseline.exr", npasses=5)
    # runner.renderD(solver=Baseline23D(nwalks=10, nsteps=64, epolison=1e-3, l=5e-3),
    #                out_file="baseline2.exr", npasses=10)
    # runner.renderD(solver=Ours3D(nwalks=10, nsteps=64, epsilon2=5e-4, clamping=5e-2),
    #                out_file="ours.exr", npasses=10)
