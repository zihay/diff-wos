from dataclasses import dataclass
from os import write
from pathlib import Path
import matplotlib as mpl
from matplotlib import cm
from matplotlib.pyplot import hist

import torch
import drjit as dr
from wos.solver import Solver
from wos.io import write_exr
from torch.utils.tensorboard import SummaryWriter
from wos.fwd import *


@dataclass
class Writer:
    out_dir: str = "./"

    def __post_init__(self):
        self.writer = SummaryWriter(self.out_dir)
        self.history = {}

    def add_scalar(self, name, value, step):
        if type(value) == torch.Tensor:
            value = value.cpu().numpy()
        self.writer.add_scalar(name, value, step)
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(value)
        np.savetxt(Path(self.out_dir) / "{}.txt".format(name),
                   np.array(self.history[name]))

    def add_tensor(self, name, value, step):
        assert (type(value) == np.ndarray)
        value = np.squeeze(value)
        self.writer.add_tensor(name, torch.tensor(value), step)
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(value)
        np.savetxt(Path(self.out_dir) / "{}.txt".format(name),
                   np.array(self.history[name]))

    def add_image(self, name, value, step, vmin=-2., vmax=2., cmap=cm.jet):
        assert (type(value) == np.ndarray)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        cimage = m.to_rgba(value)[:, :, :3]
        self.writer.add_image(name, cimage, step, dataformats='HWC')
        path = Path(self.out_dir) / name
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        write_exr(path / "{:04d}.exr".format(step), value)


@dataclass
class Model:
    def forward(self, i=0):
        raise NotImplementedError

    def report(self, i, out_dir=None):
        raise NotImplementedError


@dataclass
class TrainRunner:
    name: str = None
    out_dir: str = "./"
    model_target: Model = None
    model: Model = None
    pts: torch.Tensor = None
    niters: int = 1000
    is_l1: bool = True
    is_mask: bool = True

    def __post_init__(self):
        print("Initializing training runner...")
        print("Rendering target image...")
        with dr.suspend_grad():
            self.target_image = dr.detach(self.model_target.forward())
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.writer = Writer(self.out_dir)
        with dr.suspend_grad():
            image = self.model_target.render()  # test render
        write_exr(self.out_dir / "target.exr", image.numpy())

    def loss(self, image, target_image):
        image = dr.ravel(image)
        target_image = dr.ravel(target_image)
        if self.is_mask:
            mask = (dr.abs(image) > 0.001) & (dr.abs(target_image) > 0.001)
            image = dr.select(mask, image, 0.)
            target_image = dr.select(mask, target_image, 0.)
        if self.is_l1:
            return dr.sum(dr.ravel(dr.abs(image - target_image)))
        else:
            return dr.sum(dr.squared_norm(dr.ravel(image - target_image)))

    def run(self):
        for i in range(self.niters):
            print("Iteration: {}".format(i))

            image = self.model.forward(i)
            loss = self.loss(image, self.target_image)
            dr.backward(loss)

            # log
            param_error = torch.mean(torch.abs(self.model.parameters() -
                                               self.model_target.parameters()))
            print("Loss: {}".format(loss))
            print("param error: {}".format(param_error))
            self.writer.add_scalar("loss", loss.torch(), i)
            self.writer.add_scalar("param_error", param_error, i)
            self.model.report(i, self.writer)

            # step the optimizer
            self.model.step()

    def report(self):
        raise NotImplementedError
