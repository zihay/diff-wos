
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.signal import savgol_filter
import matplotlib as mpl
from matplotlib import cm


@dataclass
class PlotConfig:
    titlesize: int = 20
    fontsize: int = 20
    figsize: tuple = (4, 4)
    title: str = ""
    plot_smooth: bool = True
    smooth_factor: int = 5
    plot_origin: bool = True
    color: str = None
    legend_fontsize: int = 15


@dataclass
class Plotter:
    xlim: List = None
    ylim: List = None
    start: int = 0
    end: int = None
    config: PlotConfig = field(default_factory=PlotConfig)

    def __post_init__(self):
        super().__post_init__()

    def run(self):
        matplotlib.rc('pdf', fonttype=42)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({'font.size': self.config.fontsize})
        fig = plt.figure(figsize=self.config.figsize)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
        self.plot()

        plt.title(self.config.title,
                  fontsize=self.config.titlesize)
        # Run plt.tight_layout() because otherwise the offset text doesn't update
        plt.tight_layout()
        # ax = plt.gca()
        # y_offset = ax.yaxis.get_offset_text().get_text()
        # ax.yaxis.offsetText.set_visible(False)
        # ax.text(0.01, 0.01, y_offset, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left')
        plt.savefig(Path(self.out_dir, self.out_name),
                    bbox_inches='tight')
        plt.close(fig)

    def draw(self, ax=None):
        plt.style.use('seaborn-whitegrid')
        if ax:
            plt.sca(ax)
        plt.rcParams.update({'font.size': self.config.fontsize})
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

        self.plot()
        if self.config.title:
            plt.title(self.config.title,
                      fontsize=self.config.titlesize)
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    def __call__(self):
        pass


@dataclass
class SinglePlotter(Plotter):
    src: List = None
    out_dir: str = "./"
    out_name: str = "plot.pdf"

    def __post_init__(self):
        # check data
        assert self.src is not None
        # check size
        if self.end is None:
            self.end = len(self.src)
        assert self.end <= len(self.src)

        self.smoothed = savgol_filter(self.src, self.config.smooth_factor, 3)

        # chunck
        self.src = self.src[self.start:self.end]
        self.smoothed = self.smoothed[self.start:self.end]

        # smoothed data
        # check xlim, ylim
        if self.xlim is None:
            self.xlim = (self.start, min(len(self.src), self.end))
        if self.ylim is None:
            self.ylim = (np.min(self.src), np.max(self.src))

    def plot(self):
        # called by run()
        if self.config.plot_smooth:
            plt.plot(np.arange(len(self.src)),
                     self.smoothed, color=self.config.color)
            if self.config.plot_origin:
                plt.plot(np.arange(len(self.src)), self.src, '-', alpha=.25)
        else:
            if self.config.plot_origin:
                plt.plot(np.arange(len(self.src)),
                         self.src, color=self.config.color)

        # if self.config.plot_smooth:
        #     plt.plot(np.arange(len(self.src)), self.src,
        #              '-', color='red', alpha=.25)
        #     plt.plot(np.arange(len(self.src)), self.smoothed)
        # else:
        #     plt.plot(np.arange(len(self.src)), self.src)

    def __call__(self):
        self.run()


@dataclass
class SinglePlotter2(Plotter):
    srcs: List[List] = None
    names: List[str] = None
    out_dir: str = "./"
    out_name: str = "plot.pdf"

    def __post_init__(self):
        # check data
        assert self.srcs is not None
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        # check size
        if self.end is None:
            self.end = len(self.srcs[0])
        assert self.end <= len(self.srcs[0])

        # clamp
        self.smooths = [savgol_filter(
            src, self.config.smooth_factor, 3) for src in self.srcs]
        self.srcs = [src[self.start:self.end] for src in self.srcs]
        self.smooths = [smooth[self.start:self.end] for smooth in self.smooths]

        if self.xlim is None:
            self.xlim = (self.start, min(len(self.srcs[0]), self.end))
        if self.ylim is None:
            self.ylim = (np.min(self.srcs[self.xlim[0]:self.xlim[1]]),
                         np.max(self.srcs[self.xlim[0]:self.xlim[1]]))

    def plot(self):
        if self.config.plot_smooth:
            for src, smooth, name in zip(self.srcs, self.smooths, self.names):
                plt.plot(np.arange(len(src)), smooth, label=name)
            if self.config.plot_origin:
                for src, name in zip(self.srcs, self.names):
                    plt.plot(np.arange(len(src)), src,
                             '-', alpha=.25)
        else:
            if self.config.plot_origin:
                for src, name in zip(self.srcs, self.names):
                    plt.plot(np.arange(len(src)), src, label=name)

        plt.legend(prop={'size': self.config.legend_fontsize})

    def __call__(self):
        self.run()


@dataclass
class ColorMap:
    vmin: float = -2.
    vmax: float = 2.
    cmap: str = 'cubicL'
    remap: bool = False

    def __post_init__(self):
        path = Path(os.path.dirname(__file__)) / 'cubicL.txt'
        self.cubicL = LinearSegmentedColormap.from_list(
            "cubicL", np.loadtxt(path), N=256)

    def __call__(self, value):
        if self.remap:
            value = np.multiply(np.sign(value), np.log1p(np.abs(value)))
        if self.cmap == 'cubicL':
            self.cmap = self.cubicL
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        m = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        return m.to_rgba(value)
