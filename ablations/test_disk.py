from dataclasses import dataclass
from matplotlib import pyplot as plt
import matplotlib
from wos.tools import ColorMap
from wos.fwd import *
from wos.scene import ClosestPointRecord
from wos.scene import Circle
from wos.stats import Statistics
from wos.io import write_image
from wos.utils import plot_ci
from wos.wos_with_source import WoSWithSource
matplotlib.rc('pdf', fonttype=42)


@dataclass
class Disk(Circle):
    values: Float = Float(0.)
    n: int = 4  # frequency

    def largest_inscribed_ball(self, its: ClosestPointRecord):
        return dr.repeat(Float(0.5), dr.width(its.p))

    def solution(self, p):
        r = dr.norm(p)
        theta = dr.atan2(p.y, p.x)
        return dr.power(r, self.n) * dr.cos(self.n * theta)

    def source_function(self, p):
        return dr.repeat(Float(0.), dr.width(p))

    def dirichlet(self, its: ClosestPointRecord):
        t = its.t
        return dr.cos(self.n * t)

    def normal_derivative(self, its: ClosestPointRecord):
        t = its.t
        return -self.n * dr.cos(self.n * t)


@dataclass
class Test:
    solver: WoSWithSource = None
    nsamples: int = 1000
    clamping: float = 1e-2
    out_dir: Path = Path('out/disk')
    res: int = 512

    def __post_init__(self):
        self.out_dir.mkdir(exist_ok=True, parents=True)

    def make_scene(self, delta=0):
        scene = Disk(center=Array2(0., delta), radius=1.)
        return scene

    def make_points(self):
        x = dr.linspace(Float, -1., 1., self.res)
        y = dr.linspace(Float, 1., -1., self.res)
        p = Array2(dr.meshgrid(x, y))
        return p

    def solution(self):
        scene = self.make_scene()
        p = self.make_points()
        d = scene.sdf(p)
        is_inside = d < 0.
        image = scene.solution(p)
        image[~is_inside] = 0.
        image = image.numpy().reshape((self.res, self.res))
        return image

    def render(self):
        scene = self.make_scene()
        pts = self.make_points()
        image = self.solver.walk(pts, scene, seed=0)
        image = image.numpy().reshape((self.res, self.res))
        return image

    def plot_shape(self):
        scene = self.make_scene()
        vertices = scene.vertices.numpy()
        vertices = np.vstack([vertices, vertices[0]])
        plt.plot(*vertices.T, '-', alpha=1.)

    def make_boundary_intersection(self, size=1):
        p = Array2(0.5, 0.5)
        p = dr.repeat(p, size)
        scene = self.make_scene()
        its = self.solver.single_walk_preliminary(
            p, scene, PCG32(size=1, initstate=1))
        return its

    def plot(self):
        p = Array2(0., 0.)
        its = self.make_boundary_intersection()
        image = self.solution()
        color_map = ColorMap(vmin=-2., vmax=2.)
        cimage = color_map(image)
        mask = np.abs(image) < 1e-5
        cimage[mask] = 0.
        write_image(self.out_dir / "ablations.png", cimage, is_srgb=False)
        plt.imshow(cimage, vmin=-1., vmax=1., extent=[-1., 1., -1., 1.])
        plt.plot(its.p.x, its.p.y, 'go')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def show_solution(self):
        image = self.solution()
        plt.imshow(image, vmin=-1., vmax=1., extent=[-1., 1., -1., 1.])
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # normal derivative estimator

    def ground_truth(self):
        its = self.make_boundary_intersection()
        scene = self.make_scene()
        grad_n = scene.normal_derivative(its)
        return grad_n

    def ours(self):
        its = self.make_boundary_intersection(size=self.nsamples)
        scene = self.make_scene()
        sampler = PCG32(size=dr.width(its.p), initstate=0)
        grad_n = self.solver.normal_derivative(
            its, scene, sampler, clamping=self.clamping)
        grad_n = dr.dot(Array2(grad_n), its.n)
        return grad_n

    def ours_no_anti(self):
        its = self.make_boundary_intersection(size=self.nsamples)
        scene = self.make_scene()
        sampler = PCG32(size=dr.width(its.p), initstate=0)
        grad_n = self.solver.normal_derivative(
            its, scene, sampler, clamping=self.clamping, antithetic=False)
        grad_n = dr.dot(Array2(grad_n), its.n)
        return grad_n

    def ours_no_control_variates(self):
        its = self.make_boundary_intersection(size=self.nsamples)
        scene = self.make_scene()
        sampler = PCG32(size=dr.width(its.p), initstate=0)
        grad_n = self.solver.normal_derivative(
            its, scene, sampler, clamping=self.clamping, control_variates=False)
        grad_n = dr.dot(Array2(grad_n), its.n)
        return grad_n

    def ours_half_ball(self):
        its = self.make_boundary_intersection(size=self.nsamples)
        scene = self.make_scene()
        sampler = PCG32(size=dr.width(its.p), initstate=0)
        grad_n = self.solver.normal_derivative(
            its, scene, sampler, clamping=self.clamping, ball_ratio=0.1)
        grad_n = dr.dot(Array2(grad_n), its.n)
        return grad_n

    def baseline2(self):
        its = self.make_boundary_intersection(size=self.nsamples)
        scene = self.make_scene()
        sampler = PCG32(size=dr.width(its.p), initstate=0)
        grad_n = self.solver.grad(its.p + its.n * 5e-3, scene, sampler)
        grad_n = dr.dot(grad_n, its.n)
        return grad_n

    def plot_mean(self):
        ours = self.ours()
        ours_no_anti = self.ours_no_anti()
        ours_no_control_variates = self.ours_no_control_variates()
        ours_half_ball = self.ours_half_ball()
        baseline2 = self.baseline2()
        ground_truth = self.ground_truth()
        stat = Statistics()
        figure = plt.figure(figsize=(8, 5))
        plt.rcParams.update({'font.size': 22})
        import seaborn as sns
        sns.set_style("whitegrid")
        plt.plot(stat.mean(baseline2), label='baseline', alpha=0.8)
        plt.plot(stat.mean(ours_no_anti), label='ours (no anti.)', alpha=0.8)
        plt.plot(stat.mean(ours_half_ball),
                 label='ours (small ball)', alpha=0.8)
        plt.plot(stat.mean(ours), label='ours', alpha=0.8)
        plt.plot(dr.repeat(ground_truth, dr.width(ours)),
                 label='ground truth', alpha=0.8)
        gt = ground_truth.numpy()
        plt.ylim(gt-2., gt+5)
        # set legend size, set bottom right
        plt.legend(prop={'size': 18}, loc='upper right')
        plt.tight_layout()
        plt.savefig(self.out_dir / "mean.pdf", bbox_inches='tight')
        plt.show()
        plt.close(figure)

    def plot_var(self):
        ours = self.ours()
        ours_no_anti = self.ours_no_anti()
        # ours_no_control_variates = self.ours_no_control_variates()
        ours_half_ball = self.ours_half_ball()
        baseline2 = self.baseline2()
        stat = Statistics()
        var_ours = stat.var(ours)[-1]
        var_ours_no_anti = stat.var(ours_no_anti)[-1]
        # var_ours_no_control_variates = stat.var(ours_no_control_variates)[-1]
        var_ours_half_ball = stat.var(ours_half_ball)[-1]
        var_baseline2 = stat.var(baseline2)[-1]

        names = [
            'baseline',
            'ours (no anti.)',
            # 'ours (no CV)',
            'ours (small ball)',
            'ours'
        ]
        variances = [
            var_baseline2,
            var_ours_no_anti,
            #  var_ours_no_control_variates,
            var_ours_half_ball,
            var_ours,
        ]
        figure = plt.figure(figsize=(5, 5))
        plt.rcParams.update({'font.size': 30})
        import seaborn as sns
        sns.set_style("whitegrid")
        plt.bar(names, variances)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
        plt.xticks(rotation=45, ha='right')
        plt.savefig(self.out_dir / "variance.pdf", bbox_inches='tight')


if __name__ == "__main__":
    test = Test(
        nsamples=2000,
        solver=WoSWithSource(nwalks=1000, nsteps=256,
                             epsilon=2e-4, double_sided=False),
    )
    test.plot_mean()
    test.plot_var()
