import drjit as _dr
import numpy as _np
from drjit.cuda.ad import TensorXf as Tensor
from drjit.cuda.ad import Loop
from drjit.cuda.ad import PCG32
from drjit.cuda.ad import Int32 as Int
from drjit.cuda.ad import Quaternion4f as Quaternion4
from drjit.cuda.ad import Matrix2f as Matrix2
from drjit.cuda.ad import Array4i as Array4i
from drjit.cuda.ad import Array3i as Array3i
from drjit.cuda.ad import Array2i as Array2i
from drjit.cuda.ad import Array2f as Array2
from drjit.cuda.ad import Array3f as Array3
from drjit.cuda.ad import Array4f as Array4
from drjit.cuda.ad import Bool
from drjit.cuda.ad import Float32 as Float
from pathlib import Path
import drjit as dr
import mitsuba as mi
import numpy as np
import torch

basedir = Path(__file__).parent.parent

# single precision
mi.set_variant('cuda_ad_rgb')


def outer_product(a, b):
    return Matrix2(a.x * b.x, a.x * b.y,
                   a.y * b.x, a.y * b.y)


EPSILON = 1e-3


def pytype(ctype):
    if ctype == dr.cuda.ad.Float64:
        return np.float64
    if ctype == dr.cuda.ad.Float32:
        return np.float32


def to_torch(arg):
    '''
    https://github.com/mitsuba-renderer/drjit/pull/37/commits/b4bbf4806306717491d1432c0b8900a8a98cc2de#diff-cb98ac5d691c0178d6587b9312299d78d764be121566e00b5ec1d51da70d6bbf
    '''
    import torch
    import torch.autograd

    class ToTorch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, arg, handle):
            ctx.drjit_arg = arg
            return arg.torch()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_output):
            # print("drjit backward")
            _dr.set_grad(ctx.drjit_arg, grad_output)
            _dr.enqueue(_dr.ADMode.Backward, ctx.drjit_arg)
            _dr.traverse(type(ctx.drjit_arg), _dr.ADMode.Backward,
                         dr.ADFlag.ClearInterior)  # REVIEW
            # del ctx.drjit_arg # REVIEW
            return None, None

    handle = torch.empty(0, requires_grad=True)
    return ToTorch.apply(arg, handle)


def from_torch(dtype, arg):
    import torch
    if not _dr.is_diff_v(dtype) or not _dr.is_array_v(dtype):
        raise TypeError(
            "from_torch(): expected a differentiable Dr.Jit array type!")

    class FromTorch(_dr.CustomOp):
        def eval(self, arg, handle):
            self.torch_arg = arg
            return dtype(arg)

        def forward(self):
            raise TypeError("from_torch(): forward-mode AD is not supported!")

        def backward(self):
            # print("torch backward")
            grad = self.grad_out().torch()
            self.torch_arg.backward(grad)

    handle = _dr.zeros(dtype)
    _dr.enable_grad(handle)
    return _dr.custom(FromTorch, arg, handle)


def print_grad(arg):
    class Printer(_dr.CustomOp):
        def eval(self, arg):
            print(arg.numpy())
            return arg

        def forward(self):
            grad = self.grad_in('arg')
            print(grad)
            self.set_grad_out(grad)

        def backward(self):
            grad = self.grad_out()
            print('grad: ', grad.numpy())
            # print(dr.sum(grad))
            self.set_grad_in('arg', grad)

    return _dr.custom(Printer, arg)
