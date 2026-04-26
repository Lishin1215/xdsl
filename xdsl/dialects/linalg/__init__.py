from xdsl.ir import Dialect

from . import ops as _ops
from .ops import *

Linalg = Dialect(
    "linalg",
    [
        _ops.GenericOp,
        _ops.YieldOp,
        _ops.IndexOp,
        _ops.AddOp,
        _ops.ExpOp,
        _ops.LogOp,
        _ops.SubOp,
        _ops.SqrtOp,
        _ops.SelectOp,
        _ops.FillOp,
        _ops.CopyOp,
        _ops.MaxOp,
        _ops.MinOp,
        _ops.MulOp,
        _ops.TransposeOp,
        _ops.MatmulOp,
        _ops.QuantizedMatmulOp,
        _ops.PoolingNchwMaxOp,
        _ops.Conv2DNchwFchwOp,
        _ops.Conv2DNhwgcGfhwcOp,
        _ops.Conv2DNhwc_HwcfOp,
        _ops.Conv2DNgchwGfchwOp,
        _ops.Conv2DNgchwFgchwOp,
        _ops.Conv2DNhwc_FhwcOp,
        _ops.BroadcastOp,
        _ops.ReduceOp,
    ],
    [
        _ops.IteratorTypeAttr,
    ],
)
