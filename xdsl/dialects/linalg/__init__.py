import warnings

from xdsl.ir import Dialect

from . import ops


def __getattr__(name: str):
    # Check if the requested attribute exists in the new ops module
    if hasattr(ops, name):
        warnings.warn(
            f"Importing '{name}' directly from 'xdsl.dialects.linalg' is deprecated. "
            f"Please use 'from xdsl.dialects.linalg.ops import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(ops, name)

    # If it's not in ops, raise the standard AttributeError
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


Linalg = Dialect(
    "linalg",
    [
        ops.GenericOp,
        ops.YieldOp,
        ops.IndexOp,
        ops.AddOp,
        ops.ExpOp,
        ops.LogOp,
        ops.SubOp,
        ops.SqrtOp,
        ops.SelectOp,
        ops.FillOp,
        ops.CopyOp,
        ops.MaxOp,
        ops.MinOp,
        ops.MulOp,
        ops.TransposeOp,
        ops.MatmulOp,
        ops.QuantizedMatmulOp,
        ops.PoolingNchwMaxOp,
        ops.Conv2DNchwFchwOp,
        ops.Conv2DNhwgcGfhwcOp,
        ops.Conv2DNhwc_HwcfOp,
        ops.Conv2DNgchwGfchwOp,
        ops.Conv2DNgchwFgchwOp,
        ops.Conv2DNhwc_FhwcOp,
        ops.BroadcastOp,
        ops.ReduceOp,
    ],
    [
        ops.IteratorTypeAttr,
    ],
)
