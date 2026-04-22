

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ..core.utils import get_logger

logger = get_logger(__name__)

__all__ = ["export_onnx"]


def export_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    path: Union[str, Path],
    opset_version: int = 17,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    dynamic_axes: Optional[dict] = None,
    simplify: bool = False,
) -> Path:
    """Export a PyTorch model to ONNX.

    Parameters
    ----------
    model:
        The model in eval mode.
    dummy_input:
        A representative input tensor (or tuple of tensors).
    path:
        Destination file path.
    opset_version:
        ONNX opset (default 17).
    input_names / output_names:
        Human-readable I/O names.
    dynamic_axes:
        Mapping of input/output names to dynamic dimension indices
        (e.g. ``{"input": {0: "batch"}}``).
    simplify:
        If ``True`` and ``onnx-simplifier`` is available, run a
        simplification pass.

    Returns
    -------
    Path
        The path to the written ONNX file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    logger.info("Exported ONNX model to %s (opset %d).", path, opset_version)

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            onnx_model = onnx.load(str(path))
            simplified, ok = onnx_simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(path))
                logger.info("ONNX model simplified successfully.")
            else:
                logger.warning("ONNX simplification returned ok=False; kept original.")
        except ImportError:
            logger.warning("Install 'onnx' and 'onnxsim' for model simplification.")

    return path
