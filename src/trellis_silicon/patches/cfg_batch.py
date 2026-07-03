"""CFG batching patch: fold the two classifier-free-guidance forwards (positive
and negative cond) into one batch-2 forward to better use the MPS GPU."""

import os

from .common import TRELLIS_ROOT, read_file, write_file


def patch_cfg_batching():
    """Batch the two classifier-free-guidance forwards into ONE batch-2 forward.

    Every guided sampling step (guidance_strength not in {0, 1}) runs the model
    twice: once with the positive cond and once with the negative cond. At batch
    1 the MPS GPU is underutilized, so folding cond+neg_cond into a single
    batch-2 forward costs far less than 2x a single forward. The result is
    mathematically identical (only FP summation order changes).

    Works for both the dense structure flow model (torch.cat the [1,...] inputs)
    and the sparse SLat flow model (build a 2-batch SparseTensor via
    from_tensor_list, split the output by layout). Any structural surprise falls
    back to the original sequential two-pass path, so behavior is never worse.

    Opt out with CFG_BATCH=0 in the environment. Set CFG_BATCH_DEBUG=1 to print
    when the batched path bails to the sequential fallback.
    """
    path = os.path.join(
        TRELLIS_ROOT, "trellis2/pipelines/samplers/classifier_free_guidance_mixin.py"
    )
    src = read_file(path)

    if "_cfg_dual_forward" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    new_src = '''from typing import *
import os
import torch


def _cfg_batch_enabled():
    """CFG batching is ON by default; set CFG_BATCH=0 to force the sequential path."""
    return os.environ.get('CFG_BATCH', '1') != '0'


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _cfg_dual_forward(self, model, x_t, t, cond, neg_cond, **kwargs):
        """Return (pred_pos, pred_neg) for a guided step.

        When CFG batching is enabled, cond/neg_cond are folded into a single
        batch-2 forward (mathematically identical to two batch-1 forwards; only
        FP summation order may differ). Falls back to two sequential forwards on
        any structural mismatch so behavior is never worse than the original.
        """
        if _cfg_batch_enabled() and kwargs.get('concat_cond', None) is None:
            try:
                return self._cfg_batched_forward(model, x_t, t, cond, neg_cond, **kwargs)
            except Exception as e:  # noqa: BLE001 - any surprise -> safe sequential path
                if os.environ.get('CFG_BATCH_DEBUG'):
                    print(f"[CFG_BATCH] batched path failed ({e!r}); using sequential fallback")
        pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
        pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return pred_pos, pred_neg

    def _cfg_batched_forward(self, model, x_t, t, cond, neg_cond, **kwargs):
        """One batch-2 forward, split back into (pred_pos, pred_neg).

        pred_pos/pred_neg are returned in the SAME single-batch form the
        sequential path produces, so the downstream combine + CFG-rescale code
        is byte-for-byte unchanged.
        """
        if not (isinstance(cond, torch.Tensor) and isinstance(neg_cond, torch.Tensor)):
            raise TypeError("cond/neg_cond are not dense tensors")
        if cond.shape[0] != 1 or neg_cond.shape[0] != 1:
            raise ValueError("expected batch-1 cond/neg_cond")
        cond2 = torch.cat([cond, neg_cond], dim=0)

        # SparseTensor is only needed for the sparse SLat models; import lazily
        # so the dense structure phase never pays for it.
        try:
            from ...modules.sparse import SparseTensor
        except Exception:  # noqa: BLE001
            SparseTensor = ()

        if isinstance(x_t, torch.Tensor):
            # Dense structure flow model: [1, C, R, R, R] -> [2, ...] -> split.
            x2 = torch.cat([x_t, x_t], dim=0)
            out = super()._inference_model(model, x2, t, cond2, **kwargs)
            return out[0:1], out[1:2]
        elif SparseTensor and isinstance(x_t, SparseTensor):
            # Sparse SLat flow model: duplicate the single batch element (same
            # coords, same feats) into a 2-batch SparseTensor. from_tensor_list
            # rewrites the batch column to 0 / 1. Equal seqlens => the SDPA
            # attention path pads to L with zero waste.
            x2 = SparseTensor.from_tensor_list(
                [x_t.feats, x_t.feats], [x_t.coords, x_t.coords]
            )
            out = super()._inference_model(model, x2, t, cond2, **kwargs)
            # x_t.replace keeps x_t's own coords and shape[0]=1, so each half is
            # structurally identical to a plain batch-1 model output.
            pred_pos = x_t.replace(out.feats[out.layout[0]])
            pred_neg = x_t.replace(out.feats[out.layout[1]])
            return pred_pos, pred_neg
        else:
            raise TypeError(f"unsupported x_t type {type(x_t)}")

    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale=0.0, **kwargs):
        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        elif guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        else:
            pred_pos, pred_neg = self._cfg_dual_forward(model, x_t, t, cond, neg_cond, **kwargs)
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg

            # CFG rescale
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)

            return pred
'''
    write_file(path, new_src)
