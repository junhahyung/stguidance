from contextlib import contextmanager

import torch

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func as flash_varlen_qkvpacked_attn
except ImportError:
    flash_varlen_qkvpacked_attn = None

try:
    from sageattention import sageattn as sage_attn
except ImportError:
    sage_attn = None

try:
    from comfy.ldm.modules.attention import comfy_optimized_attention as comfy_attn
except ImportError:
    comfy_attn = None


from torch.nn.attention import SDPBackend, sdpa_kernel

backends = []
if torch.cuda.get_device_properties(0).major < 7:
    backends.append(SDPBackend.MATH)
if torch.cuda.get_device_properties(0).major >= 9.0:
    backends.append(SDPBackend.CUDNN_ATTENTION)
else:
    backends.append(SDPBackend.EFFICIENT_ATTENTION)

@contextmanager
def sdpa_attn_ctx():
    with sdpa_kernel(backends):
        yield
