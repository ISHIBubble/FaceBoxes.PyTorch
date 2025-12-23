# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# Try to import compiled NMS, fallback to pure Python implementation
try:
    from .nms.cpu_nms import cpu_nms, cpu_soft_nms
    COMPILED_NMS_AVAILABLE = True
except ImportError:
    from .nms.py_cpu_nms import py_cpu_nms as cpu_nms
    COMPILED_NMS_AVAILABLE = False
    print("Warning: Compiled NMS not available, using pure Python implementation (slower)")

try:
    from .nms.gpu_nms import gpu_nms
    GPU_NMS_AVAILABLE = True
except ImportError:
    GPU_NMS_AVAILABLE = False
    print("Warning: GPU NMS not available, will use CPU implementation")


# def nms(dets, thresh, force_cpu=False):
#     """Dispatch to either CPU or GPU NMS implementations."""
#
#     if dets.shape[0] == 0:
#         return []
#     if cfg.USE_GPU_NMS and not force_cpu:
#         return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
#     else:
#         return cpu_nms(dets, thresh)


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if force_cpu or not GPU_NMS_AVAILABLE:
        return cpu_nms(dets, thresh)
    return gpu_nms(dets, thresh)
