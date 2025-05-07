import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np


def setup_device(device):
    if isinstance(device, list) and len(device) == 1:
        device = torch.device(f'cuda:{device[0]}')
    else:
        device = None

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA:", torch.cuda.get_device_name(0))
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # For Apple Silicon
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("Using CPU")

    return device

def get_autocast_context(amp_dtype: str):
    if amp_dtype == "float16":
        return autocast(dtype=torch.float16)
    elif amp_dtype == "bfloat16":
        return autocast(dtype=torch.bfloat16)
    elif amp_dtype == "mixed":
        return autocast()
    else:
        # 返回一个 dummy context manager，不使用 AMP
        from contextlib import nullcontext
        return nullcontext()


def compute_logdiff(arr):
    """
    对 arr（shape=(T, D) 或 (T,)）做 ln((x_t - x_{t-1}) / x_{t-1})，
    并补齐第一个点为 0，防止 NaN/Inf。
    返回 shape 与输入相同。
    """
    eps = 1e-8
    a = np.array(arr, dtype=float)
    # 差分并防止除零
    diff = (a[1:] - a[:-1]) / (a[:-1] + eps)
    logdiff = np.log(diff + eps)
    # 补齐第一个时间点
    if a.ndim == 1:
        pad = np.array([0.0])
    else:
        pad = np.zeros((1, a.shape[1]), dtype=float)
    out = np.concatenate([pad, logdiff], axis=0)

    # 这里为了简单，我们统计替换后等于 0 的值总数，以及替换前的 NaN：
    nan_count = np.isnan(np.concatenate([pad, logdiff], axis=0)).sum()
    zero_count = (out == 0.0).sum()

    print(f"[compute_logdiff] NaN 个数 (替换前): {nan_count}, 0 值个数 (替换后): {zero_count}")

    # 将任何 NaN 或 Inf 全部替换为 0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)