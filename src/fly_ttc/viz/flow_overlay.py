"""Privacy-preserving flow/radial diagnostics from already blurred caches."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .traces import unavailable_figure


def flow_rgb(flow):
    flow = np.nan_to_num(np.asarray(flow, dtype=np.float32))
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    scale = max(float(np.percentile(magnitude, 99)), 1e-6)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / scale * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def plot_diagnostics(cache_path, path, video_id=None, dpi=130, blur_ksize=61):
    if cache_path is None:
        unavailable_figure(path, "Looming diagnostic: blurred image / flow / radial outflow",
                           "Unavailable: no cached successful pre-event looming diagnostic.\nNo raw video image is substituted.", dpi)
        return False
    with np.load(Path(cache_path), allow_pickle=False) as data:
        images, flows, radial, times = data["image"], data["flow"], data["radial"], data["t"]
        requested = data["requested_t"] if "requested_t" in data else times
    count = min(len(images), len(flows), len(radial), len(times), 3)
    if count == 0:
        unavailable_figure(path, "Looming diagnostic", "Unavailable: empty diagnostic cache.", dpi)
        return False
    fig, axes = plt.subplots(count, 3, figsize=(12, 3 * count), squeeze=False)
    vmax = max(float(np.nanpercentile(np.maximum(radial[:count], 0), 99)), 1e-6)
    for i in range(count):
        # Apply whole-frame blur again as defense in depth. Never read raw video.
        ksize = max(3, int(blur_ksize) | 1)
        safe_image = cv2.GaussianBlur(np.asarray(images[i]), (ksize, ksize), 0)
        axes[i, 0].imshow(safe_image)
        axes[i, 1].imshow(flow_rgb(flows[i]))
        heat = axes[i, 2].imshow(np.maximum(radial[i], 0), cmap="inferno", vmin=0, vmax=vmax)
        for j, title in enumerate(("Whole-frame blur", "Optical flow", "Positive radial flow")):
            axes[i, j].set_title(f"{title} | t={float(times[i]):.3f}s", fontsize=10)
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(f"Requested {float(requested[i]):.3f}s")
        fig.colorbar(heat, ax=axes[i, 2], fraction=0.046, pad=0.02)
    fig.suptitle(f"Pre-event diagnostic: {video_id or Path(cache_path).stem}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True
