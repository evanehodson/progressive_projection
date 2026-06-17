import numpy as np
from PyQt5 import QtGui

LIGHT_DIR = np.array([0.5, -0.5, 0.7], dtype=np.float32)


def compose_viewport(gbuffer, light_dir=None):
    if light_dir is None:
        light_dir = LIGHT_DIR
    light_dir = np.array(light_dir, dtype=np.float32)
    light_len = np.linalg.norm(light_dir)
    if light_len > 0:
        light_dir = light_dir / light_len

    normal = gbuffer["normal"]
    shadow = gbuffer["shadow_ao"]
    soil_color = gbuffer["soil_color"]
    landcover = gbuffer["landcover"]
    depth = gbuffer["depth"]

    h, w = depth.shape

    hillshade = np.clip(np.sum(normal * light_dir, axis=-1), 0.0, 1.0)

    hillshade[depth < 0] = 0.0

    soil_f = soil_color.astype(np.float32) / 255.0
    base = soil_f * hillshade[..., np.newaxis] * shadow[..., np.newaxis]

    from raster_stack import EVC_DISPLAY_LUT
    lc_clamped = np.clip(landcover, 0, len(EVC_DISPLAY_LUT) - 1)
    tint = EVC_DISPLAY_LUT[lc_clamped].astype(np.float32) / 255.0

    color = base * 0.6 + tint * 0.4

    color = np.clip(color * 255.0, 0, 255).astype(np.uint8)

    mask_3d = (depth < 0)[..., np.newaxis]
    color = np.where(mask_3d, np.array([30, 35, 45], dtype=np.uint8), color)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = color
    rgba[:, :, 3] = 255

    return rgba


def gbuffer_to_qimage(rgba):
    h, w = rgba.shape[:2]
    img = QtGui.QImage(rgba.data, w, h, QtGui.QImage.Format_RGBA8888)
    return img.copy()
