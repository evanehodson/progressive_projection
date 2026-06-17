import math
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class CameraState:
    cx: float = 0.0
    cy: float = 0.0
    azimuth: float = 0.0
    tilt: float = 45.0
    height_factor: float = 0.5
    fov: float = 30.0
    profile: np.ndarray = None

    def asdict(self):
        d = {k: v for k, v in asdict(self).items()}
        if isinstance(d.get("profile"), np.ndarray):
            d["profile"] = d["profile"].tolist()
        return d


class BerannCamera:
    def __init__(self, dem_x_min, dem_x_max, dem_y_min, dem_y_max):
        self.x_min = dem_x_min
        self.x_max = dem_x_max
        self.y_min = dem_y_min
        self.y_max = dem_y_max
        self.diagonal = math.sqrt((dem_x_max - dem_x_min)**2 + (dem_y_max - dem_y_min)**2)

        self.state = CameraState()
        self._z_base = 0.0

    def update(self, state: CameraState, z_base=0.0):
        self.state = state
        self._z_base = z_base

        self.az_rad = math.radians(state.azimuth)
        self.sin_a = math.sin(self.az_rad)
        self.cos_a = math.cos(self.az_rad)

        corner_d = [
            (self.x_min - state.cx) * self.sin_a + (self.y_min - state.cy) * self.cos_a,
            (self.x_min - state.cx) * self.sin_a + (self.y_max - state.cy) * self.cos_a,
            (self.x_max - state.cx) * self.sin_a + (self.y_min - state.cy) * self.cos_a,
            (self.x_max - state.cx) * self.sin_a + (self.y_max - state.cy) * self.cos_a,
        ]
        self.v_min = min(corner_d)
        self.v_max = max(corner_d)
        self.v_range = self.v_max - self.v_min
        if self.v_range <= 0:
            self.v_range = 1.0

    def _get_cam_height(self):
        return self._z_base + self.diagonal * self.state.height_factor

    def get_cam_position(self):
        return (self.state.cx, self.state.cy, self._get_cam_height())

    def compute_rays(self, out_w, out_h):
        cam_height = self._get_cam_height()
        cx = self.state.cx
        cy = self.state.cy

        el_rad = math.radians(self.state.tilt)
        az_rad = self.az_rad

        if el_rad > 0.01:
            cot_el = math.cos(el_rad) / math.sin(el_rad)
            look_dist_h = (cam_height - self._z_base) * cot_el
            focal_x = cx + look_dist_h * math.sin(az_rad)
            focal_y = cy + look_dist_h * math.cos(az_rad)
        else:
            focal_x = cx + 10000 * math.sin(az_rad)
            focal_y = cy + 10000 * math.cos(az_rad)

        cam_pos = np.array([cx, cy, cam_height], dtype=np.float64)
        focal = np.array([focal_x, focal_y, 0.0], dtype=np.float64)

        fwd = focal - cam_pos
        fwd_len = np.linalg.norm(fwd)
        if fwd_len < 1e-10:
            fwd = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            fwd = fwd / fwd_len

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(fwd, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-10:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right = right / right_len

        up = np.cross(right, fwd)
        up_len = np.linalg.norm(up)
        if up_len > 1e-10:
            up = up / up_len

        tan_half_v = math.tan(math.radians(self.state.fov) / 2.0)
        tan_half_h = tan_half_v * out_w / out_h

        # Vectorized ray construction
        col = np.arange(out_w, dtype=np.float32)
        row = np.arange(out_h, dtype=np.float32)
        sx, sy = np.meshgrid(
            (col + 0.5) / out_w * 2.0 - 1.0,
            1.0 - (row + 0.5) / out_h * 2.0,
        )

        cs_x = sx * tan_half_h
        cs_y = sy * tan_half_v
        cs_z = np.full_like(cs_x, -1.0)
        cs_len = np.sqrt(cs_x * cs_x + cs_y * cs_y + cs_z * cs_z)
        cs_x /= cs_len
        cs_y /= cs_len
        cs_z /= cs_len

        rays = np.empty((out_h, out_w, 3), dtype=np.float32)
        rays[:, :, 0] = cs_x * right[0] + cs_y * up[0] - cs_z * fwd[0]
        rays[:, :, 1] = cs_x * right[1] + cs_y * up[1] - cs_z * fwd[1]
        rays[:, :, 2] = cs_x * right[2] + cs_y * up[2] - cs_z * fwd[2]

        origins = np.empty((out_h, out_w, 3), dtype=np.float32)
        origins[:, :, 0] = cam_pos[0]
        origins[:, :, 1] = cam_pos[1]
        origins[:, :, 2] = cam_pos[2]

        return rays, origins, self.v_min, self.v_range, self.sin_a, self.cos_a
