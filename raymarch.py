import numpy as np
from numba import njit, prange


@njit(inline='always')
def sample_bilinear_float(arr, dr, dc, H, W, nodata_val):
    r0 = int(dr)
    c0 = int(dc)
    if r0 < 0 or r0 >= H - 1 or c0 < 0 or c0 >= W - 1:
        return nodata_val
    rf = dr - r0
    cf = dc - c0
    v00 = arr[r0, c0]
    v01 = arr[r0, c0 + 1]
    v10 = arr[r0 + 1, c0]
    v11 = arr[r0 + 1, c0 + 1]
    v0 = v00 + (v01 - v00) * cf
    v1 = v10 + (v11 - v10) * cf
    return v0 + (v1 - v0) * rf


@njit(inline='always')
def sample_bilinear_int(arr, dr, dc, H, W, nodata_val):
    r0 = int(dr)
    c0 = int(dc)
    if r0 < 0 or r0 >= H - 1 or c0 < 0 or c0 >= W - 1:
        return nodata_val
    rf = dr - r0
    cf = dc - c0
    v00 = arr[r0, c0]
    v01 = arr[r0, c0 + 1]
    v10 = arr[r0 + 1, c0]
    v11 = arr[r0 + 1, c0 + 1]
    v0 = v00 + (v01 - v00) * cf
    v1 = v10 + (v11 - v10) * cf
    return int(v0 + (v1 - v0) * rf)


@njit(inline='always')
def sample_dem(dem, dr, dc, H, W, nodata_val):
    return sample_bilinear_float(dem, dr, dc, H, W, nodata_val)


@njit(parallel=True)
def raymarch_kernel(
    dem, landcover, soil_codes, shadow,  # soil_codes is uint16
    soil_lut,
    profile,
    rays, origins,
    sin_a, cos_a, max_forward_dist, z_base,
    out_w, out_h,
    dem_x0, dem_y0, dem_res,
    dem_H, dem_W,
    max_steps_coarse, max_steps_binary, max_dist_m,
    dem_nodata,
    out_depth, out_normal_x, out_normal_y, out_normal_z,
    out_shadow, out_landcover, out_soil_r, out_soil_g, out_soil_b,
):
    profile_len = len(profile)
    one_over_255 = 1.0 / 255.0

    for row in prange(out_h):
        for col in range(out_w):
            ox = origins[row, col, 0]
            oy = origins[row, col, 1]
            oz = origins[row, col, 2]
            rx = rays[row, col, 0]
            ry = rays[row, col, 1]
            rz = rays[row, col, 2]

            # Compute per-pixel step size based on distance to the z_base plane
            # We want to reach z_base - 2000m (buffer for terrain relief)
            if rz < -1e-10:
                target_z = z_base - 2000.0
                t_max = (target_z - oz) / rz
                if t_max > max_dist_m:
                    t_max = max_dist_m
                if t_max < 100.0:
                    t_max = 100.0
            else:
                t_max = max_dist_m

            coarse_step = t_max / max_steps_coarse

            horiz_sq = rx * rx + ry * ry
            if horiz_sq < 1e-20:
                out_depth[row, col] = -1.0
                continue
            horiz = horiz_sq ** 0.5
            inv_horiz = 1.0 / horiz

            step_x = coarse_step * rx * inv_horiz
            step_y = coarse_step * ry * inv_horiz
            step_z = coarse_step * rz * inv_horiz

            hit = False
            hit_t = 0.0
            hit_dr = 0.0
            hit_dc = 0.0

            px = ox
            py = oy
            pz = oz

            # Coarse march
            for i in range(max_steps_coarse):
                px += step_x
                py += step_y
                pz += step_z
                t = (i + 1) * coarse_step
                if t > max_dist_m:
                    break

                dc = (px - dem_x0) / dem_res
                dr = (dem_y0 - py) / dem_res

                dem_h = sample_dem(dem, dr, dc, dem_H, dem_W, dem_nodata)
                if abs(dem_h) > 1e10:
                    continue

                # Berann-modified terrain height (multiplicative exaggeration)
                # Scale follows horizontal forward distance along the DEM (azimuth direction)
                forward_dist = (px - ox) * sin_a + (py - oy) * cos_a
                f_norm = forward_dist / max_forward_dist
                if f_norm < 0.0:
                    f_norm = 0.0
                if f_norm > 1.0:
                    f_norm = 1.0
                dep_idx = int(f_norm * 255.0)
                if dep_idx >= profile_len:
                    dep_idx = profile_len - 1

                terrain_z = dem_h * profile[dep_idx]

                if pz < terrain_z:
                    # Back up to just before crossing
                    px -= step_x
                    py -= step_y
                    pz -= step_z
                    hit = True
                    hit_t = t - coarse_step
                    hit_dr = dr - step_z / dem_res  # rough
                    hit_dc = dc
                    break

            if hit:
                # Binary refinement
                low_t = hit_t
                high_t = hit_t + coarse_step
                for bi in range(max_steps_binary):
                    mid_t = (low_t + high_t) * 0.5
                    ratio = mid_t / coarse_step
                    mpx = ox + step_x * ratio
                    mpy = oy + step_y * ratio
                    mpz = oz + step_z * ratio
                    mdc = (mpx - dem_x0) / dem_res
                    mdr = (dem_y0 - mpy) / dem_res
                    mh = sample_dem(dem, mdr, mdc, dem_H, dem_W, dem_nodata)

                    mforward_dist = (mpx - ox) * sin_a + (mpy - oy) * cos_a
                    mf_norm = mforward_dist / max_forward_dist
                    if mf_norm < 0.0:
                        mf_norm = 0.0
                    if mf_norm > 1.0:
                        mf_norm = 1.0
                    mdep_idx = int(mf_norm * 255.0)
                    if mdep_idx >= profile_len:
                        mdep_idx = profile_len - 1
                    mterrain_z = mh * profile[mdep_idx]

                    if mpz < mterrain_z:
                        high_t = mid_t
                    else:
                        low_t = mid_t

                hit_t = (low_t + high_t) * 0.5

            if hit:
                ratio = hit_t / coarse_step
                hx = ox + step_x * ratio
                hy = oy + step_y * ratio
                hz = oz + step_z * ratio
                hdc = (hx - dem_x0) / dem_res
                hdr = (dem_y0 - hy) / dem_res

                out_depth[row, col] = hit_t

                # Exaggeration factor at the exact hit position
                hforward_dist = (hx - ox) * sin_a + (hy - oy) * cos_a
                hf_norm = hforward_dist / max_forward_dist
                if hf_norm < 0.0:
                    hf_norm = 0.0
                if hf_norm > 1.0:
                    hf_norm = 1.0
                hdep_idx = int(hf_norm * 255.0)
                if hdep_idx >= profile_len:
                    hdep_idx = profile_len - 1
                exag_hit = profile[hdep_idx]

                h_dr0 = int(hdr)
                h_dc0 = int(hdc)
                if 0 <= h_dr0 < dem_H - 1 and 0 <= h_dc0 < dem_W - 1:
                    eps = 1.0
                    z00 = sample_dem(dem, hdr - eps, hdc - eps, dem_H, dem_W, dem_nodata)
                    z01 = sample_dem(dem, hdr - eps, hdc + eps, dem_H, dem_W, dem_nodata)
                    z10 = sample_dem(dem, hdr + eps, hdc - eps, dem_H, dem_W, dem_nodata)
                    z11 = sample_dem(dem, hdr + eps, hdc + eps, dem_H, dem_W, dem_nodata)

                    if (abs(z00) < 1e10 and abs(z01) < 1e10 and
                            abs(z10) < 1e10 and abs(z11) < 1e10):
                        wx = 2.0 * eps * dem_res
                        wy = 2.0 * eps * dem_res
                        dz_dx = ((z01 - z00) + (z11 - z10)) / wx
                        dz_dy = ((z10 - z00) + (z11 - z01)) / wy

                        nx = -dz_dx * exag_hit
                        ny = -dz_dy * exag_hit
                        nz = 1.0
                        nlen = (nx * nx + ny * ny + nz * nz) ** 0.5
                        if nlen > 1e-10:
                            nx /= nlen
                            ny /= nlen
                            nz /= nlen
                        out_normal_x[row, col] = nx
                        out_normal_y[row, col] = ny
                        out_normal_z[row, col] = nz
                    else:
                        out_normal_x[row, col] = 0.0
                        out_normal_y[row, col] = 0.0
                        out_normal_z[row, col] = 1.0

                    lc = sample_bilinear_int(landcover, hdr, hdc, dem_H, dem_W, 0)
                    out_landcover[row, col] = lc

                    sc = int(sample_bilinear_float(soil_codes, hdr, hdc, dem_H, dem_W, 0))
                    if sc < 0:
                        sc = 0
                    if sc >= len(soil_lut):
                        sc = len(soil_lut) - 1
                    out_soil_r[row, col] = soil_lut[sc, 0]
                    out_soil_g[row, col] = soil_lut[sc, 1]
                    out_soil_b[row, col] = soil_lut[sc, 2]

                    sh = sample_bilinear_float(shadow, hdr, hdc, dem_H, dem_W, 0.0)
                    out_shadow[row, col] = sh
                else:
                    out_normal_x[row, col] = 0.0
                    out_normal_y[row, col] = 1.0
                    out_normal_z[row, col] = 0.0
                    out_landcover[row, col] = 0
                    out_soil_r[row, col] = 117
                    out_soil_g[row, col] = 104
                    out_soil_b[row, col] = 94
                    out_shadow[row, col] = 0.0
            else:
                out_depth[row, col] = -1.0
                out_normal_x[row, col] = 0.0
                out_normal_y[row, col] = 0.0
                out_normal_z[row, col] = 0.0
                out_landcover[row, col] = 0
                out_soil_r[row, col] = 0
                out_soil_g[row, col] = 0
                out_soil_b[row, col] = 0
                out_shadow[row, col] = 0.0


def raymarch(stack, camera, out_w, out_h, quality_tier="settled"):
    if quality_tier == "interactive":
        max_steps_coarse = 20
        max_steps_binary = 8
        max_dist_m = 1000000.0
    elif quality_tier == "settled":
        max_steps_coarse = 32
        max_steps_binary = 10
        max_dist_m = 1000000.0
    else:
        max_steps_coarse = 64
        max_steps_binary = 12
        max_dist_m = 1000000.0

    (dem_filled, landcover, soil_codes, shadow,
     soil_lut, valid_mask, res,
     dem_x0, dem_y0, dem_H, dem_W) = stack.get_full_res_arrays()

    profile = camera.state.profile
    if profile is None or len(profile) < 2:
        profile = np.linspace(0.9, 1.1, 256, dtype=np.float32)
    else:
        # Widget emits values in [0, 5], used directly as multiplier
        profile = profile.astype(np.float32)
        profile = np.clip(profile, 0.0, 5.0)

    rays, origins = camera.compute_rays(out_w, out_h)
    sin_a = camera.sin_a
    cos_a = camera.cos_a
    max_forward_dist = camera.max_forward_dist
    z_base = camera._z_base

    out_depth = np.zeros((out_h, out_w), dtype=np.float32)
    out_normal_x = np.zeros((out_h, out_w), dtype=np.float32)
    out_normal_y = np.zeros((out_h, out_w), dtype=np.float32)
    out_normal_z = np.zeros((out_h, out_w), dtype=np.float32)
    out_shadow = np.zeros((out_h, out_w), dtype=np.float32)
    out_landcover = np.zeros((out_h, out_w), dtype=np.int16)
    out_soil_r = np.zeros((out_h, out_w), dtype=np.uint8)
    out_soil_g = np.zeros((out_h, out_w), dtype=np.uint8)
    out_soil_b = np.zeros((out_h, out_w), dtype=np.uint8)

    dem_nodata_val = -999999.0

    raymarch_kernel(
        dem_filled, landcover, soil_codes, shadow,
        soil_lut,
        profile.astype(np.float32),
        rays, origins,
        sin_a, cos_a, max_forward_dist, z_base,
        out_w, out_h,
        dem_x0, dem_y0, res,
        dem_H, dem_W,
        max_steps_coarse, max_steps_binary, max_dist_m,
        dem_nodata_val,
        out_depth, out_normal_x, out_normal_y, out_normal_z,
        out_shadow, out_landcover, out_soil_r, out_soil_g, out_soil_b,
    )

    gbuffer = {
        "depth": out_depth,
        "normal": np.stack([out_normal_x, out_normal_y, out_normal_z], axis=-1),
        "shadow_ao": out_shadow,
        "landcover": out_landcover,
        "soil_color": np.stack([out_soil_r, out_soil_g, out_soil_b], axis=-1).astype(np.uint8),
    }
    return gbuffer
