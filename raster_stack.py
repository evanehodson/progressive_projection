import csv
import numpy as np
import rasterio

SOIL_CSV_PATH = "../data/wurl/soil_color_lookup_extended_na.csv"

# EVC display colors for viewport (simple greens/browns/blue for key classes)
EVC_DISPLAY_LUT = np.zeros((400, 3), dtype=np.uint8)
EVC_DISPLAY_LUT[11] = (70, 107, 159)    # Open Water
EVC_DISPLAY_LUT[31] = (179, 172, 159)   # Barren
EVC_DISPLAY_LUT[101:110] = (34, 120, 40)  # Tree cover
EVC_DISPLAY_LUT[111:120] = (120, 140, 60) # Shrub
EVC_DISPLAY_LUT[121:130] = (160, 170, 80) # Herbaceous
for i in range(1, 400):
    if tuple(EVC_DISPLAY_LUT[i]) == (0, 0, 0):
        EVC_DISPLAY_LUT[i] = (140, 130, 110)  # default brown


def _build_soil_lut(csv_path, target_slice=125, max_code=600):
    lut = np.zeros((max_code + 1, 3), dtype=np.uint8)
    # Default gray-brown
    lut[:, 0] = 117
    lut[:, 1] = 104
    lut[:, 2] = 94
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for parts in reader:
                if not parts or len(parts) < 7:
                    continue
                try:
                    row_slice = int(float(parts[3]))
                    if row_slice == target_slice:
                        code = int(float(parts[1]))
                        if 0 <= code <= max_code:
                            lut[code, 0] = int(float(parts[4]))
                            lut[code, 1] = int(float(parts[5]))
                            lut[code, 2] = int(float(parts[6]))
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass
    return lut


class RasterStack:
    def __init__(self, dem_path, shadow_path, lc_path, soil_path):
        print("Loading raster stack...", flush=True)

        with rasterio.open(dem_path) as src:
            self.dem = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
        print(f"  DEM: {self.dem.shape}, {self.dem.dtype}", flush=True)

        with rasterio.open(shadow_path) as src:
            self.shadow = src.read(1).astype(np.float32)
        print(f"  Shadow/AO: {self.shadow.shape}, {self.shadow.dtype}", flush=True)

        with rasterio.open(lc_path) as src:
            self.landcover = src.read(1).astype(np.int16)
        print(f"  Landcover: {self.landcover.shape}, {self.landcover.dtype}", flush=True)

        with rasterio.open(soil_path) as src:
            self.soil_codes = src.read(1)
        print(f"  Soil codes: {self.soil_codes.shape}, {self.soil_codes.dtype}", flush=True)

        self.H, self.W = self.dem.shape
        self.res = abs(self.transform[0])

        # DEM bounds
        self.x_min = self.transform[2]
        self.x_max = self.x_min + self.W * self.res
        self.y_max = self.transform[5]
        self.y_min = self.y_max - self.H * self.res

        # Fill nodata in DEM
        dem_valid = self.dem != self.nodata if self.nodata is not None else np.ones_like(self.dem, dtype=bool)
        self._dem_mean = np.nanmean(self.dem[dem_valid])
        self.dem_filled = self.dem.copy()
        self.dem_filled[~dem_valid] = self._dem_mean
        self._dem_valid = dem_valid

        # Fill shadow nodata
        sv = self.shadow != src.nodata if hasattr(src, 'nodata') and src.nodata is not None else np.isfinite(self.shadow)
        self.shadow[~sv] = 0.0

        # Soil LUT
        self.soil_lut = _build_soil_lut(SOIL_CSV_PATH)
        print(f"  Soil LUT: {self.soil_lut.shape}, {len(np.unique(self.soil_codes[self.soil_codes != 65535]))} codes mapped", flush=True)

        print("  Stack loaded.", flush=True)

    def get_full_res_arrays(self):
        return (self.dem_filled, self.landcover, self.soil_codes, self.shadow,
                self.soil_lut, self._dem_valid, self.res,
                self.x_min, self.y_max, self.H, self.W)

    def get_elevation_at(self, world_x, world_y):
        dc = (world_x - self.x_min) / self.res
        dr = (self.y_max - world_y) / self.res
        c0 = int(dc)
        r0 = int(dr)
        if r0 < 0 or r0 >= self.H - 1 or c0 < 0 or c0 >= self.W - 1:
            return self._dem_mean
        rf = dr - r0
        cf = dc - c0
        return (self.dem_filled[r0, c0] * (1 - rf) * (1 - cf) +
                self.dem_filled[r0, c0 + 1] * (1 - rf) * cf +
                self.dem_filled[r0 + 1, c0] * rf * (1 - cf) +
                self.dem_filled[r0 + 1, c0 + 1] * rf * cf)
