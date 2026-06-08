import numpy as np
import rasterio

DEM_TIF_PATH = "../data/wurl/USGS_13_n41w112_20260519.tif"
OUTPUT_PLY_PATH = "flat_terrain_base.ply"

DOWNSAMPLE_STEP = 2  # Keep dense for clean mesh topology and accurate Raycasts
TARGET_LAT = 40.5754067
TARGET_LON = -111.7915164

def write_flat_ply(path, vertices_xyz, faces_0idx):
    """Writes a clean binary little-endian PLY file of the unwarped terrain."""
    n_verts = len(vertices_xyz)
    n_faces = len(faces_0idx)

    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"element face {n_faces}\n"
        f"property list uchar uint vertex_indices\n"
        f"end_header\n"
    ).encode("ascii")

    # Pack structure for coordinates
    vert_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    vert_struct = np.empty(n_verts, dtype=vert_dtype)
    vert_struct["x"] = vertices_xyz[:, 0]
    vert_struct["y"] = vertices_xyz[:, 1]
    vert_struct["z"] = vertices_xyz[:, 2]

    # Pack structure for quad faces
    face_dtype = np.dtype([("count", "u1"), ("v0", "<u4"), ("v1", "<u4"), ("v2", "<u4"), ("v3", "<u4")])
    face_struct = np.empty(n_faces, dtype=face_dtype)
    face_struct["count"] = 4
    face_struct["v0"] = faces_0idx[:, 0]
    face_struct["v1"] = faces_0idx[:, 1]
    face_struct["v2"] = faces_0idx[:, 2]
    face_struct["v3"] = faces_0idx[:, 3]

    with open(path, "wb") as f:
        f.write(header)
        f.write(vert_struct.tobytes())
        f.write(face_struct.tobytes())
    print(f"\nSuccessfully generated base mesh asset: {path}")

def main():
    print(f"Reading source GeoTIFF dataset: {DEM_TIF_PATH}...")
    
    with rasterio.open(DEM_TIF_PATH) as src:
        out_h = int(src.height // DOWNSAMPLE_STEP)
        out_w = int(src.width // DOWNSAMPLE_STEP)
        
        elevation = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear)
        
        cols, rows = np.meshgrid(
            np.arange(0, out_w * DOWNSAMPLE_STEP, DOWNSAMPLE_STEP),
            np.arange(0, out_h * DOWNSAMPLE_STEP, DOWNSAMPLE_STEP)
        )
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons = np.array(xs).reshape(out_h, out_w)
        lats = np.array(ys).reshape(out_h, out_w)

    print(f"Grid size pulled: {out_w}x{out_h} ({out_w * out_h} points)")

    # Align geographic coordinate matrices to linear meters
    lat_to_meters = 111320.0
    lon_to_meters = 111320.0 * np.cos(np.radians(TARGET_LAT))

    x_metric = (lons - TARGET_LON) * lon_to_meters
    y_metric = (lats - TARGET_LAT) * lat_to_meters
    z_metric = elevation * 2.5  # Native scale factor

    # Scale geometry down to fit predictably inside standard scene layouts
    SCALE_FACTOR = 0.001 
    X_local = x_metric * SCALE_FACTOR
    Y_local = y_metric * SCALE_FACTOR
    Z_local = z_metric * SCALE_FACTOR

    # Compile the 1D structured arrays
    flat_raw_pts = np.column_stack((X_local.ravel(), Y_local.ravel(), Z_local.ravel()))

    # Build standard 0-indexed quad polygon faces
    ri = np.arange(out_h - 1)[:, None]
    ci = np.arange(out_w - 1)
    tl = ri * out_w + ci
    tr = tl + 1
    br = tl + out_w + 1
    bl = tl + out_w
    faces_quads = np.column_stack((tl.ravel(), bl.ravel(), br.ravel(), tr.ravel()))

    # Save output unwarped asset file
    write_flat_ply(OUTPUT_PLY_PATH, flat_raw_pts, faces_quads)

if __name__ == "__main__":
    main()