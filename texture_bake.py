import numpy as np
import rasterio
from plyfile import PlyData, PlyElement

# --- USER CONFIGURATION ---
INPUT_PLY_PATH = "../data/wurl/terrain_shading_base.ply"
TEXTURE_SHADE_TIF = "../data/wurl/texture_shade_75.tif"  
OUTPUT_PLY_PATH = "../data/wurl/terrain_shading_final.ply"    
# --------------------------

def main():
    print(f"Loading texture shader raster: {TEXTURE_SHADE_TIF}...")
    with rasterio.open(TEXTURE_SHADE_TIF) as src:
        tex_data = src.read(1)  
        img_h, img_w = tex_data.shape
        
    print(f"Reading input vertex-based PLY mesh: {INPUT_PLY_PATH}...")
    plydata = PlyData.read(INPUT_PLY_PATH)
    vertices = plydata['vertex']
    
    x, y = vertices['x'], vertices['y']
    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()
    
    range_x = (max_x - min_x) if max_x != min_x else 1.0
    range_y = (max_y - min_y) if max_y != min_y else 1.0
    
    print("Mapping vertex coordinates directly to raster pixels...")
    u = (x - min_x) / range_x
    v = (y - min_y) / range_y
    
    cols = np.clip((u * (img_w - 1)).astype(np.int32), 0, img_w - 1)
    rows = np.clip(((1.0 - v) * (img_h - 1)).astype(np.int32), 0, img_h - 1)
    
    sampled_blue_channel = tex_data[rows, cols]
    
    print("Packing channels directly into Vertex arrays...")
    dtype_dict = []
    for name in vertices.data.dtype.names:
        dtype_dict.append((name, vertices.data.dtype[name]))
        
    # If the file doesn't have the color properties yet, inject their headers
    if not all(prop in vertices.data.dtype.names for prop in ['red', 'green', 'blue']):
        dtype_dict.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])
        
    new_vertex_data = np.empty(vertices.count, dtype=dtype_dict)
    
    for name in vertices.data.dtype.names:
        if name not in ['red', 'green', 'blue', 'alpha']:
            new_vertex_data[name] = vertices[name]
            
    # Keep Blender's vertex-baked Red (Shadow) and Green (AO) channels intact
    new_vertex_data['red'] = vertices['red']
    new_vertex_data['green'] = vertices['green']
    # Inject the Texture Shade array directly into Blue
    new_vertex_data['blue'] = sampled_blue_channel
    new_vertex_data['alpha'] = np.full(vertices.count, 255, dtype=np.uint8)
    
    elements_list = [PlyElement.describe(new_vertex_data, 'vertex')]
    if 'face' in plydata:
        elements_list.append(plydata['face'])
        
    print(f"Writing final robust vertex-packed file: {OUTPUT_PLY_PATH}...")
    PlyData(elements_list, text=False).write(OUTPUT_PLY_PATH)
    print("Success! Vertex architecture is locked down.")

if __name__ == "__main__":
    main()