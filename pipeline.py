import os
import gc
import sys
import numpy as np
import pyvista as pv
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

# =================================================================
# GLOBAL DIAGNOSTIC LEAK DETECTOR
# =================================================================
# If the script succeeds with this list active, it proves that 
# VTK was crashing due to Python garbage collecting shallow-copied 
# numpy arrays (deep=False).
# =================================================================
DEBUG_LEAK_DETECTOR = []

class CartographicPipeline:
    def __init__(self):
        self.mesh = None
        self.mesh_actor = None
        self.lightweight_points = None
        self._cached_arrays = {}
        
        # Color palettes for classification layers
        self.lc_colors = {
            0: [0.118, 0.227, 0.541], 
            1: [0.133, 0.773, 0.369], 
            2: [0.918, 0.702, 0.031], 
            3: [0.937, 0.267, 0.267], 
            4: [0.443, 0.443, 0.478]
        }
        self.soil_colors = {
            0: [0.471, 0.208, 0.059], 
            1: [0.706, 0.325, 0.035], 
            2: [0.851, 0.467, 0.024], 
            3: [0.631, 0.631, 0.667]
        }
        self.active_layer_idx = 0

    def log(self, message):
        print(f"[PIPELINE] {message}", flush=True)

    # ... (Keep class initialization, colors, and shaders exactly as you have them) ...

    def load_mesh(self, file_path, plotter, progress_callback=None):
        def update_progress(val, message):
            if progress_callback:
                progress_callback(val, message)

        update_progress(0, "Opening PLY asset architecture...")

        # -----------------------------------------------------------------
        # 1. HEADER PARSING
        # -----------------------------------------------------------------
        num_vertices = None
        num_faces = None
        header_offset = None

        with open(file_path, 'rb') as f:
            header_bytes = b""
            while b"end_header" not in header_bytes:
                line = f.readline()
                if not line:
                    break
                header_bytes += line
                line_str = line.decode('ascii', errors='ignore').strip()
                if line_str.startswith("element vertex"):
                    num_vertices = int(line_str.split()[-1])
                elif line_str.startswith("element face"):
                    num_faces = int(line_str.split()[-1])
            header_offset = len(header_bytes)

        if num_vertices is None or num_faces is None or header_offset is None:
            raise ValueError("[FATAL] Malformed PLY file structure.")

        update_progress(5, f"Header mapped. Parsing {num_vertices:,} vertices...")

        # -----------------------------------------------------------------
        # 2. PARSE VERTICES
        # -----------------------------------------------------------------
        vertex_dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')
        ])
        v_data = np.fromfile(file_path, dtype=vertex_dtype, count=num_vertices, offset=header_offset)
        vertices = np.column_stack((v_data['x'], v_data['y'], v_data['z']))
        
        update_progress(15, f"Vertices loaded. Allocating topology memory grid...")

        # -----------------------------------------------------------------
        # 3. CHUNKED FACE STRIDE PROCESSING (SMOOTH PROGRESS)
        # -----------------------------------------------------------------
        face_stride = 21  
        num_corners = 3   
        face_start_offset = header_offset + (num_vertices * 28)
        
        # Pre-allocate output arrays to avoid slow dynamically resizing arrays
        face_indices = np.empty((num_faces, 3), dtype=np.int32)
        landcover_data = np.empty(num_faces, dtype=np.int32)
        soil_data = np.empty(num_faces, dtype=np.int32)

        # Process the 56M faces in 50 distinct structural slices
        chunk_count = 50
        faces_per_chunk = num_faces // chunk_count
        
        with open(file_path, 'rb') as f:
            f.seek(face_start_offset)
            
            for i in range(chunk_count):
                # Calculate precise dynamic boundaries
                start_face = i * faces_per_chunk
                end_face = num_faces if i == (chunk_count - 1) else (start_face + faces_per_chunk)
                current_chunk_size = end_face - start_face
                
                # Stream binary block directly from disk
                raw_bytes = f.read(current_chunk_size * face_stride)
                chunk_faces = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((current_chunk_size, face_stride))
                
                # Unpack sub-byte arrays cleanly
                face_indices[start_face:end_face, 0] = chunk_faces[:, 1:5].view(np.int32).ravel()
                face_indices[start_face:end_face, 1] = chunk_faces[:, 5:9].view(np.int32).ravel()
                face_indices[start_face:end_face, 2] = chunk_faces[:, 9:13].view(np.int32).ravel()
                
                landcover_data[start_face:end_face] = chunk_faces[:, 13:17].view(np.int32).ravel()
                soil_data[start_face:end_face] = chunk_faces[:, 17:21].view(np.int32).ravel()
                
                # Incrementally scale progress tracker between 15% and 65%
                current_pct = int(15 + (i / chunk_count) * 50)
                update_progress(current_pct, f"Streaming topology block: {end_face:,} / {num_faces:,} entries")

        # -----------------------------------------------------------------
        # 4. CONSTRUCT VTK CELL LAYOUT
        # -----------------------------------------------------------------
        update_progress(68, "Compiling flat topological grid matrix...")
        cells = np.empty((num_faces, num_corners + 1), dtype=np.int32)
        cells[:, 0] = num_corners
        cells[:, 1:] = face_indices

        del face_indices
        gc.collect()

        update_progress(72, "Generating solid surface geometry...")
        cells_flat = cells.ravel()
        self.mesh = pv.PolyData(vertices, cells_flat)

        del cells, cells_flat
        gc.collect()

        # -----------------------------------------------------------------
        # 5. ATTACH SCALAR ATTRIBUTES
        # -----------------------------------------------------------------
        step = max(1, num_vertices // 40000)
        self.lightweight_points = vertices[::step, :].copy()

        del vertices
        gc.collect()

        update_progress(80, "Normalizing RGBA point channels...")
        rgba_float = np.empty((num_vertices, 4), dtype=np.float32)
        rgba_float[:, 0] = v_data['red'] / 255.0
        rgba_float[:, 1] = v_data['green'] / 255.0
        rgba_float[:, 2] = v_data['blue'] / 255.0
        rgba_float[:, 3] = v_data['alpha'] / 255.0

        del v_data
        gc.collect()

        names = ["Hillshade", "AO", "Texture", "Vegetation"]
        for i, name in enumerate(names):
            channel = np.ascontiguousarray(rgba_float[:, i])
            vtk_arr = numpy_to_vtk(channel, deep=True, array_type=vtk.VTK_FLOAT)
            vtk_arr.SetName(name)
            self.mesh.GetPointData().AddArray(vtk_arr)
            del channel

        del rgba_float
        gc.collect()

        update_progress(90, "Deep-copying categorical data blocks...")
        lc_contiguous = np.ascontiguousarray(landcover_data, dtype=np.int32)
        soil_contiguous = np.ascontiguousarray(soil_data, dtype=np.int32)
        
        lc_arr = numpy_to_vtk(lc_contiguous, deep=True, array_type=vtk.VTK_INT)
        lc_arr.SetName("landcover")
        self.mesh.GetCellData().AddArray(lc_arr)

        soil_arr = numpy_to_vtk(soil_contiguous, deep=True, array_type=vtk.VTK_INT)
        soil_arr.SetName("soil_color")
        self.mesh.GetCellData().AddArray(soil_arr)

        del landcover_data, soil_data, lc_contiguous, soil_contiguous
        gc.collect()

        update_progress(98, "Mesh loaded completely. Finalizing engine components...")

    def _inject_shaders(self):
        sp = self.mesh_actor.GetShaderProperty()
        sp.ClearAllShaderReplacements()
        impl_code = """
            float d = distance(vertexMC.xy, u_focalCenter);
            float normDist = d / u_maxDist;
            float verticalLift = u_amplitude * (1.0 - exp(-u_kDecay * normDist));
            vertexMC.z += verticalLift;
        """
        sp.AddVertexShaderReplacement("//VTK::PositionVC::Impl", False, impl_code, False)

    def update_shader_uniforms(self, cx, cy, max_dist, amplitude, k_decay):
        if not self.mesh_actor: return
        shader_params = self.mesh_actor.GetShaderProperty().GetVertexCustomUniforms()
        shader_params.SetUniformf("u_amplitude", amplitude)
        shader_params.SetUniformf("u_kDecay", k_decay)
        shader_params.SetUniform2f("u_focalCenter", (cx, cy))
        shader_params.SetUniformf("u_maxDist", max_dist)

    def update_hardware_lut(self, layer_idx=None):
        if layer_idx is not None:
            self.active_layer_idx = layer_idx
        
        if not self.mesh_actor: return
        
        mapper = self.mesh_actor.GetMapper()
        layer_map = {0: "Hillshade", 1: "AO", 2: "Texture", 3: "Vegetation", 4: "landcover", 5: "soil_color"}
        target_array = layer_map.get(self.active_layer_idx, "Hillshade")

        if self.active_layer_idx in [0, 1, 2, 3]:
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray(target_array)
            
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(256)
            lut.SetTableRange(0.0, 1.0)
            for idx in range(256):
                val = idx / 255.0
                lut.SetTableValue(idx, val, val, val, 1.0)
            lut.Build()
            mapper.SetLookupTable(lut)
        else:
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(target_array)
            
            palette = self.lc_colors if self.active_layer_idx == 4 else self.soil_colors
            max_id = max(palette.keys()) if palette else 0
            
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(max_id + 1)
            lut.SetTableRange(0, max_id)
            
            for cid in range(max_id + 1):
                if cid in palette:
                    r, g, b = palette[cid]
                    lut.SetTableValue(cid, r, g, b, 1.0)
                else:
                    lut.SetTableValue(cid, 0.0, 0.0, 0.0, 1.0)
            lut.Build()
            mapper.SetLookupTable(lut)

        mapper.Update()

    def execute_multipass_export(self, plotter, base_filename="export"):
        if not self.mesh_actor: return
        plotter.render_window.SetAnimate(0)
        original_layer = self.active_layer_idx
        
        layer_names = ["Hillshade", "AO", "Texture", "Vegetation", "Landcover", "Soil"]
        for i, name in enumerate(layer_names):
            self.update_hardware_lut(layer_idx=i)
            plotter.render()
            plotter.screenshot(f"{base_filename}_{name}.png", transparent_background=True)
            
        self.update_hardware_lut(layer_idx=original_layer)
        plotter.render_window.SetAnimate(1)
        plotter.render()