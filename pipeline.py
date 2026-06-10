import os
import csv
import gc
import sys
import numpy as np
import pyvista as pv
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

class CartographicPipeline:
    def __init__(self):
        self.mesh = None
        self.mesh_actor = None
        self.lightweight_points = None
        self._cached_arrays = {}
        self.active_layer_idx = 0
        
        # 1. Build standard high-fidelity USGS NLCD color definitions
        self.nlcd_lut = self._build_nlcd_lookup_table()
        
        # 2. Build Soil Palette dynamically from your project folder CSV schema
        self.soil_lut = self._build_soil_lookup_table("soil_color_lookup_extended_na.csv")

    def log(self, message):
        # Explicit un-silenced standard out streaming for real-time console tracing
        print(f"[PIPELINE DIAGNOSTIC] {message}")
        sys.stdout.flush()

    def _build_nlcd_lookup_table(self):
        """Constructs an official multi-colored USGS NLCD lookup index."""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256) 
        lut.Build()
        for i in range(256):
            lut.SetTableValue(i, 0.12, 0.16, 0.23, 1.0)
            
        nlcd_colors = {
            11: (70, 107, 159), 12: (209, 222, 248), 21: (222, 197, 197),
            22: (217, 146, 130), 23: (235, 0, 0), 24: (171, 0, 0),
            31: (179, 172, 159), 41: (104, 171, 95), 42: (28, 95, 44),
            43: (181, 197, 143), 52: (204, 184, 121), 71: (223, 223, 194),
            81: (220, 217, 57), 82: (171, 108, 40), 90: (184, 217, 235),
            95: (108, 159, 184)
        }
        for code, rgb in nlcd_colors.items():
            lut.SetTableValue(code, rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, 1.0)
        lut.SetTableRange(0, 255)
        return lut

    def _build_soil_lookup_table(self, csv_path):
        lut = vtk.vtkLookupTable()
        if not os.path.exists(csv_path):
            lut.SetNumberOfTableValues(256)
            lut.Build()
            for i in range(256):
                f = i / 255.0
                lut.SetTableValue(i, 0.35 - f*0.1, 0.25 - f*0.08, 0.18 - f*0.05, 1.0)
            lut.SetTableRange(0, 255)
            return lut

        mapping = {}
        target_slice = 125  
        with open(csv_path, mode='r', encoding='utf-8') as f:
            f.readline() 
            for line in f:
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 7: continue
                try:
                    row_slice = int(float(parts[3]))
                    if row_slice == target_slice:
                        code = int(float(parts[1]))
                        if code < 0: continue
                        mapping[code] = (int(float(parts[4])), int(float(parts[5])), int(float(parts[6])))
                except (ValueError, IndexError): continue

        if not mapping:
            lut.SetNumberOfTableValues(256)
            lut.Build()
            return lut

        valid_codes = [c for c in mapping.keys() if c < 60000]
        max_real_code = max(valid_codes) if valid_codes else 255
        lut.SetNumberOfTableValues(max_real_code + 1)
        lut.Build()
        for i in range(max_real_code + 1):
            lut.SetTableValue(i, 0.46, 0.41, 0.37, 1.0)
        for code, rgb in mapping.items():
            if 0 <= code <= max_real_code:
                lut.SetTableValue(code, rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, 1.0)
        return lut

    def load_mesh(self, file_path, plotter, progress_callback=None):
        def update_progress(val, message):
            if progress_callback: progress_callback(val, message)

        self.log(f"Starting execution loop for file: {file_path}")
        update_progress(0, "Opening PLY asset architecture...")

        num_vertices = None
        num_faces = None
        header_offset = None

        with open(file_path, 'rb') as f:
            header_bytes = b""
            while b"end_header" not in header_bytes:
                line = f.readline()
                if not line: break
                header_bytes += line
                line_str = line.decode('ascii', errors='ignore').strip()
                if line_str.startswith("element vertex"):
                    num_vertices = int(line_str.split()[-1])
                elif line_str.startswith("element face"):
                    num_faces = int(line_str.split()[-1])
            header_offset = len(header_bytes)

        if num_vertices is None or num_faces is None or header_offset is None:
            raise ValueError("[FATAL] Malformed PLY file structure.")

        self.log(f"Parsed Header Header Offset={header_offset} bytes | Vertices={num_vertices:,} | Faces={num_faces:,}")
        update_progress(5, f"Header mapped. Parsing {num_vertices:,} vertices...")

        vertex_dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')
        ])
        v_data = np.fromfile(file_path, dtype=vertex_dtype, count=num_vertices, offset=header_offset)
        vertices = np.column_stack((v_data['x'], v_data['y'], v_data['z']))
        
        self.log(f"Vertices read into NumPy. Memory Shape={vertices.shape}")
        update_progress(15, f"Vertices loaded. Allocating topology memory grid...")

        face_stride = 21  
        num_corners = 3   
        face_start_offset = header_offset + (num_vertices * 28)
        
        face_indices = np.empty((num_faces, 3), dtype=np.int32)
        landcover_data = np.empty(num_faces, dtype=np.int32)
        soil_data = np.empty(num_faces, dtype=np.int32)

        chunk_count = 50
        faces_per_chunk = num_faces // chunk_count
        
        with open(file_path, 'rb') as f:
            f.seek(face_start_offset)
            for i in range(chunk_count):
                start_face = i * faces_per_chunk
                end_face = num_faces if i == (chunk_count - 1) else (start_face + faces_per_chunk)
                current_chunk_size = end_face - start_face
                
                raw_bytes = f.read(current_chunk_size * face_stride)
                chunk_faces = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((current_chunk_size, face_stride))
                
                face_indices[start_face:end_face, 0] = chunk_faces[:, 1:5].view(np.int32).ravel()
                face_indices[start_face:end_face, 1] = chunk_faces[:, 5:9].view(np.int32).ravel()
                face_indices[start_face:end_face, 2] = chunk_faces[:, 9:13].view(np.int32).ravel()
                
                landcover_data[start_face:end_face] = chunk_faces[:, 13:17].view(np.int32).ravel()
                soil_data[start_face:end_face] = chunk_faces[:, 17:21].view(np.int32).ravel()
                
                current_pct = int(15 + (i / chunk_count) * 50)
                update_progress(current_pct, f"Streaming topology block: {end_face:,} / {num_faces:,} entries")

        self.log("Face byte stride extraction successful.")
        update_progress(68, "Compiling flat topological grid matrix...")
        cells = np.empty((num_faces, num_corners + 1), dtype=np.int32)
        cells[:, 0] = num_corners
        cells[:, 1:] = face_indices

        del face_indices
        gc.collect()

        update_progress(72, "Generating solid surface geometry...")
        cells_flat = cells.ravel()
        
        self.log("Assembling pyvista.PolyData allocation...")
        self.mesh = pv.PolyData(vertices, cells_flat)
        self.log(f"PolyData Mesh Created Successfully. Bounds={self.mesh.bounds}")

        del cells, cells_flat
        gc.collect()

        step = max(1, num_vertices // 40000)
        self.lightweight_points = vertices[::step, :].copy()
        self.log(f"Generated lightweight downsampled tracking points. Shape={self.lightweight_points.shape}")

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
            self.log(f"Attached Point Scalar Array: '{name}'")
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

        self.log("Categorical cell textures successfully bound to spatial context.")
        del landcover_data, soil_data, lc_contiguous, soil_contiguous
        gc.collect()

        update_progress(98, "Mesh loaded completely. Finalizing engine components...")

    def _inject_shaders(self):
        if not self.mesh_actor:
            self.log("WARNING: Attempted _inject_shaders on a null actor reference.")
            return
        
        self.log("Injecting hardware GLSL shader code block overrides into Actor custom definitions...")
        sp = self.mesh_actor.GetShaderProperty()
        sp.ClearAllShaderReplacements()
        
        # 1. Declare custom uniform parameters in the Vertex Shader Header
        decl_code = """
            uniform vec2 u_focalCenter;
            uniform float u_maxDist;
            uniform float u_amplitude;
            uniform float u_kDecay;
        """
        sp.AddVertexShaderReplacement("//VTK::CustomUniforms::Dec", True, decl_code, False)

        # 2. Implement Progressive Warp and assign structural GLSL output variables cleanly
        impl_code = """
            // Calculate distance utilizing native OpenGLActor model coordinates
            float d = distance(vertexMC.xy, u_focalCenter);
            float normDist = clamp(d / u_maxDist, 0.0, 1.0);
            float verticalLift = u_amplitude * (1.0 - exp(-u_kDecay * normDist));
            
            // Warp the local vertex positions prior to matrix space projection
            vec4 warpedVertex = vec4(vertexMC.x, vertexMC.y, vertexMC.z + verticalLift, 1.0);
            
            // Assign mandatory VTK internal positional targets
            vertexVCVSOutput = MCVCMatrix * warpedVertex;
            gl_Position = MCDCMatrix * warpedVertex;
        """
        # Overriding the core positions implementation block completely replaces the native projection
        sp.AddVertexShaderReplacement("//VTK::PositionVC::Impl", True, impl_code, False)
        self.log("Shader strings injected successfully.")

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

        self.log(f"Shifting hardware color mapping strategy to: {target_array} (Layer index={self.active_layer_idx})")

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
            mapper.SetScalarRange(0.0, 1.0)

        elif self.active_layer_idx == 4:
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(target_array)
            self.nlcd_lut.SetUseAboveRangeColor(True)
            self.nlcd_lut.SetUseBelowRangeColor(True)
            self.nlcd_lut.SetAboveRangeColor(0.12, 0.16, 0.23, 1.0)
            self.nlcd_lut.SetBelowRangeColor(0.12, 0.16, 0.23, 1.0)
            mapper.SetLookupTable(self.nlcd_lut)
            mapper.SetScalarRange(0, 255)

        elif self.active_layer_idx == 5:
            mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(target_array)
            self.soil_lut.SetUseAboveRangeColor(True)
            self.soil_lut.SetUseBelowRangeColor(True)
            fallback_dark_slate = [0.12, 0.14, 0.18, 1.0]
            self.soil_lut.SetAboveRangeColor(fallback_dark_slate)
            self.soil_lut.SetBelowRangeColor(fallback_dark_slate)
            mapper.SetLookupTable(self.soil_lut)
            mapper.SetScalarRange(self.soil_lut.GetTableRange())

        mapper.Update()
        self.mesh_actor.Modified()
        self.log("Hardware lookup update cycle completed.")

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