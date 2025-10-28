import argparse
import os
import yaml
import pickle
import torch
import numpy as np
import trimesh
import glob
import json
from torch.utils.data import DataLoader
from ignite.metrics import IoU, ConfusionMatrix
from torchmetrics import F1Score, Accuracy


def overall_accuracy(preds, targets, ignore_index=None):
    """
    preds: torch.Tensor of shape (N,) containing predicted class indices
    targets: torch.Tensor of the same shape containing ground-truth class indices
    ignore_index: class index to ignore in accuracy computation (e.g., for padded tokens)
    """
    assert preds.shape == targets.shape, "Predictions and targets must have the same shape"
    
    # Create mask to exclude ignore_index (e.g., padded tokens)
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]
    
    # If no valid samples after filtering, return 0
    if len(targets) == 0:
        return 0.0
    
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

from tools.mesh_dataset import MeshDataset, custom_collate_fn
from mesh_texture_dataset import MeshTextureDataset, texture_custom_collate_fn
from tools.model_base import nomeformer
from tools.downst import DownstreamClassifier
from integrated_texture_geometry_model import IntegratedTextureGeometryModel, IntegratedDownstreamClassifier
from tools.auto_mesh_slicer import (
    parse_ply_file, 
    generate_automatic_bounding_boxes,
    generate_adaptive_bounding_boxes,
    slice_mesh_with_bounding_boxes,
    write_ply_file
)


def slice_large_mesh(mesh_path, output_dir, grid_divisions=(3, 3, 1), slicing_mode='grid', 
                     target_faces_per_box=5000, min_boxes=4, max_boxes=50, save_mapping=True,
                     texture_dir=None, texture_output_dir=None):
    """
    Slice a large mesh into smaller parts for efficient inference.
    
    Args:
        mesh_path: Path to the original large mesh
        output_dir: Directory to save sliced meshes
        grid_divisions: Tuple of (x, y, z) divisions for grid-based slicing
        slicing_mode: 'grid' or 'adaptive'
        target_faces_per_box: Target faces per box for adaptive mode
        min_boxes: Minimum boxes for adaptive mode
        max_boxes: Maximum boxes for adaptive mode
        save_mapping: Whether to save mapping from slice faces to original faces
        
    Returns:
        List of paths to sliced mesh files
    """
    print(f"\n{'='*60}")
    print(f"Slicing large mesh: {os.path.basename(mesh_path)}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the mesh
    print("Loading mesh...")
    vertices, faces = parse_ply_file(mesh_path)
    print(f"Loaded {len(vertices)} vertices and {len(faces)} faces")
    
    # Generate bounding boxes
    if slicing_mode == 'adaptive':
        print(f"\nGenerating adaptive bounding boxes (target: {target_faces_per_box} faces/box)...")
        bounding_boxes = generate_adaptive_bounding_boxes(
            vertices, faces, target_faces_per_box, min_boxes, max_boxes
        )
    else:  # grid mode
        print(f"\nGenerating grid-based bounding boxes {grid_divisions}...")
        bounding_boxes = generate_automatic_bounding_boxes(
            vertices, grid_divisions=grid_divisions, overlap=0.0
        )
    
    print(f"Generated {len(bounding_boxes)} bounding boxes")
    
    # Slice the mesh with face mapping tracking
    print("\nSlicing mesh...")
    split_meshes = []
    face_mappings = {}  # slice_name -> list of original face indices
    
    for bbox_index, bbox in enumerate(bounding_boxes):
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        sliced_face_indices = []  # Track which original faces go into this slice
        sliced_faces = []
        
        for orig_face_idx, face_indices in enumerate(faces):
            v0_idx, v1_idx, v2_idx = int(face_indices[0]), int(face_indices[1]), int(face_indices[2])
            v0 = vertices[v0_idx, :3]
            v1 = vertices[v1_idx, :3]
            v2 = vertices[v2_idx, :3]

            # Check if all vertices are inside the bounding box
            if all(x_min <= v[0] <= x_max and y_min <= v[1] <= y_max and z_min <= v[2] <= z_max 
                   for v in [v0, v1, v2]):
                sliced_faces.append(face_indices)
                sliced_face_indices.append(orig_face_idx)

        if sliced_faces:
            # Get unique vertices used by these faces
            used_vertex_indices = set()
            for face in sliced_faces:
                used_vertex_indices.update(face[:3].astype(int))
            
            # Create mapping from old to new vertex indices
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_vertex_indices))}
            
            # Create new vertices and faces arrays
            cleaned_vertices = vertices[sorted(used_vertex_indices)]
            cleaned_faces = []
            for face in sliced_faces:
                new_face = np.array([old_to_new[int(face[0])], old_to_new[int(face[1])], old_to_new[int(face[2])]] + list(face[3:]))
                cleaned_faces.append(new_face)
            cleaned_faces = np.array(cleaned_faces, dtype=object)
            
            # Keep mapping aligned with split_meshes index (0..len(split_meshes)-1)
            # Use the slice index, not the bbox index (some bboxes may produce no slice)
            slice_index = len(split_meshes)
            split_meshes.append((cleaned_vertices, cleaned_faces, bbox))
            face_mappings[slice_index] = sliced_face_indices
    
    print(f"Created {len(split_meshes)} mesh slices")
    
    # Write sliced meshes and save mappings
    print("\nWriting sliced meshes...")
    sliced_mesh_paths = []
    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    # Texture setup (optional)
    texture_slice_map = {}
    original_texture_list = None
    if texture_dir is not None:
        # Determine texture base name (remove _labeled suffix if present)
        texture_base_name = base_name[:-8] if base_name.endswith('_labeled') else base_name
        candidate = os.path.join(texture_dir, texture_base_name + '.pkl')
        candidate_alt = os.path.join(texture_dir, texture_base_name + '_pixels_test.pkl')
        texture_path = candidate if os.path.isfile(candidate) else (candidate_alt if os.path.isfile(candidate_alt) else None)
        if texture_path is None:
            print(f"Warning: No texture file found for {base_name} in {texture_dir}")
        else:
            try:
                with open(texture_path, 'rb') as pf:
                    original_texture_list = pickle.load(pf)
                # Ensure output dir for textures
                texture_output_dir = texture_output_dir or output_dir
                os.makedirs(texture_output_dir, exist_ok=True)
                print(f"Loaded texture data from {texture_path} (faces: {len(original_texture_list) if isinstance(original_texture_list, list) else 'unknown'})")
            except Exception as e:
                print(f"Warning: Failed to load texture file {texture_path}: {e}")
                original_texture_list = None
    
    for i, (slice_vertices, slice_faces, bbox) in enumerate(split_meshes):
        output_path = os.path.join(output_dir, f"{base_name}_slice_{i:03d}.ply")
        write_ply_file(output_path, slice_vertices, slice_faces)
        sliced_mesh_paths.append(output_path)
        
        reduction_percent = 100.0 * (1 - len(slice_vertices) / len(vertices))
        print(f"  Slice {i:03d}: {len(slice_faces)} faces, {len(slice_vertices)} vertices "
              f"(reduced by {reduction_percent:.1f}%)")
        # If textures are available, write per-slice texture .pkl aligned to slice face order
        if original_texture_list is not None:
            try:
                # face_mappings[i] aligns with the order of faces in this slice
                original_indices = face_mappings[i]
                slice_textures = []
                for idx in original_indices:
                    # Bounds check and fallback to empty if missing
                    if isinstance(original_texture_list, list) and 0 <= idx < len(original_texture_list):
                        slice_textures.append(original_texture_list[idx])
                    else:
                        slice_textures.append((0, []))
                tex_out_path = os.path.join(texture_output_dir or output_dir, f"{base_name}_slice_{i:03d}.pkl")
                with open(tex_out_path, 'wb') as tf:
                    pickle.dump(slice_textures, tf)
                texture_slice_map[f"{base_name}_slice_{i:03d}.ply"] = os.path.basename(tex_out_path)
                print(f"    Wrote texture slice: {tex_out_path}")
            except Exception as e:
                print(f"    Warning: Failed writing texture slice {i:03d}: {e}")
    
    # Save mapping file if requested
    if save_mapping:
        mapping_path = os.path.join(output_dir, f"{base_name}_face_mapping.json")
        print(f"Saving face mapping to: {mapping_path}")
        mapping_data = {
            'original_mesh': mesh_path,
            'original_num_faces': len(faces),
            'slices': {
                f"{base_name}_slice_{i:03d}.ply": face_mappings[i]
                for i in range(len(split_meshes))
            }
        }
        # Include texture slice filenames if generated
        if texture_slice_map:
            mapping_data['texture_slices'] = texture_slice_map
        with open(mapping_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        print(f"\nSaved face mapping to: {mapping_path}")
    
    print(f"\n{'='*60}")
    print(f"Slicing complete! Created {len(sliced_mesh_paths)} slices")
    print(f"Saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return sliced_mesh_paths


def process_mesh_directory_with_slicing(mesh_dir, slice_output_dir, auto_slice_config):
    """
    Process all meshes in a directory, slicing if needed.
    
    Args:
        mesh_dir: Directory containing original meshes
        slice_output_dir: Directory to save sliced meshes
        auto_slice_config: Configuration dict for slicing
        
    Returns:
        Directory path containing meshes to use for inference
    """
    # Check if we should slice the meshes
    should_slice = auto_slice_config.get('enabled', True)
    
    if not should_slice:
        print("Auto-slicing disabled. Using original meshes.")
        return mesh_dir
    
    # Check if sliced meshes already exist (check for .ply files specifically)
    existing_slices = glob.glob(os.path.join(slice_output_dir, "*.ply")) if os.path.exists(slice_output_dir) else []
    if len(existing_slices) > 0:
        reuse_existing = auto_slice_config.get('reuse_existing_slices', True)
        if reuse_existing:
            print(f"\nSliced meshes already exist at {slice_output_dir}")
            print("Reusing existing slices. Set 'reuse_existing_slices: false' to regenerate.")
            return slice_output_dir
        else:
            print(f"\nRemoving existing slices in {slice_output_dir}...")
            for f in existing_slices:
                os.remove(f)
    
    # Get all mesh files
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.ply")))
    
    if not mesh_files:
        print(f"Warning: No PLY files found in {mesh_dir}")
        return mesh_dir
    
    print(f"\nFound {len(mesh_files)} mesh(es) to slice")
    
    # Get slicing parameters
    slicing_mode = auto_slice_config.get('mode', 'adaptive')  # Default to adaptive
    
    if slicing_mode == 'adaptive':
        # Adaptive mode parameters
        target_faces_per_box = auto_slice_config.get('target_faces_per_box', 30000)
        min_boxes = auto_slice_config.get('min_boxes', 4)
        max_boxes = auto_slice_config.get('max_boxes', 50)
        grid_divisions = None  # Not used in adaptive mode
    else:
        # Grid mode parameters
        grid_divisions = tuple(auto_slice_config.get('grid_divisions', [3, 3, 1]))
        target_faces_per_box = None  # Not used in grid mode
        min_boxes = None
        max_boxes = None
    
    # Texture slicing parameters (optional)
    texture_dir_cfg = auto_slice_config.get('texture_dir', None)
    texture_output_dir_cfg = auto_slice_config.get('texture_output_dir', slice_output_dir)

    # Slice each mesh
    all_sliced_paths = []
    for mesh_path in mesh_files:
        try:
            if slicing_mode == 'adaptive':
                sliced_paths = slice_large_mesh(
                    mesh_path=mesh_path,
                    output_dir=slice_output_dir,
                    slicing_mode=slicing_mode,
                    target_faces_per_box=target_faces_per_box,
                    min_boxes=min_boxes,
                    max_boxes=max_boxes,
                    texture_dir=texture_dir_cfg,
                    texture_output_dir=texture_output_dir_cfg
                )
            else:  # grid mode
                sliced_paths = slice_large_mesh(
                    mesh_path=mesh_path,
                    output_dir=slice_output_dir,
                    grid_divisions=grid_divisions,
                    slicing_mode=slicing_mode,
                    texture_dir=texture_dir_cfg,
                    texture_output_dir=texture_output_dir_cfg
                )
            all_sliced_paths.extend(sliced_paths)
        except Exception as e:
            print(f"Error slicing {mesh_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTotal sliced meshes created: {len(all_sliced_paths)}")
    
    return slice_output_dir if all_sliced_paths else mesh_dir


def _parse_ply_properties(ply_path: str):
    """Parse PLY file to extract properties and data.
    
    Returns:
        dict with keys: 'vertex_properties', 'face_properties', 'vertex_data', 'face_data'
    """
    result = {
        'vertex_properties': [],
        'face_properties': [],
        'vertex_data': [],
        'face_data': []
    }
    
    vertex_count = 0
    face_count = 0
    current_element = None
    
    with open(ply_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
                current_element = 'vertex'
            elif line.startswith('element face'):
                face_count = int(line.split()[2])
                current_element = 'face'
            elif line.startswith('property'):
                parts = line.split()
                if len(parts) >= 3:
                    # Handle both "property type name" and "property list uchar int name"
                    if parts[1] == 'list':
                        # Format: property list uchar int vertex_indices
                        prop_type = 'list'
                        list_size_type = parts[2]
                        list_item_type = parts[3]
                        prop_name = parts[4] if len(parts) >= 5 else 'unknown'
                        result[current_element + '_properties'].append({
                            'type': f'list {list_size_type} {list_item_type}', 
                            'name': prop_name
                        })
                    else:
                        prop_type = parts[1]
                        prop_name = parts[2]
                        result[current_element + '_properties'].append({'type': prop_type, 'name': prop_name})
            elif line.startswith('end_header'):
                # Now read data
                for _ in range(vertex_count):
                    data_line = f.readline().strip()
                    result['vertex_data'].append(data_line)
                
                for _ in range(face_count):
                    data_line = f.readline().strip()
                    result['face_data'].append(data_line)
                break
    
    return result


def _write_ascii_ply_with_labels(file_path: str, vertices_xyz: np.ndarray, faces_idx: np.ndarray, 
                                   vertex_labels: np.ndarray, face_labels: np.ndarray,
                                   preserve_properties=None) -> None:
    """Write an ASCII PLY with per-vertex and per-face classification labels.

    Args:
        file_path: Output PLY file path
        vertices_xyz: (N, 3) float32/float64 positions from the actual mesh
        faces_idx: (M, 3) int vertex indices (triangles) from the actual mesh
        vertex_labels: (N,) int vertex classification labels
        face_labels: (M,) int face classification labels
        preserve_properties: Optional dict with original mesh properties and data to preserve
    """
    num_vertices = int(vertices_xyz.shape[0])
    num_faces = int(faces_idx.shape[0])

    with open(file_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        
        # Write vertex properties
        cls_pos = None
        skip_indices = []  # Track which vertex indices to skip (nx, ny, nz)
        if preserve_properties and preserve_properties.get('vertex_properties'):
            # Preserve all properties except skip nx, ny, nz (trimesh drops them) and replace 'cls' with 'label'
            for idx, prop in enumerate(preserve_properties['vertex_properties']):
                if prop['name'] in ['nx', 'ny', 'nz']:
                    skip_indices.append(idx)
                    continue  # Skip normal properties
                elif prop['name'] in ['cls', 'label']:
                    f.write("property int cls\n")
                    cls_pos = idx
                else:
                    f.write(f"property {prop['type']} {prop['name']}\n")
        else:
            # Default: only x, y, z, cls
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property int cls\n")
        
        f.write(f"element face {num_faces}\n")
        
        # Write face properties
        face_cls_pos = None
        if preserve_properties and preserve_properties.get('face_properties'):
            # Preserve all properties (keep 'cls' as 'cls')
            current_pos = 0  # Track position in face properties
            for prop in preserve_properties['face_properties']:
                if prop['name'] in ['cls', 'label']:
                    f.write("property int cls\n")
                    face_cls_pos = current_pos
                else:
                    f.write(f"property {prop['type']} {prop['name']}\n")
                
                # Track position for later
                if 'list' in prop['type']:
                    # List properties don't add to count until we see the data
                    pass
                else:
                    current_pos += 1
        else:
            # Default: only vertex_indices and cls
            f.write("property list uchar int vertex_indices\n")
            f.write("property int cls\n")
        
        f.write("end_header\n")

        # Vertices
        has_full_vertex_data = preserve_properties and preserve_properties.get('vertex_data') and len(preserve_properties['vertex_data']) == num_vertices
        
        if has_full_vertex_data:
            # Try to preserve original vertex properties while using actual positions
            # This works when the mesh hasn't been sliced
            for i in range(num_vertices):
                if i >= len(vertex_labels):
                    break
                
                # Get original vertex line and split into parts
                original_parts = preserve_properties['vertex_data'][i].split()
                x, y, z = vertices_xyz[i]
                label = int(vertex_labels[i])
                
                # Replace x, y, z with actual mesh positions
                if len(original_parts) >= 3:
                    original_parts[0] = str(float(x))
                    original_parts[1] = str(float(y))
                    original_parts[2] = str(float(z))
                    
                    # Replace cls if it exists
                    if cls_pos is not None and cls_pos < len(original_parts):
                        original_parts[cls_pos] = str(label)
                    elif cls_pos is None:
                        # No cls property existed, append label
                        original_parts.append(str(label))
                    
                    # Filter out values at skip_indices (nx, ny, nz)
                    filtered_parts = [val for idx, val in enumerate(original_parts) 
                                     if idx not in skip_indices]
                    
                    f.write(' '.join(filtered_parts) + '\n')
                else:
                    # Fallback to simple format
                    f.write(f"{float(x)} {float(y)} {float(z)} {label}\n")
        elif preserve_properties and preserve_properties.get('vertex_properties'):
            # We have property info but not matching vertex data
            # Write vertices with zeros for properties we don't have
            vertex_props = preserve_properties['vertex_properties']
            
            for i in range(num_vertices):
                if i >= len(vertex_labels):
                    break
                    
                x, y, z = vertices_xyz[i]
                label = int(vertex_labels[i])
                
                # Build vertex line with all declared properties (skip nx, ny, nz)
                vertex_values = []
                for prop in vertex_props:
                    if prop['name'] in ['nx', 'ny', 'nz']:
                        continue  # Skip normal properties (trimesh drops them)
                    
                    if prop['name'] in ['x', 'y', 'z']:
                        if prop['name'] == 'x':
                            vertex_values.append(str(float(x)))
                        elif prop['name'] == 'y':
                            vertex_values.append(str(float(y)))
                        else:  # z
                            vertex_values.append(str(float(z)))
                    elif prop['name'] in ['cls', 'label']:
                        vertex_values.append(str(label))
                    else:
                        # Property we don't have - fill with zero
                        if prop['type'] in ['int', 'uchar', 'uint', 'short', 'ushort']:
                            vertex_values.append('0')
                        elif prop['type'] in ['float', 'double']:
                            vertex_values.append('0.0')
                        else:
                            vertex_values.append('0')
                
                f.write(' '.join(vertex_values) + '\n')
        else:
            # Use simple format when we don't have property info
            print(f"Note: Using simple format - no property information available")
            for i in range(num_vertices):
                if i >= len(vertex_labels):
                    break
                    
                x, y, z = vertices_xyz[i]
                label = int(vertex_labels[i])
                f.write(f"{float(x)} {float(y)} {float(z)} {label}\n")

        # Faces
        has_full_face_data = preserve_properties and preserve_properties.get('face_data') and len(preserve_properties['face_data']) == num_faces
        
        if has_full_face_data:
            # Try to preserve original face properties
            face_props = preserve_properties.get('face_properties', [])
            
            for j in range(num_faces):
                if j >= len(face_labels):
                    break
                
                # Get original face line and split
                original_parts = preserve_properties['face_data'][j].split()
                if len(original_parts) == 0:
                    continue
                
                # Replace vertex indices with actual mesh indices
                v0, v1, v2 = faces_idx[j]
                label = int(face_labels[j])
                
                # Replace vertex indices (they should be in positions 1-3 after the count)
                if len(original_parts) > 3:
                    try:
                        count = int(original_parts[0])
                        if count >= 3:
                            original_parts[1:4] = [str(int(v0)), str(int(v1)), str(int(v2))]
                    except (ValueError, IndexError):
                        pass
                
                # Check if 'cls' property exists in face properties
                has_cls_property = any(prop['name'] == 'cls' or prop['name'] == 'label' for prop in face_props)
                
                if has_cls_property:
                    # Try to replace the last element if it looks like a cls value
                    # or insert after vertex indices (position 4)
                    if len(original_parts) > 4:
                        # There are extra properties - cls is likely at the end
                        original_parts[-1] = str(label)
                    elif len(original_parts) == 4:
                        # Exactly 4 elements means [count, v0, v1, v2] - no cls yet
                        # Insert at position 4 (after v2)
                        original_parts.append(str(label))
                    else:
                        # Fallback: just append
                        original_parts.append(str(label))
                else:
                    # No cls property existed - append label at end
                    original_parts.append(str(label))
                
                f.write(' '.join(original_parts) + '\n')
        elif preserve_properties and preserve_properties.get('face_properties'):
            # We have property info but not matching face data
            # Write faces with zeros for properties we don't have
            face_props = preserve_properties.get('face_properties', [])
            
            for j in range(num_faces):
                if j >= len(face_labels):
                    break
                    
                v0, v1, v2 = faces_idx[j]
                label = int(face_labels[j])
                
                # Build face line with all declared properties
                # First property is always vertex_indices (list type)
                f.write(f"3 {int(v0)} {int(v1)} {int(v2)}")  # vertex_indices list
                
                # Add other properties
                for prop in face_props:
                    if prop['name'] == 'vertex_indices':
                        continue  # Already written above
                    elif prop['name'] in ['cls', 'label']:
                        f.write(f" {label}")
                    else:
                        # Property we don't have - fill with zero
                        if prop['type'] in ['int', 'uchar', 'uint', 'short', 'ushort']:
                            f.write(" 0")
                        elif prop['type'] in ['float', 'double']:
                            f.write(" 0.0")
                        else:
                            f.write(" 0")
                
                f.write("\n")
        else:
            # Use simple format
            for j in range(num_faces):
                if j >= len(face_labels):
                    break
                    
                v0, v1, v2 = faces_idx[j]
                label = int(face_labels[j])
                f.write(f"3 {int(v0)} {int(v1)} {int(v2)} {label}\n")


def _propagate_labels_to_duplicate_faces(mesh: trimesh.Trimesh, labels: np.ndarray) -> np.ndarray:
    """Copy a representative label to all duplicates so duplicate faces share identical labels.
    
    Args:
        mesh: Trimesh object
        labels: (M,) array of face labels
        
    Returns:
        Updated labels array with duplicates propagated
    """
    faces = mesh.faces
    if faces.size == 0:
        return labels
    faces_norm = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(faces_norm, axis=0, return_inverse=True, return_counts=True)
    if not np.any(counts > 1):
        return labels
    # Group indices by group id
    group_to_indices = {}
    for face_index, group_id in enumerate(inverse):
        gid = int(group_id)
        if gid not in group_to_indices:
            group_to_indices[gid] = []
        group_to_indices[gid].append(int(face_index))
    for indices in group_to_indices.values():
        if len(indices) <= 1:
            continue
        ref = indices[0]
        labels[indices] = labels[ref]
    return labels


def save_mesh_with_predictions(mesh_path: str, predictions: np.ndarray, output_path: str, 
                                dataset=None, mesh_idx: int = None) -> None:
    """Save a mesh with predicted face labels.
    
    Args:
        mesh_path: Path to the original mesh file
        predictions: Array of predicted face labels
        output_path: Path to save the labeled mesh
        dataset: Optional dataset to get face ordering information
        mesh_idx: Optional mesh index in the dataset
    """
    try:
        # Load the mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        num_faces = mesh.faces.shape[0]
        num_vertices = mesh.vertices.shape[0]
        
        # Initialize face labels
        face_labels = np.zeros(num_faces, dtype=np.int32)
        
        # Assign predictions to faces
        # Note: predictions should match the face ordering from the dataset
        pred_len = min(len(predictions), num_faces)
        face_labels[:pred_len] = predictions[:pred_len].astype(np.int32)
        
        # Propagate labels to duplicate faces
        face_labels = _propagate_labels_to_duplicate_faces(mesh, face_labels)
        
        # Derive vertex labels by majority voting from incident faces
        vertex_labels = np.zeros(num_vertices, dtype=np.int32)
        vertex_label_counts = {}  # vertex_idx -> {label: count}
        
        for face_idx, (v0, v1, v2) in enumerate(mesh.faces):
            face_label = int(face_labels[face_idx])
            for v in [v0, v1, v2]:
                if v not in vertex_label_counts:
                    vertex_label_counts[v] = {}
                vertex_label_counts[v][face_label] = vertex_label_counts[v].get(face_label, 0) + 1
        
        # Assign vertex labels based on majority vote
        for v_idx, label_counts in vertex_label_counts.items():
            vertex_labels[v_idx] = max(label_counts.items(), key=lambda x: x[1])[0]
        
        # Try to preserve original PLY properties if available
        preserve_properties = None
        try:
            preserve_properties = _parse_ply_properties(mesh_path)
            if preserve_properties and preserve_properties.get('vertex_properties'):
                vertex_prop_names = [prop['name'] for prop in preserve_properties['vertex_properties']]
                print(f"  Preserving vertex properties: {vertex_prop_names}")
        except Exception as e:
            print(f"Note: Could not preserve original PLY properties: {e}")
            preserve_properties = None
        
        # Save mesh with labels
        vertices_xyz = mesh.vertices.astype(np.float64)
        faces_idx = mesh.faces.astype(np.int64)
        _write_ascii_ply_with_labels(output_path, vertices_xyz, faces_idx, vertex_labels, face_labels,
                                       preserve_properties=preserve_properties)
        
        # Print statistics
        unique_labels, counts = np.unique(face_labels, return_counts=True)
        print(f"  Face label distribution: {dict(zip(unique_labels, counts))}")
        
    except Exception as e:
        print(f"Error saving mesh {mesh_path}: {e}")


def stitch_sliced_predictions(slice_predictions_dir, original_mesh_dir, output_dir, sliced_mesh_dir):
    """
    Stitch together predictions from sliced meshes back to original meshes.
    
    Args:
        slice_predictions_dir: Directory containing predicted sliced meshes
        original_mesh_dir: Directory containing original (pre-sliced) meshes
        output_dir: Directory to save stitched meshes
        sliced_mesh_dir: Directory containing sliced meshes and mapping files
    """
    print("\n" + "="*60)
    print("Stitching sliced predictions back to original meshes...")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all mapping files
    mapping_files = glob.glob(os.path.join(sliced_mesh_dir, "*_face_mapping.json"))
    
    if not mapping_files:
        print("Warning: No face mapping files found. Cannot stitch predictions.")
        print(f"Looked in: {sliced_mesh_dir}")
        return
    
    print(f"Found {len(mapping_files)} mapping file(s)")
    
    for mapping_file in mapping_files:
        try:
            # Load mapping
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            original_mesh_path = mapping_data['original_mesh']
            original_num_faces = mapping_data['original_num_faces']
            slice_mappings = mapping_data['slices']
            
            original_basename = os.path.splitext(os.path.basename(original_mesh_path))[0]
            print(f"\nProcessing: {original_basename}")
            print(f"  Original faces: {original_num_faces}")
            print(f"  Number of slices: {len(slice_mappings)}")
            
            # Initialize array for all face predictions
            all_face_predictions = np.zeros(original_num_faces, dtype=np.int32)
            face_prediction_counts = np.zeros(original_num_faces, dtype=np.int32)  # Track how many times each face was predicted
            
            # Collect predictions from all slices
            for slice_name, original_face_indices in slice_mappings.items():
                # Find the predicted slice mesh
                slice_basename = os.path.splitext(slice_name)[0]
                predicted_slice_path = os.path.join(slice_predictions_dir, f"{slice_basename}_predicted.ply")
                
                if not os.path.exists(predicted_slice_path):
                    print(f"  Warning: Predicted slice not found: {predicted_slice_path}")
                    continue
                
                # Load the predicted slice
                try:
                    slice_mesh = trimesh.load(predicted_slice_path, force='mesh')
                    
                    # Extract face labels from the mesh
                    # Assuming the face labels are stored in the 'cls' property
                    if hasattr(slice_mesh, 'face_attributes') and 'label' in slice_mesh.face_attributes:
                        slice_predictions = slice_mesh.face_attributes['label']
                    elif hasattr(slice_mesh, 'metadata') and 'face_labels' in slice_mesh.metadata:
                        slice_predictions = slice_mesh.metadata['face_labels']
                    else:
                        # Try to read from PLY file directly
                        slice_predictions = read_face_labels_from_ply(predicted_slice_path)
                    
                    if slice_predictions is None:
                        print(f"  Warning: Could not extract labels from {slice_basename}")
                        continue
                    
                    # Map predictions back to original face indices
                    for slice_face_idx, orig_face_idx in enumerate(original_face_indices):
                        if slice_face_idx < len(slice_predictions):
                            all_face_predictions[orig_face_idx] += slice_predictions[slice_face_idx]
                            face_prediction_counts[orig_face_idx] += 1
                    
                    print(f"  Processed slice: {slice_basename} ({len(original_face_indices)} faces)")
                    
                except Exception as e:
                    print(f"  Error processing slice {slice_basename}: {e}")
                    continue
            
            # Average predictions for faces that appear in multiple slices (if any overlap)
            for i in range(original_num_faces):
                if face_prediction_counts[i] > 1:
                    all_face_predictions[i] = int(all_face_predictions[i] / face_prediction_counts[i] + 0.5)
            
            # Load original mesh and save with stitched predictions
            original_mesh_path_full = os.path.join(original_mesh_dir, os.path.basename(original_mesh_path))
            if not os.path.exists(original_mesh_path_full):
                original_mesh_path_full = original_mesh_path  # Try the path from mapping file
            
            if os.path.exists(original_mesh_path_full):
                output_path = os.path.join(output_dir, f"{original_basename}_stitched.ply")
                save_mesh_with_predictions(original_mesh_path_full, all_face_predictions, output_path)
                print(f"  Saved stitched mesh: {output_path}")
            else:
                print(f"  Warning: Original mesh not found: {original_mesh_path_full}")
                
        except Exception as e:
            print(f"Error processing mapping file {mapping_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Stitching complete!")
    print(f"Stitched meshes saved to: {output_dir}")
    print("="*60 + "\n")


def read_face_labels_from_ply(ply_path):
    """
    Read face labels directly from a PLY file.
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        numpy array of face labels, or None if not found
    """
    try:
        face_labels = []
        reading_faces = False
        face_count = 0
        
        with open(ply_path, 'r') as f:
            for line in f:
                if line.startswith('element face'):
                    face_count = int(line.split()[2])
                elif line.startswith('end_header'):
                    reading_faces = True
                    continue
                elif reading_faces and face_count > 0:
                    parts = line.strip().split()
                    # The label should be the last integer in the face line
                    # Format: "3 v0 v1 v2 ... label"
                    if len(parts) > 4:
                        try:
                            label = int(parts[-1])  # Last element is usually the label
                            face_labels.append(label)
                        except:
                            # If last element isn't an int, try to find 'label' property
                            pass
                    face_count -= 1
                    if face_count == 0:
                        break
        
        return np.array(face_labels) if face_labels else None
    except Exception as e:
        print(f"Error reading face labels from {ply_path}: {e}")
        return None


def read_cls_property_from_ply(ply_path):
    """
    Read the 'cls' property from a PLY file for each face.
    
    Based on the actual PLY format observed:
    Format: "3 v0 v1 v2 [texture_coords...] [cls nvert area_perc]"
    Example: "3 8949 8988 9806 6 0.48521339893341064 0.6747585535049438 0.48500359058380127 0.6745004653930664 0.48534053564071655 0.6744533181190491 1 1 3 0"
    
    The cls property is the 3rd from the end (parts[-3]).
    
    Args:
        ply_path: Path to PLY file
        
    Returns:
        numpy array of cls values for each face, or None if not found
    """
    try:
        cls_values = []
        reading_faces = False
        face_count = 0
        vertex_count = 0
        has_cls_property = False
        
        with open(ply_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Check if cls property is defined in header
                if 'property' in line and 'cls' in line:
                    has_cls_property = True
                
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[2])
                    
                elif line.startswith('element face'):
                    face_count = int(line.split()[2])
                    
                elif line.startswith('end_header'):
                    reading_faces = True
                    # Skip vertex lines
                    for _ in range(vertex_count):
                        next(f)
                    continue
                    
                elif reading_faces and face_count > 0:
                    parts = line.split()
                    if len(parts) >= 15:  # Ensure it's a valid face line with all properties
                        try:
                            # The cls property is the 3rd from the end
                            cls_value = int(parts[-3])
                            cls_values.append(cls_value)
                                
                        except (ValueError, IndexError) as e:
                            # Skip lines that can't be parsed
                            continue
                    
                    face_count -= 1
                    if face_count == 0:
                        break
        
        if not has_cls_property:
            print(f"No 'cls' property found in PLY header: {ply_path}")
            return None
            
        return np.array(cls_values) if cls_values else None
        
    except Exception as e:
        print(f"Error reading cls property from {ply_path}: {e}")
        return None


def build_model_from_config(config, device, use_texture=False):
    feature_dim = 16
    embedding_dim = config['model']['embedding_dim']
    num_heads = config['model']['num_heads']
    num_attention_blocks = config['model']['num_attention_blocks']
    N_class = config['model']['n_classes']
    dropout = config['model'].get('dropout', 0.0)
    use_hierarchical = config['model'].get('use_hierarchical', True)
    fourier = config['model'].get('fourier', False)
    relative_positional_encoding = config['model'].get('relative_positional_encoding', False)

    if use_texture:
        # Create integrated texture-geometry model
        texture_embed_dim = config['model'].get('texture_embed_dim', 64)
        fusion_method = config['model'].get('fusion_method', 'gated')
        max_texture_pixels = config['model'].get('max_texture_pixels', 128)
        
        encoder = IntegratedTextureGeometryModel(
            geometry_feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            texture_embed_dim=texture_embed_dim,
            num_heads=num_heads,
            num_attention_blocks=num_attention_blocks,
            dropout=dropout,
            summary_mode='cls',
            use_hierarchical=use_hierarchical,
            fourier=fourier,
            relative_positional_encoding=relative_positional_encoding,
            fusion_method=fusion_method,
            max_texture_pixels=max_texture_pixels
        )
        model = IntegratedDownstreamClassifier(
            integrated_encoder=encoder,
            num_classes=N_class,
            embedding_dim=embedding_dim,
            dropout=dropout,
            freeze_encoder_layers=0,
            fusion_method=fusion_method
        ).to(device)
        print("Created integrated texture-geometry model for inference")
    else:
        # Create geometry-only model
        encoder = nomeformer(
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_attention_blocks=num_attention_blocks,
            dropout=dropout,
            summary_mode='cls',
            use_hierarchical=use_hierarchical,
            num_hierarchical_stages=1,
            fourier=fourier,
            relative_positional_encoding=relative_positional_encoding,
        )
        model = DownstreamClassifier(encoder, N_class, embedding_dim, dropout, True, 0).to(device)
        print("Created geometry-only model for inference")
    
    return model


def predict(model, data_loader, num_classes, device, use_texture=False, output_dir=None, 
             save_meshes=True, test_mesh_dir=None, dataset=None):
    """
    Run inference without labels - just get predictions.
    
    Args:
        model: The trained model
        data_loader: DataLoader for the test data
        num_classes: Number of classes
        device: Device to run on
        use_texture: Whether using texture features
        output_dir: Optional directory to save predictions
        save_meshes: Whether to save meshes with predicted labels
        test_mesh_dir: Directory containing the original mesh files
        dataset: Dataset object to get mesh file information
        
    Returns:
        all_predictions: List of predictions for each mesh
    """
    model.eval()
    all_predictions = []
    mesh_predictions = {}  # mesh_file -> list of predictions
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            if use_texture:
                # Texture dataset returns: (geometry_features, labels, texture_sequences, masks, texture_masks, face_ids_tensor)
                geometry_features, labels, texture_sequences, masks, texture_masks, face_ids_tensor = data
                geometry_features = geometry_features.to(device)
                texture_sequences = texture_sequences.to(device)
                masks = masks.to(device)
                texture_masks = texture_masks.to(device)
                
                logits = model(geometry_features, texture_sequences, masks, texture_masks)
            else:
                # Geometry-only dataset returns: (batch, labels, masks)
                batch, labels, masks = data
                batch = batch.to(device)
                masks = masks.to(device)
                
                logits = model(batch, masks)
            
            # Get predictions
            pred = logits.reshape(-1, num_classes)
            mask_flat = masks.view(-1)
            valid_mask = mask_flat == 1
            
            # Only keep predictions for valid (non-padded) faces
            pred_valid = pred[valid_mask]
            pred_classes = pred_valid.argmax(dim=1)
            pred_np = pred_classes.cpu().numpy()
            
            all_predictions.append(pred_np)
            
            # Track predictions per mesh if saving meshes
            if save_meshes and dataset is not None:
                # Get the mesh file for this batch
                # The dataset index corresponds to cluster batches, need to map back to mesh
                mesh_idx = batch_idx // (dataset.n_clusters // dataset.clusters_per_batch) if hasattr(dataset, 'n_clusters') else batch_idx
                if mesh_idx < len(dataset.mesh_files):
                    mesh_file = dataset.mesh_files[mesh_idx]
                    if mesh_file not in mesh_predictions:
                        mesh_predictions[mesh_file] = {'predictions': [], 'face_ids': []}
                    
                    if use_texture:
                        # Use face_ids_tensor to map predictions back to original face indices
                        face_ids_flat = face_ids_tensor.view(-1).to(device)
                        face_ids_valid = face_ids_flat[valid_mask].cpu().numpy()
                        mesh_predictions[mesh_file]['predictions'].append(pred_np)
                        mesh_predictions[mesh_file]['face_ids'].append(face_ids_valid)
                    else:
                        # For geometry-only, use simple concatenation (legacy behavior)
                        if 'predictions' not in mesh_predictions[mesh_file]:
                            mesh_predictions[mesh_file] = []
                        mesh_predictions[mesh_file].append(pred_np)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")
    
    print(f"Prediction complete! Processed {len(all_predictions)} batches.")
    
    # Save meshes with predicted labels
    if save_meshes and test_mesh_dir is not None and mesh_predictions:
        mesh_output_dir = os.path.join(output_dir, "predicted_meshes") if output_dir else "predicted_meshes"
        os.makedirs(mesh_output_dir, exist_ok=True)
        
        print(f"\nSaving {len(mesh_predictions)} meshes with predicted labels...")
        for mesh_file, pred_data in mesh_predictions.items():
            # Handle both old format (list) and new format (dict with face_ids)
            if isinstance(pred_data, dict) and 'predictions' in pred_data:
                # New format with face_ids_tensor mapping
                pred_list = pred_data['predictions']
                face_ids_list = pred_data['face_ids']
                
                # Concatenate predictions and face IDs
                all_preds = np.concatenate(pred_list) if len(pred_list) > 1 else pred_list[0]
                all_face_ids = np.concatenate(face_ids_list) if len(face_ids_list) > 1 else face_ids_list[0]
                
                # Create properly ordered predictions array
                mesh_path = os.path.join(test_mesh_dir, mesh_file)
                mesh = trimesh.load(mesh_path, force='mesh')
                num_faces = mesh.faces.shape[0]
                
                # Initialize predictions array for all faces
                ordered_predictions = np.zeros(num_faces, dtype=np.int32)
                
                # Map predictions back to correct face indices
                for pred, face_id in zip(all_preds, all_face_ids):
                    if 0 <= face_id < num_faces:
                        ordered_predictions[face_id] = pred
                
                all_preds = ordered_predictions
            else:
                # Old format (geometry-only) - simple concatenation
                pred_list = pred_data
                all_preds = np.concatenate(pred_list) if len(pred_list) > 1 else pred_list[0]
            
            # Construct paths
            mesh_path = os.path.join(test_mesh_dir, mesh_file)
            base_name = os.path.splitext(mesh_file)[0]
            output_path = os.path.join(mesh_output_dir, f"{base_name}_predicted.ply")
            
            # Save mesh with predictions
            print(f"Saving {mesh_file}...")
            save_mesh_with_predictions(mesh_path, all_preds, output_path)
        
        print(f"Predicted meshes saved to {mesh_output_dir}")
    
    return all_predictions


def evaluate(model, data_loader, num_classes, ignore_index, device, use_texture=False):
    """
    Run evaluation with labels - compute metrics.
    """
    model.eval()
    cm = ConfusionMatrix(num_classes=num_classes)
    if ignore_index is not None:
        miou_metric = IoU(cm=cm, ignore_index=ignore_index)
        f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='none', ignore_index=ignore_index).to(device)
    else:
        miou_metric = IoU(cm=cm)
        f1_metric = F1Score(task='multiclass', num_classes=num_classes, average='none').to(device)

    f1_metric.reset()
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in data_loader:
            if use_texture:
                # Texture dataset returns: (geometry_features, labels, texture_sequences, masks, texture_masks, face_ids_tensor)
                geometry_features, labels, texture_sequences, masks, texture_masks, face_ids_tensor = data
                geometry_features = geometry_features.to(device)
                texture_sequences = texture_sequences.to(device)
                labels = labels.to(device) +1
                masks = masks.to(device)
                texture_masks = texture_masks.to(device)
                
                logits = model(geometry_features, texture_sequences, masks, texture_masks)
            else:
                # Geometry-only dataset returns: (batch, labels, masks)
                batch, labels, masks = data
                batch = batch.to(device)
                labels = labels.to(device) + 1
                masks = masks.to(device)
                
                logits = model(batch, masks)
            
            pred = logits.reshape(-1, num_classes)
            target = labels.reshape(-1).long()
            mask = masks.view(-1)
            valid_mask = mask == 1
            pred = pred[valid_mask]
            target = target[valid_mask]
            
            # ConfusionMatrix expects logits
            cm.update((pred, target))
            
            # Get predictions for overall accuracy
            pred_classes = pred.argmax(dim=1)
            f1_metric.update(pred_classes, target)
            
            # Collect all predictions and targets for overall accuracy
            all_preds.append(pred_classes)
            all_targets.append(target)

    # Compute metrics
    f1_scores = f1_metric.compute()
    if ignore_index is not None and ignore_index < num_classes:
        class_mask = torch.arange(num_classes, device=f1_scores.device) != ignore_index
        mean_f1 = f1_scores[class_mask].mean().item()
    else:
        mean_f1 = f1_scores.mean().item()
    
    # Compute overall accuracy using simple correct/total approach
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    overall_acc = overall_accuracy(all_preds, all_targets, ignore_index=ignore_index)
    
    miou = miou_metric.compute()

    return mean_f1, overall_acc, miou, f1_scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained downstream model on a test directory.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config containing evaluation paths')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    n_clusters = config['model']['n_clusters']
    clusters_per_batch = config['model'].get('clusters_per_batch', 300)
    PE = config['model'].get('use_pe', True)
    batch_size = config['model'].get('batch_size', 1)
    ignore_index = config['paths'].get('ignore_index', None)
    ignore_index = ignore_index + 1
    num_classes = config['model']['n_classes']
    include_normals = config['model'].get('include_normals', True)
    additional_geometrical_features = config['model'].get('additional_geometrical_features', False)
    
    # Texture-specific configuration
    use_texture = config['model'].get('use_texture', True)
    texture_patch_size = config['model'].get('texture_patch_size', 16)
    max_texture_pixels = config['model'].get('max_texture_pixels', 128)
    
    # Read evaluation paths from config
    test_mesh_dir = config['paths']['test_mesh_dir']
    test_label_dir = config['paths'].get('test_label_dir', None)
    test_json_dir = config['paths'].get('test_json_dir', None)
    test_texture_dir = config['paths'].get('test_texture_dir', None)
    checkpoint_path = config['paths']['checkpoint_path']
    output_dir = config['paths'].get('prediction_out_path', None)
    
    # Auto-slicing configuration
    auto_slice_config = config.get('auto_slice', {})
    auto_slice_enabled = auto_slice_config.get('enabled', True)
    
    # Process mesh slicing if enabled
    original_mesh_dir = test_mesh_dir  # Keep reference to original
    if auto_slice_enabled:
        # Determine output directory for sliced meshes
        slice_output_dir = auto_slice_config.get('output_dir', None)
        if slice_output_dir is None:
            # Default: create 'sliced_meshes' subdirectory next to original mesh directory
            parent_dir = os.path.dirname(test_mesh_dir.rstrip('/\\'))
            slice_output_dir = os.path.join(parent_dir, 'sliced_meshes')
        
        # Clear existing sliced meshes at the beginning of each inference run
        if os.path.exists(slice_output_dir) and len(os.listdir(slice_output_dir)) > 0:
            print("\n" + "="*60)
            print("Clearing existing sliced meshes before inference...")
            print("="*60)
            sliced_files = glob.glob(os.path.join(slice_output_dir, "*.ply"))
            for f in sliced_files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: Could not remove {f}: {e}")
            print(f"Removed {len(sliced_files)} sliced mesh files from {slice_output_dir}")
            print("="*60 + "\n")
        
        
        # Determine texture directories for slicing (optional)
        # Defaults: use test_texture_dir and write textures next to sliced meshes
        if 'texture_dir' not in auto_slice_config and test_texture_dir is not None:
            auto_slice_config['texture_dir'] = test_texture_dir
        if 'texture_output_dir' not in auto_slice_config:
            auto_slice_config['texture_output_dir'] = slice_output_dir

        # Process meshes with slicing
        test_mesh_dir = process_mesh_directory_with_slicing(
            mesh_dir=test_mesh_dir,
            slice_output_dir=slice_output_dir,
            auto_slice_config=auto_slice_config
        )
        
        print(f"\nUsing mesh directory for inference: {test_mesh_dir}")
    else:
        print("\nAuto-slicing is disabled. Using original meshes.")
    
    # Check if we should save meshes with predictions
    save_meshes = config['paths'].get('save_meshes', True)
    
    # Check if we have labels (evaluation mode) or not (prediction mode)
    # Labels can now come from PLY cls property or from separate label directory
    has_labels = config['paths'].get('has_labels', False)
    
    # If no label directory is provided, we'll try to read labels from PLY cls property
    if test_label_dir is None:
        print("No separate label directory provided - will attempt to read labels from PLY cls property")
        # We'll still try to load labels from PLY files if has_labels is True
    else:
        print(f"Label directory provided: {test_label_dir}")
        print("Will use separate label files instead of PLY cls property")
    
    if has_labels:
        print("Running in evaluation mode (with labels)")
    else:
        print("Running in prediction mode (without labels)")
        if save_meshes:
            print("Will save predicted meshes with labels")

    # Create dataset and dataloader based on whether texture is used
    if use_texture and test_texture_dir is not None:
        print(f"Using MeshTextureDataset with texture features from {test_texture_dir}")
        dataset = MeshTextureDataset(
            mesh_dir=test_mesh_dir,
            label_dir=test_label_dir if has_labels else None,
            texture_dir=test_texture_dir,
            n_clusters=n_clusters,
            clusters_per_batch=clusters_per_batch,
            PE=PE,
            json_dir=test_json_dir,
            augmentation=None,
            transform=None,
            include_normals=include_normals,
            additional_geometrical_features=additional_geometrical_features,
            texture_patch_size=texture_patch_size,
            max_texture_pixels=max_texture_pixels,
            pe_bbox_normalized=True,
            require_labels=has_labels
        )
        # Clear cached meshes at the beginning of inference
        print("\n" + "="*60)
        print("Clearing cached meshes before inference...")
        print("="*60)
        dataset.clear_cache()
        print("="*60 + "\n")
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=texture_custom_collate_fn)
    else:
        print(f"Using MeshDataset with geometry features only from {test_mesh_dir}")
        dataset = MeshDataset(
            mesh_dir=test_mesh_dir,
            label_dir=test_label_dir if has_labels else None,
            n_clusters=n_clusters,
            clusters_per_batch=clusters_per_batch,
            PE=PE,
            json_dir=test_json_dir,
            augmentation=None,
            transform=None,
            include_normals=include_normals,
            additional_geometrical_features=additional_geometrical_features,
            require_labels=has_labels
        )
        # Clear cached meshes at the beginning of inference
        print("\n" + "="*60)
        print("Clearing cached meshes before inference...")
        print("="*60)
        dataset.clear_cache()
        print("="*60 + "\n")
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Build model with texture support if needed
    model = build_model_from_config(config, device, use_texture=use_texture and test_texture_dir is not None)
    
    # Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    current_state = model.state_dict()
    checkpoint_state = ckpt['model_state_dict']
    total_param_tensors = len(current_state)
    matched = {k: v for k, v in checkpoint_state.items() if k in current_state and current_state[k].shape == v.shape}
    current_state.update(matched)
    model.load_state_dict(current_state, strict=False)
    
    # Report classifier restoration explicitly
    classifier_keys = [k for k in matched.keys() if k.startswith('classifier.')]
    print(f"Restored {len(classifier_keys)} classifier params; matched {len(matched)} / {total_param_tensors} total parameter tensors.")
    model = model.to(device)

    # Run inference or evaluation based on mode
    if has_labels:
        # Evaluation mode - compute metrics
        mean_f1, overall_acc, miou, f1_scores = evaluate(
            model, data_loader, num_classes, ignore_index, device, 
            use_texture=use_texture and test_texture_dir is not None
        )
        print(f"Evaluation Results:\n  Mean F1: {mean_f1:.4f}\n  Overall Accuracy: {overall_acc:.4f}\n  mIoU: {miou}")
        print(f"Per-class F1: {f1_scores}")
    else:
        # Prediction mode - just get predictions
        predictions = predict(
            model, data_loader, num_classes, device,
            use_texture=use_texture and test_texture_dir is not None,
            output_dir=output_dir,
            save_meshes=save_meshes,
            test_mesh_dir=test_mesh_dir,
            dataset=dataset
        )
        print(f"Generated predictions for {len(predictions)} batches")
        if output_dir:
            print(f"Predictions saved to: {output_dir}")
        
        # If auto-slicing was used, stitch predictions back to original meshes
        if auto_slice_enabled and save_meshes and output_dir:
            slice_predictions_dir = os.path.join(output_dir, "predicted_meshes")
            stitched_output_dir = os.path.join(output_dir, "stitched_meshes")
            
            stitch_sliced_predictions(
                slice_predictions_dir=slice_predictions_dir,
                original_mesh_dir=original_mesh_dir,
                output_dir=stitched_output_dir,
                sliced_mesh_dir=test_mesh_dir  # This is the sliced mesh directory
            )


if __name__ == '__main__':
    main()


