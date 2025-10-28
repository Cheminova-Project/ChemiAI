import numpy as np
import os
import argparse
import json

def plane_equation(plane_origin, plane_normal):
    A, B, C = plane_normal
    D = -np.dot(plane_normal, plane_origin)
    return A, B, C, D

def is_point_in_region(point, planes):
    """Checks if a point is inside all planes of the region."""
    return all(np.dot(plane[:3], point) + plane[3] >= 0 for plane in planes)

def points_to_planes(points):
    """Generates plane equations from the four corner points."""
    p1, p2, p3, p4 = points

    # Define planes based on the four points
    planes = []
    # Plane from p1, p2, p3
    planes.append(plane_equation(p1, np.cross(p2 - p1, p3 - p1)))
    # Plane from p1, p2, p4
    planes.append(plane_equation(p1, np.cross(p2 - p1, p4 - p1)))
    # Plane from p2, p3, p4
    planes.append(plane_equation(p2, np.cross(p3 - p2, p4 - p2)))
    # Plane from p1, p3, p4
    planes.append(plane_equation(p1, np.cross(p3 - p1, p4 - p1)))

    return planes

def generate_automatic_bounding_boxes(vertices, grid_divisions=(10, 10, 10), overlap=0.0):
    """
    Automatically generates bounding boxes based on mesh bounds.
    
    Args:
        vertices: Mesh vertices array
        grid_divisions: Tuple of (x_div, y_div, z_div) specifying divisions along each axis
        overlap: Percentage overlap between adjacent boxes (0.0 to 0.5)
    
    Returns:
        List of bounding boxes as (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    # Get the bounding box of the entire mesh
    coords = vertices[:, :3]
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    
    x_min_global, y_min_global, z_min_global = min_coords
    x_max_global, y_max_global, z_max_global = max_coords
    
    x_div, y_div, z_div = grid_divisions
    
    # Calculate step sizes for each dimension
    x_step = (x_max_global - x_min_global) / x_div
    y_step = (y_max_global - y_min_global) / y_div
    z_step = (z_max_global - z_min_global) / z_div
    
    # Calculate overlap in absolute units
    x_overlap = x_step * overlap
    y_overlap = y_step * overlap
    z_overlap = z_step * overlap
    
    bounding_boxes = []
    
    for i in range(x_div):
        for j in range(y_div):
            for k in range(z_div):
                x_min = x_min_global + i * x_step - x_overlap
                x_max = x_min_global + (i + 1) * x_step + x_overlap
                y_min = y_min_global + j * y_step - y_overlap
                y_max = y_min_global + (j + 1) * y_step + y_overlap
                z_min = z_min_global + k * z_step - z_overlap
                z_max = z_min_global + (k + 1) * z_step + z_overlap
                
                # Clamp to global bounds
                x_min = max(x_min, x_min_global)
                x_max = min(x_max, x_max_global)
                y_min = max(y_min, y_min_global)
                y_max = min(y_max, y_max_global)
                z_min = max(z_min, z_min_global)
                z_max = min(z_max, z_max_global)
                
                bounding_boxes.append((x_min, x_max, y_min, y_max, z_min, z_max))
    
    return bounding_boxes

def generate_adaptive_bounding_boxes(vertices, faces, target_faces_per_box=1000, min_boxes=1, max_boxes=100):
    """
    Generates bounding boxes adaptively based on face density.
    
    Args:
        vertices: Mesh vertices array
        faces: Mesh faces array
        target_faces_per_box: Approximate number of faces per bounding box
        min_boxes: Minimum number of boxes to generate (set to 1 to allow no slicing)
        max_boxes: Maximum number of boxes to generate
    
    Returns:
        List of bounding boxes as (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    total_faces = len(faces)
    
    # Calculate desired number of boxes based on face count
    desired_boxes = max(1, total_faces // target_faces_per_box)
    num_boxes = max(min_boxes, min(max_boxes, desired_boxes))
    
    # If only 1 box is needed, don't slice
    if num_boxes == 1:
        print(f"Mesh is small enough ({total_faces} faces < {target_faces_per_box} target). Creating 1 box (no slicing).")
        return generate_automatic_bounding_boxes(vertices, grid_divisions=(1, 1, 1), overlap=0.0)
    
    # Calculate grid divisions
    # Try to make it roughly cubic
    divisions_per_axis = int(np.ceil(num_boxes ** (1/3)))
    
    # Adjust to get closer to desired number of boxes
    grid_divisions = (divisions_per_axis, divisions_per_axis, divisions_per_axis)
    
    return generate_automatic_bounding_boxes(vertices, grid_divisions, overlap=0.0)

def remove_unused_vertices(vertices, faces):
    """
    Removes vertices that are not referenced by any face and remaps face indices.
    
    Args:
        vertices: Array of vertices
        faces: List of faces (each face contains vertex indices and properties)
    
    Returns:
        cleaned_vertices: Array of only used vertices
        cleaned_faces: List of faces with remapped vertex indices
    """
    # Find all vertex indices used in faces
    used_vertices = set()
    for face in faces:
        v0_idx, v1_idx, v2_idx = int(face[0]), int(face[1]), int(face[2])
        used_vertices.add(v0_idx)
        used_vertices.add(v1_idx)
        used_vertices.add(v2_idx)
    
    # Map old vertex indices to new ones
    new_index_map = {}
    cleaned_vertices = []
    
    for old_idx in sorted(used_vertices):
        new_index_map[old_idx] = len(cleaned_vertices)
        cleaned_vertices.append(vertices[old_idx])
    
    # Update face indices with new vertex indices
    cleaned_faces = []
    for face in faces:
        old_v0, old_v1, old_v2 = int(face[0]), int(face[1]), int(face[2])
        new_face = [
            new_index_map[old_v0],
            new_index_map[old_v1],
            new_index_map[old_v2]
        ] + list(face[3:])  # Keep the rest of the face properties
        cleaned_faces.append(new_face)
    
    return np.array(cleaned_vertices), cleaned_faces

def slice_mesh_with_bounding_boxes(vertices, faces, bounding_boxes):
    """Slices the mesh based on multiple bounding boxes and removes unused vertices."""
    all_sliced_meshes = []
    
    for bbox_index, bbox in enumerate(bounding_boxes):
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        sliced_faces = []
        
        for face_indices in faces:
            v0_idx, v1_idx, v2_idx = face_indices[:3]
            v0 = vertices[v0_idx, :3]
            v1 = vertices[v1_idx, :3]
            v2 = vertices[v2_idx, :3]

            # Check if all vertices are inside the bounding box
            if all(x_min <= v[0] <= x_max and y_min <= v[1] <= y_max and z_min <= v[2] <= z_max for v in [v0, v1, v2]):
                sliced_faces.append(face_indices)

        if sliced_faces:
            # Remove unused vertices from this slice
            cleaned_vertices, cleaned_faces = remove_unused_vertices(vertices, sliced_faces)
            all_sliced_meshes.append((cleaned_vertices, cleaned_faces, bbox))
    
    return all_sliced_meshes

def write_ply_file(file_path, vertices, faces):
    with open(file_path, 'w') as f:
        # Write the header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property int cls\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property list uchar float texcoord\n")
        f.write("property int texnumber\n")
        f.write("property short cls\n")
        f.write("property char nvert\n")
        f.write("property char area_perc\n")
        f.write("end_header\n")
        
        # Write vertices
        for vertex in vertices:
            vertex_str = " ".join(map(str, vertex[:6])) + f" {int(vertex[6])}\n"  # Ensuring cls is an integer
            f.write(vertex_str)
        
        # Write faces
        print('faces out', len(faces))
        for face in faces:
            vertex_indices = " ".join(map(str, face[:3]))
            properties = " ".join(map(str, face[3:]))
            f.write(f"3 {vertex_indices} {properties}\n")

def parse_ply_file(mesh_path):
    vertices = []
    faces = []
    reading_vertices = False
    reading_faces = False
    vertex_count = 0
    face_count = 0

    with open(mesh_path, 'r') as f:
        for line in f:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
                print('vertex_count', vertex_count)
            elif line.startswith('element face'):
                face_count = int(line.split()[2])
                print('face_count', face_count)
            elif line.startswith('end_header'):
                reading_vertices = True
                continue
            elif reading_vertices and vertex_count > 0:
                vertex_data = list(map(float, line.strip().split()[:6])) + [int(line.strip().split()[6])]
                vertices.append(vertex_data)
                vertex_count -= 1
                if vertex_count == 0:
                    reading_vertices = False
                    reading_faces = True
            elif reading_faces and face_count > 0:
                # First three elements should be integers (vertex indices)
                face_indices = list(map(int, line.strip().split()[1:5]))
                # The rest are floating-point face properties (e.g., texture coordinates, normals)
                face_properties = list(map(float, line.strip().split()[5:11]))
                face_properties_2 = list(map(int, line.strip().split()[11:]))
                faces.append(face_indices + face_properties + face_properties_2)  # Concatenate indices and properties
                face_count -= 1
                if face_count == 0:
                    break

    return np.array(vertices), np.array(faces, dtype=object)


def parse_bounding_boxes_spec(spec, vertices, faces):
    """
    Parse the --bounding_boxes specification into a list of boxes.

    Supported formats:
    - '@/path/to/boxes.json' -> JSON file containing [[xmin,xmax,ymin,ymax,zmin,zmax], ...]
    - 'grid:X,Y,Z' or 'grid:X,Y,Z,OVERLAP' -> integers (divisions per axis), optional float overlap [0..0.5]
    - 'adaptive:TARGET[,MIN[,MAX]]' -> integers; defaults MIN=1, MAX=100
    - Manual list: 'xmin,xmax,ymin,ymax,zmin,zmax[;|]...'
    """
    spec = spec.strip()
    if not spec:
        return None

    if spec.startswith('@'):
        path = spec[1:]
        with open(path, 'r') as f:
            data = json.load(f)
        boxes = []
        for item in data:
            if len(item) != 6:
                raise ValueError("Each bounding box in JSON must have 6 numbers")
            boxes.append(tuple(float(v) for v in item))
        return boxes

    if spec.lower().startswith('grid:'):
        parts = spec.split(':', 1)[1].split(',')
        if len(parts) not in (3, 4):
            raise ValueError("grid spec must be 'grid:X,Y,Z' or 'grid:X,Y,Z,OVERLAP'")
        x_div, y_div, z_div = (int(parts[0]), int(parts[1]), int(parts[2]))
        overlap = float(parts[3]) if len(parts) == 4 else 0.0
        return generate_automatic_bounding_boxes(vertices, (x_div, y_div, z_div), overlap=overlap)

    if spec.lower().startswith('adaptive:'):
        parts = spec.split(':', 1)[1].split(',')
        if len(parts) < 1:
            raise ValueError("adaptive spec must be 'adaptive:TARGET[,MIN[,MAX]]'")
        target = int(parts[0])
        min_boxes = int(parts[1]) if len(parts) >= 2 else 1
        max_boxes = int(parts[2]) if len(parts) >= 3 else 100
        return generate_adaptive_bounding_boxes(vertices, faces, target_faces_per_box=target, min_boxes=min_boxes, max_boxes=max_boxes)

    # Manual list parsing; allow ';' or '|' as separator between boxes
    box_separators = [';', '|']
    tokens = [spec]
    for sep in box_separators:
        if sep in spec:
            tokens = [t for t in spec.split(sep) if t.strip()]
            break
    boxes = []
    for tok in tokens:
        nums = [n for n in tok.replace(' ', '').split(',') if n]
        if len(nums) != 6:
            raise ValueError("Each manual bounding box must have exactly 6 comma-separated numbers")
        boxes.append(tuple(float(n) for n in nums))
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Slice a PLY mesh into sub-meshes using bounding boxes.")
    parser.add_argument('--mesh_directory', required=True, help='Path to input mesh .ply file')
    parser.add_argument('--output_directory', required=True, help='Directory to write output .ply slices')
    parser.add_argument('--bounding_boxes', required=False, default='grid:3,3,1',
                        help=(
                            "Bounding box spec. Formats: "
                            "'@/path/boxes.json' | "
                            "'grid:X,Y,Z[,OVERLAP]' | "
                            "'adaptive:TARGET[,MIN[,MAX]]' | "
                            "'xmin,xmax,ymin,ymax,zmin,zmax[;|]...'. "
                            "Default: grid:3,3,1"
                        ))
    args = parser.parse_args()

    mesh_path = args.mesh_directory
    output_directory_path = args.output_directory

    os.makedirs(output_directory_path, exist_ok=True)

    print("Parsing PLY file...")
    vertices, faces = parse_ply_file(mesh_path)
    print(f"Loaded {len(vertices)} vertices and {len(faces)} faces")

    # Build bounding boxes per spec
    print("\nParsing bounding boxes spec...")
    bounding_boxes = parse_bounding_boxes_spec(args.bounding_boxes, vertices, faces)
    if bounding_boxes is None:
        # Fallback to a single global box
        print("No bounding boxes provided. Using a single global bounding box.")
        bounding_boxes = generate_automatic_bounding_boxes(vertices, grid_divisions=(1, 1, 1), overlap=0.0)

    print(f"Generated {len(bounding_boxes)} bounding boxes")
    print("\nBounding boxes:")
    for i, bbox in enumerate(bounding_boxes):
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        print(f"  Box {i}: X[{x_min:.2f}, {x_max:.2f}] Y[{y_min:.2f}, {y_max:.2f}] Z[{z_min:.2f}, {z_max:.2f}]")

    print("\nSlicing mesh...")
    split_meshes = slice_mesh_with_bounding_boxes(vertices, faces, bounding_boxes)
    print(f"Created {len(split_meshes)} mesh slices")

    print("\nWriting output files...")
    total_original_vertices = len(vertices)
    for i, (slice_vertices, slice_faces, bbox) in enumerate(split_meshes):
        output_ply_file_path = os.path.join(output_directory_path, f'split_mesh_part_{i}.ply')
        write_ply_file(output_ply_file_path, slice_vertices, slice_faces)
        reduction_percent = 100.0 * (1 - len(slice_vertices) / total_original_vertices)
        print(f"  Written: {output_ply_file_path}")
        print(f"    Faces: {len(slice_faces)}, Vertices: {len(slice_vertices)} (reduced by {reduction_percent:.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()



