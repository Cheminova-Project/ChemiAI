import os
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F
import trimesh.transformations as tf
import json
import pickle
import tempfile
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from tools.mesh_dataset import MeshAugmentation, compute_cluster_centroids, reorder_clusters_by_proximity

# Import the cls property reader function
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

def get_texture_base_name(mesh_base_name):
    """
    Convert mesh base name to texture base name by removing _labeled suffix if present.
    
    Args:
        mesh_base_name (str): Base name of mesh file (without extension)
        
    Returns:
        texture_base_name (str): Base name for texture file
    """
    if mesh_base_name.endswith('_labeled'):
        return mesh_base_name[:-8]  # Remove '_labeled'
    return mesh_base_name

class MeshTextureDataset(Dataset):
    """
    PyTorch Dataset for loading 3D mesh data with texture information.
    
    Texture data format:
    - .pkl files with same name as mesh files
    - Each .pkl contains a list where each element corresponds to a face
    - Each face element is: (num_pixels, [[R, G, B], [R, G, B], ...])
    - Example: (18, [[44, 33, 29], [39, 28, 24], [45, 34, 30], ...])
    
    Args:
        mesh_dir (str): Directory with mesh files (.obj or .ply).
        label_dir (str): Directory with label files (.txt).
        texture_dir (str): Directory with texture files (.pkl).
        n_clusters (int): Number of clusters for K-Means.
        clusters_per_batch (int): Number of clusters per batch.
        PE (bool): Whether to use positional encoding.
        json_dir (str, optional): Directory with JSON feature files.
        augmentation (callable, optional): Augmentation function to apply to meshes.
        transform (callable, optional): Optional transform to apply to samples.
        include_normals (bool): Whether to include face normals.
        additional_geometrical_features (bool): Whether to include additional geometrical features.
        texture_patch_size (int): Size to resize texture patches to (square patches).
        max_texture_pixels (int): Maximum number of pixels per face (for padding/truncation).
        coords_use_cluster_center (bool): Whether to use cluster center for coordinate computation (default: True).
        pe_bbox_normalized (bool): Whether to use bbox-normalized coordinates instead of relative coordinates (default: False).
    """
    def __init__(self, 
                 mesh_dir, 
                 label_dir, 
                 texture_dir,
                 n_clusters, 
                 clusters_per_batch, 
                 PE, 
                 json_dir=None, 
                 augmentation=None, 
                 transform=None, 
                 include_normals=True, 
                 additional_geometrical_features=False,
                 texture_patch_size=16,
                 max_texture_pixels=32,
                 max_texture_pixels_per_face=None,
                 coords_use_cluster_center=True,
                 pe_bbox_normalized=False,
                 require_labels=True):
        
        self.mesh_dir = mesh_dir
        self.label_dir = label_dir
        self.texture_dir = texture_dir
        self.json_dir = json_dir
        self.mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj') or f.endswith('.ply')]
        self.n_clusters = n_clusters
        self.clusters_per_batch = clusters_per_batch
        self.transform = transform
        self.PE = PE
        self.augmentation = augmentation
        self.include_normals = include_normals
        self.AGF = additional_geometrical_features
        self.texture_patch_size = texture_patch_size
        self.max_texture_pixels = max_texture_pixels
        self.coords_use_cluster_center = coords_use_cluster_center
        self.pe_bbox_normalized = pe_bbox_normalized
        self.require_labels = require_labels

        # Caching setup
        self.cache_dir = os.path.join(mesh_dir, ".cluster_texture_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._in_memory_cache = {}  # mesh_path -> payload

        # Verify directories exist
        if require_labels and label_dir is not None and not os.path.isdir(label_dir):
            raise FileNotFoundError(f"The label directory {label_dir} does not exist.")
        if not os.path.isdir(texture_dir):
            raise FileNotFoundError(f"The texture directory {texture_dir} does not exist.")
        if json_dir and not os.path.isdir(json_dir):
            raise FileNotFoundError(f"The json directory {json_dir} does not exist.")

        # Load labels conditionally
        if require_labels:
            self.labels = self.load_labels()
            print(f"Found {len(self.mesh_files)} mesh files")
            print(f"Found {len(self.labels)} label files")
        else:
            self.labels = {}
            print(f"Skipping label loading (prediction mode)")
            print(f"Found {len(self.mesh_files)} mesh files")
            print(f"Found 0 label files")
        
        # Check texture files
        self.texture_files = [f for f in os.listdir(texture_dir) if f.endswith('.pkl')]
        print(f"Found {len(self.texture_files)} texture files")
        
        # Check how many mesh files have corresponding texture files
        matched_textures = 0
        for mesh_file in self.mesh_files:
            base_name = os.path.splitext(mesh_file)[0]
            texture_base_name = get_texture_base_name(base_name)
            
            # Check in mesh directory first (for sliced meshes)
            texture_file_path = os.path.join(self.mesh_dir, texture_base_name + '.pkl')
            texture_file_path_alt = os.path.join(self.mesh_dir, texture_base_name + '_pixels_test.pkl')
            
            # If not found in mesh directory, check texture directory
            if not (os.path.isfile(texture_file_path) or os.path.isfile(texture_file_path_alt)):
                texture_file_path = os.path.join(texture_dir, texture_base_name + '.pkl')
                texture_file_path_alt = os.path.join(texture_dir, texture_base_name + '_pixels_test.pkl')
            
            if os.path.isfile(texture_file_path) or os.path.isfile(texture_file_path_alt):
                matched_textures += 1
        print(f"Found {matched_textures} mesh files with matching texture files")

    def load_labels(self):
        """
        Loads the labels for each mesh file.
        If label_dir is provided, loads from separate .txt files.
        Otherwise, tries to load from cls property in PLY files.
        """
        if not self.require_labels:
            print("Skipping label loading (prediction mode)")
            return {}
            
        labels = {}
        
        # If label_dir is provided, use separate .txt files (original behavior)
        if self.label_dir is not None and os.path.exists(self.label_dir):
            print(f"Loading labels from separate label files in {self.label_dir}...")
            for file_name in os.listdir(self.label_dir):
                if file_name.endswith('.txt'):
                    base_name = os.path.splitext(file_name)[0]
                    with open(os.path.join(self.label_dir, file_name), 'r') as f:
                        labels[base_name] = [int(x) for x in f.read().split()]
            print(f"Loaded labels for {len(labels)} files from separate label files.")
        else:
            # Try to load from cls property in PLY files
            print(f"Loading labels from cls property in PLY files...")
            
            for mesh_file in self.mesh_files:
                if mesh_file.endswith('.ply'):
                    mesh_path = os.path.join(self.mesh_dir, mesh_file)
                    base_name = os.path.splitext(mesh_file)[0]
                    
                    # Try to read cls property from PLY file
                    cls_values = read_cls_property_from_ply(mesh_path)
                    
                    if cls_values is not None:
                        labels[base_name] = cls_values.tolist()
                        print(f"Loaded {len(cls_values)} labels from {mesh_file}")
                    else:
                        print(f"Warning: Could not load cls property from {mesh_file}")
                        
            print(f"Loaded labels for {len(labels)} files from PLY cls properties.")
        
        return labels

    def clear_cache(self):
        """Clear both in-memory and disk caches for the dataset."""
        # Clear in-memory cache
        self._in_memory_cache.clear()
        print(f"[CACHE] Cleared in-memory cache")
        
        # Clear disk cache
        if os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            for cache_file in cache_files:
                try:
                    os.remove(os.path.join(self.cache_dir, cache_file))
                except Exception as e:
                    print(f"[CACHE] Warning: Could not remove {cache_file}: {e}")
            print(f"[CACHE] Cleared {len(cache_files)} disk cache files from {self.cache_dir}")

    def load_texture_data(self, base_name):
        """
        Loads texture data from .pkl file for the given mesh.
        
        Args:
            base_name (str): Base name of the mesh file (without extension).
                            Handles both regular names and names with _labeled suffix.
            
        Returns:
            texture_data (list): List of texture sequences for each face.
            texture_masks (list): List of validity masks for each face.
        """
        # Convert mesh base name to texture base name (remove _labeled if present)
        texture_base_name = get_texture_base_name(base_name)
        
        # First, try to find texture file in the same directory as the mesh files (for sliced meshes)
        texture_file_path = os.path.join(self.mesh_dir, texture_base_name + '.pkl')
        print(f"Loading texture data from {texture_file_path}")
        
        # If not found in mesh directory, try the original texture directory
        if not os.path.isfile(texture_file_path):
            texture_file_path = os.path.join(self.texture_dir, texture_base_name + '.pkl')
            print(f"Texture not found in mesh directory, trying {texture_file_path}")
        
        # If the standard texture file doesn't exist, try with _pixels_test suffix
        if not os.path.isfile(texture_file_path):
            # Try _pixels_test suffix in mesh directory first
            texture_file_path_alt = os.path.join(self.mesh_dir, texture_base_name + '_pixels_test.pkl')
            if os.path.isfile(texture_file_path_alt):
                texture_file_path = texture_file_path_alt
            else:
                # Try _pixels_test suffix in texture directory
                texture_file_path_alt = os.path.join(self.texture_dir, texture_base_name + '_pixels_test.pkl')
                if os.path.isfile(texture_file_path_alt):
                    texture_file_path = texture_file_path_alt
                else:
                    print(f"Warning: No texture file found for {base_name} (tried {texture_base_name}.pkl and {texture_base_name}_pixels_test.pkl in both mesh and texture directories)")
                    return None, None
            
        try:
            with open(texture_file_path, 'rb') as f:
                texture_data = pickle.load(f)
            
            if not isinstance(texture_data, list):
                print(f"Warning: Texture data for {base_name} is not a list")
                return None, None
            
            # Process texture data - keep as sequences, don't reshape to 2D
            processed_textures = []
            texture_masks = []
            
            for face_idx, face_texture in enumerate(texture_data):
                # Try to extract pixel data from various possible formats
                pixels = None
                num_pixels = 0
                
                if isinstance(face_texture, tuple) and len(face_texture) == 2:
                    # Standard format: (num_pixels, pixel_list)
                    num_pixels, pixel_list = face_texture
                    pixels = np.array(pixel_list, dtype=np.uint8)
                elif isinstance(face_texture, tuple) and len(face_texture) == 1:
                    # Alternative format: (pixel_list,) - assume it's just the pixel list
                    pixel_list = face_texture[0]
                    pixels = np.array(pixel_list, dtype=np.uint8)
                    num_pixels = len(pixels)
                elif isinstance(face_texture, list):
                    # Direct list format: just pixel_list
                    pixels = np.array(face_texture, dtype=np.uint8)
                    num_pixels = len(pixels)
                elif isinstance(face_texture, np.ndarray):
                    # NumPy array format - handle directly
                    pixels = face_texture.astype(np.uint8)
                    num_pixels = len(pixels)
                elif face_texture is None:
                    # None value - treat as empty
                    pixels = None
                    num_pixels = 0
                else:
                    # Unknown format - try to extract data if possible
                    if hasattr(face_texture, '__iter__') and not isinstance(face_texture, str):
                        try:
                            # Try to convert to numpy array
                            pixels = np.array(face_texture, dtype=np.uint8)
                            num_pixels = len(pixels)
                            print(f"Debug: Converted unknown format {type(face_texture)} to numpy array for face {face_idx} in {base_name}")
                        except Exception as e:
                            pixels = None
                            num_pixels = 0
                            print(f"Debug: Failed to convert {type(face_texture)} for face {face_idx} in {base_name}: {e}")
                    else:
                        pixels = None
                        num_pixels = 0
                        print(f"Debug: Unhandled texture format {type(face_texture)} for face {face_idx} in {base_name}")
                
                # Process the extracted pixel data
                if pixels is None or num_pixels == 0:
                    # Empty or invalid texture - create minimal zero sequence
                    processed_textures.append(np.zeros((1, 3), dtype=np.uint8))  # Minimal size, collate will pad
                    texture_masks.append(False)
                    continue
                
                try:
                    
                    # Validate pixel data format
                    if len(pixels.shape) == 1:
                        # 1D array - might be flattened RGB data
                        if len(pixels) % 3 == 0:
                            # Reshape to (num_pixels, 3)
                            pixels = pixels.reshape(-1, 3)
                            num_pixels = len(pixels)
                        else:
                            print(f"Warning: Invalid 1D pixel data length for face {face_idx} in {base_name}")
                            processed_textures.append(np.zeros((1, 3), dtype=np.uint8))
                            texture_masks.append(False)
                            continue
                    elif len(pixels.shape) == 2 and pixels.shape[1] == 3:
                        # 2D array with RGB channels - this is what we expect
                        num_pixels = len(pixels)
                    else:
                        print(f"Warning: Invalid pixel data shape {pixels.shape} for face {face_idx} in {base_name}")
                        processed_textures.append(np.zeros((1, 3), dtype=np.uint8))
                        texture_masks.append(False)
                        continue
                    
                    # Truncate if too long (but let collate function handle padding)
                    if self.max_texture_pixels is not None and num_pixels > self.max_texture_pixels:
                        print(f"Truncating {num_pixels} pixels to {self.max_texture_pixels} pixels for face {face_idx} in {base_name}")
                        pixels = pixels[:self.max_texture_pixels]
                        num_pixels = self.max_texture_pixels
                        
                    processed_textures.append(pixels)
                    texture_masks.append(True)
                    
                except Exception as e:
                    print(f"Warning: Error processing pixel data for face {face_idx} in {base_name}: {e}")
                    processed_textures.append(np.zeros((1, 3), dtype=np.uint8))
                    texture_masks.append(False)
            
            return processed_textures, texture_masks
            
        except Exception as e:
            print(f"Error loading texture data for {base_name}: {e}")
            return None, None

    def _resize_texture_patch(self, patch, target_size):
        """
        Simple nearest neighbor resize for texture patches.
        """
        if patch.shape[:2] == (target_size, target_size):
            return patch
        
        # Use simple interpolation
        from scipy.ndimage import zoom
        zoom_factors = (target_size / patch.shape[0], target_size / patch.shape[1], 1)
        resized = zoom(patch, zoom_factors, order=0)  # order=0 for nearest neighbor
        return resized.astype(np.uint8)

    def load_json_features(self, base_name):
        """Loads additional features from the JSON file for the given mesh."""
        if self.json_dir:
            json_file_path = os.path.join(self.json_dir, base_name + '.json')
            if os.path.isfile(json_file_path):
                with open(json_file_path, 'r') as f:
                    json_features = json.load(f)
                return json_features
        return None

    def _compute_and_cache(self, mesh_path, label_file_name):
        """Computes and caches mesh data with texture information."""
        mesh = trimesh.load(mesh_path, force='mesh')
        vertices = mesh.vertices
        faces = mesh.faces

        # Normalize coordinates to the range [0, 1]
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        normalized_vertices = (vertices - min_coords) / (max_coords - min_coords)
        if self.require_labels:
            if self.require_labels:
                face_labels = self.labels[label_file_name]
                if len(face_labels) != len(faces):
                    raise ValueError(f"Number of labels {len(face_labels)} does not match number of faces {len(faces)} for {mesh_path}")
            else:
                # In prediction mode, create dummy labels
                face_labels = [0] * len(faces)
        else:
            # In prediction mode, create dummy labels
            face_labels = [0] * len(faces)

        # Remove duplicate faces
        seen = set()
        unique_faces = []
        unique_face_labels = []
        unique_face_indices = []
        for face_idx, face in enumerate(faces):
            face_tuple = tuple(face)
            if face_tuple not in seen:
                seen.add(face_tuple)
                unique_faces.append(face)
                unique_face_labels.append(face_labels[face_idx])
                unique_face_indices.append(face_idx)
        faces = np.array(unique_faces)
        face_labels = unique_face_labels
        unique_faces = np.array(unique_faces)

        # KMeans clustering on vertices
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(vertices)

        # Compute cluster centroids and reorder for consistency
        centroids = compute_cluster_centroids(vertices, labels, self.n_clusters)
        reorder_indices = reorder_clusters_by_proximity(centroids)
        centroids_reordered = centroids[reorder_indices]

        # Reorder clusters based on proximity and assign faces to clusters
        clusters = [[] for _ in range(self.n_clusters)]
        face_cluster_indices = []
        for face in faces:
            face_cluster = labels[face]
            most_common_cluster = np.bincount(face_cluster).argmax()
            reordered_cluster = reorder_indices[most_common_cluster]
            clusters[reordered_cluster].append(face)
            face_cluster_indices.append(reordered_cluster)

        mesh_normals = mesh.face_normals[unique_face_indices] if self.include_normals else None
        mesh_area = mesh.area_faces[unique_face_indices]
        mesh_angle = mesh.face_angles[unique_face_indices]
        min_angle = mesh_angle.min()
        max_angle = mesh_angle.max()
        mesh_angle = (mesh_angle - min_angle) / (max_angle - min_angle)
        min_area = mesh_area.min()
        max_area = mesh_area.max()
        mesh_area = (mesh_area - min_area) / (max_area - min_area)

        # Precompute coords for PE
        if self.pe_bbox_normalized:
            precomputed_coords = [normalized_vertices[face].flatten().tolist() for face in faces]
        else:
            # Relative to either cluster center or mesh center
            precomputed_coords = []
            if self.coords_use_cluster_center:
                for i, face in enumerate(faces):
                    cluster_idx = face_cluster_indices[i]
                    center = centroids_reordered[cluster_idx]
                    rel = (vertices[face] - center).flatten().tolist()
                    precomputed_coords.append(rel)
            else:
                print("Using mesh center")
                mesh_center = mesh.centroid if hasattr(mesh, 'centroid') else vertices.mean(axis=0)
                for _, face in enumerate(faces):
                    rel = (vertices[face] - mesh_center).flatten().tolist()
                    precomputed_coords.append(rel)
        face_to_index = {tuple(face): i for i, face in enumerate(faces)}

        # Additional geometrical features (slope, height, roughness)
        add_slope = None
        add_height = None
        add_roughness = None
        if getattr(self, 'AGF', False):
            face_centroids = vertices[faces].mean(axis=1)
            if self.include_normals and mesh_normals is not None:
                nz = np.clip(np.abs(mesh_normals[:, 2]), 0.0, 1.0)
            else:
                nz = np.clip(np.abs(mesh.face_normals[unique_face_indices][:, 2]), 0.0, 1.0)
            slope = np.arccos(nz)
            z_vals = vertices[:, 2]
            z_thresh = np.percentile(z_vals, 5.0)
            ground_pts = vertices[z_vals <= z_thresh]
            if ground_pts.shape[0] >= 3:
                gp_mean = ground_pts.mean(axis=0)
                gp_centered = ground_pts - gp_mean
                cov = gp_centered.T @ gp_centered / max(ground_pts.shape[0] - 1, 1)
                evals, evecs = np.linalg.eigh(cov)
                n = evecs[:, np.argmin(evals)]
                if n[2] < 0:
                    n = -n
                n = n / (np.linalg.norm(n) + 1e-12)
                d = -np.dot(n, gp_mean)
                norm_height = face_centroids @ n + d
            else:
                z_min = z_vals.min()
                norm_height = face_centroids[:, 2] - z_min
            bbox_diag = float(np.linalg.norm(max_coords - min_coords))
            radius = 0.05 * bbox_diag if bbox_diag > 0 else 0.05
            tree = cKDTree(face_centroids)
            rough = np.zeros(len(face_centroids), dtype=np.float32)
            for i in range(len(face_centroids)):
                neigh = tree.query_ball_point(face_centroids[i], r=radius)
                if len(neigh) <= 1:
                    rough[i] = 0.0
                else:
                    rough[i] = float(np.std(norm_height[neigh]))
            add_slope = slope
            add_height = norm_height
            add_roughness = rough

        # Load texture data
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]
        texture_data, texture_masks = self.load_texture_data(base_name)
        
        # If no texture data, create dummy data
        if texture_data is None:
            texture_data = [np.zeros((1, 3), dtype=np.uint8) for _ in range(len(faces))]
            texture_masks = [False] * len(faces)

        # Cache the data
        cache_data = {
            'clusters': clusters,
            'mesh_angle': mesh_angle,
            'mesh_normals': mesh_normals,
            'mesh_area': mesh_area,
            'precomputed_coords': precomputed_coords,
            'face_to_index': face_to_index,
            'face_labels': face_labels,
            'add_slope': add_slope,
            'add_height': add_height,
            'add_roughness': add_roughness,
            'texture_data': texture_data,
            'texture_masks': texture_masks
        }

        # Save to cache file
        cache_file = os.path.join(
            self.cache_dir,
            f"{os.path.basename(mesh_path)}__clusters{self.n_clusters}_PE{int(self.PE)}_NORM{int(self.include_normals)}_AGF{int(self.AGF)}_CC{int(self.coords_use_cluster_center)}_PEBN{int(self.pe_bbox_normalized)}_TEX{self.texture_patch_size}_MAXPIX{self.max_texture_pixels}.pkl"
        )
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        # Store in in-memory cache
        self._in_memory_cache[mesh_path] = cache_data

        return cache_data

    def __len__(self):
        return len(self.mesh_files) * (self.n_clusters // self.clusters_per_batch)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mesh_idx = idx // (self.n_clusters // self.clusters_per_batch)
        cluster_batch_idx = idx % (self.n_clusters // self.clusters_per_batch)
        mesh_file = self.mesh_files[mesh_idx]
        mesh_path = os.path.join(self.mesh_dir, mesh_file)
        label_file_name = os.path.splitext(mesh_file)[0]
        cache_file = os.path.join(
            self.cache_dir,
            f"{os.path.basename(mesh_path)}__clusters{self.n_clusters}_PE{int(self.PE)}_NORM{int(self.include_normals)}_AGF{int(self.AGF)}_CC{int(self.coords_use_cluster_center)}_PEBN{int(self.pe_bbox_normalized)}_TEX{self.texture_patch_size}_MAXPIX{self.max_texture_pixels}.pkl"
        )
        use_cache = self.augmentation is None  # Only cache un-augmented
        if use_cache:
            if mesh_path not in self._in_memory_cache:
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            payload = pickle.load(f)
                        if not isinstance(payload, dict):
                            raise ValueError(f"cache payload is {type(payload)}")
                        self._in_memory_cache[mesh_path] = payload
                        print(f"[CACHE] Loaded mesh {os.path.basename(mesh_path)} from cache.")
                    except (EOFError, pickle.UnpicklingError, ValueError):
                        os.remove(cache_file)
                        self._compute_and_cache(mesh_path, label_file_name)
                        print(f"[CACHE] Cache for {os.path.basename(mesh_path)} was corrupted or invalid. Recomputed and cached.")
                else:
                    self._compute_and_cache(mesh_path, label_file_name)
                    print(f"[CACHE] No cache found for {os.path.basename(mesh_path)}. Computed and cached.")
            else:
                print(f"[CACHE] Used in-memory cache for {os.path.basename(mesh_path)}.")
            cache = self._in_memory_cache[mesh_path]
            clusters = cache['clusters']
            mesh_angle = cache['mesh_angle']
            mesh_normals = cache.get('mesh_normals', None)
            mesh_area = cache['mesh_area']
            precomputed_coords = cache['precomputed_coords']
            face_to_index = cache['face_to_index']
            face_labels = cache['face_labels']
            add_slope = cache.get('add_slope', None)
            add_height = cache.get('add_height', None)
            add_roughness = cache.get('add_roughness', None)
            texture_data = cache['texture_data']
            texture_masks = cache['texture_masks']
        else:
            # If augmentation is used, recompute on the fly
            mesh = trimesh.load(mesh_path, force='mesh')
            if self.augmentation and np.random.rand() < 0.2:
                mesh = self.augmentation(mesh)
            print(f"[CACHE] Augmentation enabled or requested. Mesh {os.path.basename(mesh_path)} computed on the fly (not cached).")
            vertices = mesh.vertices
            faces = mesh.faces
            min_coords = vertices.min(axis=0)
            max_coords = vertices.max(axis=0)
            normalized_vertices = (vertices - min_coords) / (max_coords - min_coords)
            if self.require_labels:
                face_labels = self.labels[label_file_name]
                if len(face_labels) != len(faces):
                    raise ValueError(f"Number of labels {len(face_labels)} does not match number of faces {len(faces)} for {mesh_path}")
            else:
                # In prediction mode, create dummy labels
                face_labels = [0] * len(faces)

            # Remove duplicate faces
            seen = set()
            unique_faces = []
            unique_face_labels = []
            unique_face_indices = []
            for face_idx, face in enumerate(faces):
                face_tuple = tuple(face)
                if face_tuple not in seen:
                    seen.add(face_tuple)
                    unique_faces.append(face)
                    unique_face_labels.append(face_labels[face_idx])
                    unique_face_indices.append(face_idx)
            faces = np.array(unique_faces)
            face_labels = unique_face_labels
            unique_faces = np.array(unique_faces)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(vertices)

            # Compute cluster centroids and reorder for consistency
            centroids = compute_cluster_centroids(vertices, labels, self.n_clusters)
            reorder_indices = reorder_clusters_by_proximity(centroids)
            centroids_reordered = centroids[reorder_indices]

            # Reorder clusters based on proximity and assign faces to clusters
            clusters = [[] for _ in range(self.n_clusters)]
            face_cluster_indices = []
            for face in faces:
                face_cluster = labels[face]
                most_common_cluster = np.bincount(face_cluster).argmax()
                reordered_cluster = reorder_indices[most_common_cluster]
                clusters[reordered_cluster].append(face)
                face_cluster_indices.append(reordered_cluster)
            mesh_normals = mesh.face_normals[unique_face_indices] if self.include_normals else None
            mesh_area = mesh.area_faces[unique_face_indices]
            mesh_angle = mesh.face_angles[unique_face_indices]
            min_angle = mesh_angle.min()
            max_angle = mesh_angle.max()
            mesh_angle = (mesh_angle - min_angle) / (max_angle - min_angle)
            min_area = mesh_area.min()
            max_area = mesh_area.max()
            mesh_area = (mesh_area - min_area) / (max_area - min_area)
            # Precompute coords for PE
            if self.pe_bbox_normalized:
                precomputed_coords = [normalized_vertices[face].flatten().tolist() for face in faces]
            else:
                # Relative to either cluster center or mesh center
                precomputed_coords = []
                if self.coords_use_cluster_center:
                    for i, face in enumerate(faces):
                        cluster_idx = face_cluster_indices[i]
                        center = centroids_reordered[cluster_idx]
                        rel = (vertices[face] - center).flatten().tolist()
                        precomputed_coords.append(rel)
                else:
                    mesh_center = mesh.centroid if hasattr(mesh, 'centroid') else vertices.mean(axis=0)
                    for _, face in enumerate(faces):
                        rel = (vertices[face] - mesh_center).flatten().tolist()
                        precomputed_coords.append(rel)
            face_to_index = {tuple(face): i for i, face in enumerate(faces)}
            # Additional geometrical features (slope, height, roughness)
            add_slope = None
            add_height = None
            add_roughness = None
            if getattr(self, 'AGF', False):
                face_centroids = vertices[faces].mean(axis=1)
                if self.include_normals and mesh_normals is not None:
                    nz = np.clip(np.abs(mesh_normals[:, 2]), 0.0, 1.0)
                else:
                    nz = np.clip(np.abs(mesh.face_normals[unique_face_indices][:, 2]), 0.0, 1.0)
                slope = np.arccos(nz)
                z_vals = vertices[:, 2]
                z_thresh = np.percentile(z_vals, 5.0)
                ground_pts = vertices[z_vals <= z_thresh]
                if ground_pts.shape[0] >= 3:
                    gp_mean = ground_pts.mean(axis=0)
                    gp_centered = ground_pts - gp_mean
                    cov = gp_centered.T @ gp_centered / max(ground_pts.shape[0] - 1, 1)
                    evals, evecs = np.linalg.eigh(cov)
                    n = evecs[:, np.argmin(evals)]
                    if n[2] < 0:
                        n = -n
                    n = n / (np.linalg.norm(n) + 1e-12)
                    d = -np.dot(n, gp_mean)
                    norm_height = face_centroids @ n + d
                else:
                    z_min = z_vals.min()
                    norm_height = face_centroids[:, 2] - z_min
                bbox_diag = float(np.linalg.norm(max_coords - min_coords))
                radius = 0.05 * bbox_diag if bbox_diag > 0 else 0.05
                tree = cKDTree(face_centroids)
                rough = np.zeros(len(face_centroids), dtype=np.float32)
                for i in range(len(face_centroids)):
                    neigh = tree.query_ball_point(face_centroids[i], r=radius)
                    if len(neigh) <= 1:
                        rough[i] = 0.0
                    else:
                        rough[i] = float(np.std(norm_height[neigh]))
                add_slope = slope
                add_height = norm_height
                add_roughness = rough

            # Load texture data for augmentation case
            base_name = os.path.splitext(os.path.basename(mesh_path))[0]
            texture_data, texture_masks = self.load_texture_data(base_name)
            if texture_data is None:
                texture_data = [np.zeros((1, 3), dtype=np.uint8) for _ in range(len(faces))]
                texture_masks = [False] * len(faces)

        start_cluster = cluster_batch_idx * self.clusters_per_batch
        end_cluster = start_cluster + self.clusters_per_batch
        selected_clusters = clusters[start_cluster:end_cluster]
        nested_list_faces = []
        nested_list_labels = []
        nested_list_textures = []
        nested_list_texture_masks = []
        nested_list_face_ids = []
        
        for cluster in selected_clusters:
            cluster_list_faces = []
            cluster_list_labels = []
            cluster_list_textures = []
            cluster_list_texture_masks = []
            cluster_face_ids = []
            
            for face in cluster:
                face_features = []
                face_idx = face_to_index[tuple(face)]
                if self.PE:
                    coords = precomputed_coords[face_idx]
                    face_features.append(coords)
                face_features.append(mesh_angle[face_idx])
                if self.include_normals and mesh_normals is not None:
                    face_features.append(mesh_normals[face_idx])
                if getattr(self, 'AGF', False) and (add_slope is not None):
                    face_features.append([add_slope[face_idx]])
                    face_features.append([add_height[face_idx]])
                    face_features.append([add_roughness[face_idx]])
                face_features = np.concatenate(face_features).tolist()
                face_features.append(mesh_area[face_idx])
                label = face_labels[face_idx]
                cluster_list_faces.append(face_features)
                cluster_list_labels.append(label)
                cluster_face_ids.append(face_idx)
                
                # Add texture data - convert to tensor for each face
                cluster_list_textures.append(torch.tensor(np.array(texture_data[face_idx]), dtype=torch.uint8))
                
                # Store only the validity flag - collate function will create proper pixel-level masks
                cluster_list_texture_masks.append(texture_masks[face_idx])  # Just True/False per face
                
            if cluster_list_faces:
                feats_tensor = torch.tensor(np.array(cluster_list_faces))
                # ---- POST-CACHE AUGMENTATION ----
                if self.augmentation is not None:
                    feats_tensor = self.augmentation(feats_tensor)
                # ---------------------------------
                nested_list_faces.append(feats_tensor)
                nested_list_labels.append(torch.tensor(np.array(cluster_list_labels)))
                
                # Pad texture data within this cluster to the same size
                if cluster_list_textures:
                    max_texture_size = max(tex.size(0) for tex in cluster_list_textures)
                    padded_textures = []
                    for tex in cluster_list_textures:
                        if tex.size(0) < max_texture_size:
                            # Pad with zeros (invalid pixels)
                            padded_tex = F.pad(tex, (0, 0, 0, max_texture_size - tex.size(0)), value=0)
                            padded_textures.append(padded_tex)
                        else:
                            padded_textures.append(tex)
                    nested_list_textures.append(padded_textures)
                else:
                    # Empty cluster
                    nested_list_textures.append([])
                
                # Store face-level validity flags (not pixel-level masks yet)
                nested_list_texture_masks.append(cluster_list_texture_masks)  # List of True/False values
                nested_list_face_ids.append(torch.tensor(np.array(cluster_face_ids), dtype=torch.long))
                
        if self.transform:
            nested_list_faces = self.transform(nested_list_faces)
            nested_list_labels = self.transform(nested_list_labels)
            nested_list_textures = self.transform(nested_list_textures)
            nested_list_texture_masks = self.transform(nested_list_texture_masks)
        # print(f"Nested list faces shape: {len(nested_list_faces[0])}")
        # print(f"Nested list labels shape: {len(nested_list_labels[0])}")
        # print(f"Nested list textures shape: {len(nested_list_textures[0])}")
        # print(f"Nested list texture masks shape: {len(nested_list_texture_masks[0])}")
        # print(f"Nested list faces shape: {nested_list_faces[0][0].shape}")
        # print(f"Nested list labels shape: {nested_list_labels[0][0].shape}")
        # print(f"Nested list textures shape: {nested_list_textures[0][0].shape}")
        # print(f"Nested list texture masks shape: {nested_list_texture_masks[0][0].shape}")
        return nested_list_faces, nested_list_labels, nested_list_textures, nested_list_texture_masks, nested_list_face_ids


def texture_custom_collate_fn(batch):
    """
    Custom collate function for texture data that handles varying sizes.
    Based on the reference implementation structure.
    
    Args:
        batch (list): List of tuples (nested_list_faces, nested_list_labels, nested_list_textures, nested_list_texture_masks).
        
    Returns:
        padded_batches (Tensor): Batched and padded face features, shape (B, P, S, F).
        padded_labels (Tensor): Batched and padded labels, shape (B, P, L).
        padded_textures (Tensor): Batched and padded texture sequences, shape (B, P, S, T, C).
        masks (Tensor): Mask tensor indicating valid entries, shape (B, P, S).
        texture_masks (Tensor): Texture validity masks, shape (B, P, S, T).
    """
    max_patch_size = max(len(item[0]) for item in batch)
    max_sequence_length = max(max(patch.size(0) for patch in item[0]) for item in batch)
    max_label_length = max(max(patch.size(0) for patch in item[1]) for item in batch)
    
    padded_batches = []
    padded_labels = []
    padded_textures = []
    masks = []
    texture_masks = []
    face_id_batches = []

    for data, labels, textures, tex_masks, face_ids in batch:
        # Pad face features
        padded_data = [F.pad(patch, (0, 0, 0, max_sequence_length - patch.size(0))) for patch in data]
        padded_data += [torch.zeros(max_sequence_length, data[0].size(1))] * (max_patch_size - len(data))
        padded_batches.append(torch.stack(padded_data))

        # Pad labels
        padded_label = [F.pad(patch, (0, max_label_length - patch.size(0))) for patch in labels]
        padded_label += [torch.zeros(max_label_length)] * (max_patch_size - len(labels))
        padded_labels.append(torch.stack(padded_label))

        # Create masks
        batch_mask = torch.zeros((max_patch_size, max_sequence_length), dtype=torch.bool)
        for i, patch in enumerate(data):
            batch_mask[i, :patch.size(0)] = 1
        masks.append(batch_mask)

        # Handle textures - preserve cluster structure to match geometry
        # Model expects [B, Clusters, Faces_per_cluster, max_pixels, C]
        if textures:
            # print(f"DEBUG: Number of texture clusters: {len(textures)}")
            # print(f"DEBUG: Faces in first cluster: {len(textures[0]) if textures else 0}")
            # print(f"DEBUG: First face texture shape: {textures[0][0].shape if textures and textures[0] else 'None'}")
            
            # Find max dimensions across all textures
            max_pixels = max(tex.size(0) for tex_list in textures for tex in tex_list) if any(textures) else 1
            channels = textures[0][0].size(1) if textures and textures[0] and len(textures[0][0].shape) > 1 else 3
            print(f"DEBUG: max_pixels: {max_pixels}, channels: {channels}")
            
            padded_texture_clusters = []
            cluster_texture_masks = []
            
            for tex_list, tex_mask_list in zip(textures, tex_masks):
                # Pad each face texture in this cluster
                padded_faces = []
                pixel_masks = []
                
                for face_tex, face_valid in zip(tex_list, tex_mask_list):
                    # Get original size before padding
                    original_pixels = face_tex.size(0)
                    
                    if len(face_tex.shape) == 1:
                        # 1D texture - pad to max_pixels and add channel dim
                        padded = F.pad(face_tex, (0, max_pixels - face_tex.size(0)))
                        padded = padded.unsqueeze(-1).expand(-1, channels)
                    else:
                        # 2D texture - pad pixels dimension
                        padded = F.pad(face_tex, (0, 0, 0, max_pixels - face_tex.size(0)))
                    # Convert to float32 to avoid potential uint8 issues
                    padded_faces.append(padded.float())
                    
                    # Create pixel-level mask based on actual padding done here
                    pixel_mask = torch.zeros(max_pixels, dtype=torch.bool)
                    if face_valid:  # If face has valid texture data
                        pixel_mask[:original_pixels] = True  # Mark original pixels as valid
                    # If face_valid is False, all pixels remain False (invalid)
                    pixel_masks.append(pixel_mask)
                
                # Pad to max_sequence_length faces per cluster
                while len(padded_faces) < max_sequence_length:
                    padded_faces.append(torch.zeros(max_pixels, channels, dtype=torch.float32))
                    pixel_masks.append(torch.zeros(max_pixels, dtype=torch.bool))  # All pixels invalid
                
                padded_texture_clusters.append(torch.stack(padded_faces))
                cluster_texture_masks.append(torch.stack(pixel_masks))
            
            # Pad to max_patch_size clusters
            while len(padded_texture_clusters) < max_patch_size:
                empty_cluster = torch.zeros(max_sequence_length, max_pixels, channels, dtype=torch.float32)
                empty_mask = torch.zeros(max_sequence_length, max_pixels, dtype=torch.bool)
                padded_texture_clusters.append(empty_cluster)
                cluster_texture_masks.append(empty_mask)
            
            final_texture_tensor = torch.stack(padded_texture_clusters)
            final_mask_tensor = torch.stack(cluster_texture_masks)
            # print(f"DEBUG: Final texture tensor shape: {final_texture_tensor.shape}")
            # print(f"DEBUG: Final mask tensor shape: {final_mask_tensor.shape}")
            padded_textures.append(final_texture_tensor)
            texture_masks.append(final_mask_tensor)
        else:
            # Handle case with no textures
            padded_textures.append(torch.zeros((max_patch_size, max_sequence_length, 1, 3), dtype=torch.float32))
            texture_masks.append(torch.zeros((max_patch_size, max_sequence_length, 1), dtype=torch.bool))
        
        # Process face_ids
        padded_face_ids = []
        for face_id_list in face_ids:
            # Pad face_ids to max_sequence_length
            padded_face_id = F.pad(face_id_list, (0, max_sequence_length - face_id_list.size(0)), value=-1)
            padded_face_ids.append(padded_face_id)
        
        # Pad to max_patch_size
        while len(padded_face_ids) < max_patch_size:
            padded_face_ids.append(torch.full((max_sequence_length,), -1, dtype=torch.long))
        
        face_id_batches.append(torch.stack(padded_face_ids))

    final_batch = torch.stack(padded_batches)
    final_labels = torch.stack(padded_labels) 
    final_textures = torch.stack(padded_textures)
    final_masks = torch.stack(masks)
    final_texture_masks = torch.stack(texture_masks)
    
    # print(f"DEBUG: Final batch return shapes:")
    # print(f"  Batches: {final_batch.shape}, dtype: {final_batch.dtype}")
    # print(f"  Labels: {final_labels.shape}, dtype: {final_labels.dtype}")
    # print(f"  Textures: {final_textures.shape}, dtype: {final_textures.dtype}")
    # print(f"  Masks: {final_masks.shape}, dtype: {final_masks.dtype}")
    # print(f"  Texture masks: {final_texture_masks.shape}, dtype: {final_texture_masks.dtype}")
    
    # print(f"DEBUG: Texture tensor info:")
    # print(f"  Is contiguous: {final_textures.is_contiguous()}")
    # print(f"  Storage size: {final_textures.storage().size()}")
    # print(f"  Element size: {final_textures.element_size()}")
    
    face_ids_tensor = torch.stack(face_id_batches)
    return final_batch, final_labels, final_textures, final_masks, final_texture_masks, face_ids_tensor

