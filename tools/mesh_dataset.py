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

# ---------- helpers: clustering like your first code ----------
def compute_cluster_centroids(vertices, labels, n_clusters):
    centroids = np.zeros((n_clusters, 3), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for i, v in enumerate(vertices):
        c = labels[i]
        centroids[c] += v
        counts[c] += 1
    counts = np.maximum(counts, 1)  # avoid divide-by-zero
    centroids /= counts[:, None]
    return centroids

def reorder_clusters_by_proximity(centroids):
    if len(centroids) <= 1:
        return list(range(len(centroids)))
    dist_matrix = cdist(centroids, centroids, metric='euclidean')
    mst = minimum_spanning_tree(dist_matrix).toarray()
    mst[mst == 0] = np.inf
    visited, ordered = set(), []
    def dfs(u):
        if u in visited: return
        visited.add(u); ordered.append(u)
        for v in np.argsort(mst[u]):
            if v not in visited and np.isfinite(mst[u, v]):
                dfs(v)
    for i in range(len(centroids)):
        dfs(i)
    if len(ordered) < len(centroids):
        remaining = set(range(len(centroids))) - set(ordered)
        remaining = sorted(remaining, key=lambda x: np.min(dist_matrix[x, ordered]))
        ordered.extend(remaining)
    return ordered


class MeshAugmentation:
    """
    Applies random augmentations to a mesh (geometry) or to feature tensors (post-cache).
    Args:
        rotation_range (float): Maximum rotation angle in degrees.
        scale_range (tuple): Min and max scaling factors.
        noise_std (float): Standard deviation of Gaussian noise (for mesh or features).
        flip_probability (float): Probability of flipping the mesh.
        feature_noise_std (float): Standard deviation of noise for feature tensors.
    """
    def __init__(self, rotation_range=15, scale_range=(0.9, 1.1), noise_std=0.01, flip_probability=0.5, feature_noise_std=0.01):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.flip_probability = flip_probability
        self.feature_noise_std = feature_noise_std

    def __call__(self, mesh, mode='RSN'):
        """
        Applies augmentations to the mesh geometry.
        Args:
            mesh (trimesh.Trimesh): The mesh to augment.
            mode (str): Which augmentations to apply ('R'=rotation, 'S'=scaling, 'N'=noise, 'F'=flip).
        Returns:
            mesh (trimesh.Trimesh): The augmented mesh.
        """
        # Apply flipping (if enabled and randomly selected)
        if 'F' in mode and np.random.rand() < 0.5:
            if np.random.rand() < self.flip_probability:
                mesh.invert()

        # Apply rotation (if enabled and randomly selected)
        if 'R' in mode and np.random.rand() < 0.5:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle=angle, 
                direction=axis, 
                point=mesh.centroid
            )
            mesh.apply_transform(rotation_matrix)

        # Apply scaling (if enabled and randomly selected)
        if 'S' in mode and np.random.rand() < 0.5:
            scaling_matrix = tf.scale_matrix(self.scale_range)
            mesh.apply_transform(scaling_matrix)

        # Apply noise (if enabled and randomly selected)
        if 'N' in mode and np.random.rand() < 0.5:
            noise = np.random.randn(*mesh.vertices.shape) * self.noise_std
            mesh.vertices += noise

        return mesh

    def augment_features(self, features):
        """
        Applies augmentation to feature tensors (post-cache).
        Args:
            features (torch.Tensor): The input features tensor.
        Returns:
            torch.Tensor: Augmented features.
        """
        if not torch.is_tensor(features):
            features = torch.tensor(features)
        noise = torch.randn_like(features) * self.feature_noise_std
        return features + noise



class MeshDataset(Dataset):
    """
    PyTorch Dataset for loading 3D mesh data, labels, and optional features, with clustering and augmentation.
    Args:
        mesh_dir (str): Directory with mesh files (.obj or .ply).
        label_dir (str): Directory with label files (.txt).
        n_clusters (int): Number of clusters for K-Means.
        clusters_per_batch (int): Number of clusters per batch.
        PE (bool): Whether to use positional encoding.
        json_dir (str, optional): Directory with JSON feature files.
        augmentation (callable, optional): Augmentation function to apply to meshes.
        transform (callable, optional): Optional transform to apply to samples.
    """
    def __init__(self, mesh_dir, label_dir, n_clusters, clusters_per_batch, PE, json_dir=None, augmentation=None, transform=None, include_normals=True, additional_geometrical_features=False, coords_use_cluster_center=False, pe_bbox_normalized=True, require_labels=True):
        self.mesh_dir = mesh_dir
        self.label_dir = label_dir
        self.json_dir = json_dir
        self.require_labels = require_labels
        self.mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj') or f.endswith('.ply')]
        self.n_clusters = n_clusters
        self.clusters_per_batch = clusters_per_batch
        self.transform = transform
        self.PE = PE
        self.augmentation = augmentation
        self.include_normals = include_normals
        self.AGF = additional_geometrical_features
        self.coords_use_cluster_center = coords_use_cluster_center
        self.pe_bbox_normalized = pe_bbox_normalized

        # Caching setup
        self.cache_dir = os.path.join(mesh_dir, ".cluster_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._in_memory_cache = {}  # mesh_path -> payload

        # Verify the label directory exists (only if label_dir is provided)
        if require_labels and label_dir is not None and not os.path.isdir(label_dir):
            raise FileNotFoundError(f"The label directory {label_dir} does not exist.")

        # Verify the JSON directory exists if provided
        if json_dir and not os.path.isdir(json_dir):
            raise FileNotFoundError(f"The json directory {json_dir} does not exist.")

        # Load labels
        self.labels = self.load_labels()
        print(f"Found {len(self.mesh_files)} mesh files")

    def load_labels(self):
        """
        Loads the labels for each mesh file.
        If label_dir is provided, loads from separate .txt files.
        Otherwise, tries to load from cls property in PLY files.
        Returns:
            labels (dict): Mapping from mesh base name to list of face labels.
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
        else:
            print(f"[CACHE] No disk cache directory found at {self.cache_dir}")

    def load_json_features(self, base_name):
        """
        Loads additional features from the JSON file for the given mesh.
        Args:
            base_name (str): Base name of the mesh file (without extension).
        Returns:
            json_features (dict or None): Loaded JSON features or None if not found.
        """
        if self.json_dir:
            json_file_path = os.path.join(self.json_dir, base_name + '.json')
            if os.path.isfile(json_file_path):
                with open(json_file_path, 'r') as f:
                    json_features = json.load(f)
                return json_features
        return None

    def _compute_and_cache(self, mesh_path, label_file_name):
        mesh = trimesh.load(mesh_path, force='mesh')
        vertices = mesh.vertices
        faces = mesh.faces

        # Normalize coordinates to the range [0, 1]
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        normalized_vertices = (vertices - min_coords) / (max_coords - min_coords)
        
        # Load labels or use dummy labels for prediction mode
        if self.require_labels:
            face_labels = self.labels[label_file_name]
            if len(face_labels) != len(faces):
                raise ValueError(f"Number of labels {len(face_labels)} does not match number of faces {len(faces)} for {mesh_path}")
        else:
            # Use dummy labels (0) for all faces in prediction mode
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

        mesh_normals = mesh.face_normals[unique_face_indices]
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
            nz = np.clip(np.abs(mesh_normals[:, 2]), 0.0, 1.0)
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
            # Memory-efficient neighborhood via KD-tree
            tree = cKDTree(face_centroids)
            rough = np.zeros(len(face_centroids), dtype=np.float32)
            # Iterate in chunks to avoid Python overhead for extreme sizes
            for i in range(len(face_centroids)):
                neigh = tree.query_ball_point(face_centroids[i], r=radius)
                if len(neigh) <= 1:
                    rough[i] = 0.0
                else:
                    rough[i] = float(np.std(norm_height[neigh]))
            add_slope = slope
            add_height = norm_height
            add_roughness = rough

        payload = {
            'clusters': clusters,
            'mesh_angle': mesh_angle,
            'mesh_normals': mesh_normals if self.include_normals else None,
            'mesh_area': mesh_area,
            'precomputed_coords': precomputed_coords,
            'face_to_index': face_to_index,
            'face_labels': face_labels,
            'add_slope': add_slope,
            'add_height': add_height,
            'add_roughness': add_roughness
        }
        cache_file = os.path.join(
            self.cache_dir,
            f"{os.path.basename(mesh_path)}__clusters{self.n_clusters}_PE{int(self.PE)}_NORM{int(self.include_normals)}_AGF{int(self.AGF)}_CC{int(self.coords_use_cluster_center)}_PEBN{int(self.pe_bbox_normalized)}.pkl"
        )
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir)
        with os.fdopen(fd, 'wb') as tmpf:
            pickle.dump(payload, tmpf)
        os.replace(tmp_path, cache_file)
        self._in_memory_cache[mesh_path] = payload

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            length (int): Number of mesh samples (mesh files * cluster batches per mesh).
        """
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
            f"{os.path.basename(mesh_path)}__clusters{self.n_clusters}_PE{int(self.PE)}_NORM{int(self.include_normals)}_AGF{int(self.AGF)}_CC{int(self.coords_use_cluster_center)}_PEBN{int(self.pe_bbox_normalized)}.pkl"
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
            face_labels = self.labels[label_file_name]
            if len(face_labels) != len(faces):
                raise ValueError(f"Number of labels {len(face_labels)} does not match number of faces {len(faces)} for {mesh_file}")
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
                # Memory-efficient neighborhood via KD-tree
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
        start_cluster = cluster_batch_idx * self.clusters_per_batch
        end_cluster = start_cluster + self.clusters_per_batch
        selected_clusters = clusters[start_cluster:end_cluster]
        nested_list_faces = []
        nested_list_labels = []
        for cluster in selected_clusters:
            cluster_list_faces = []
            cluster_list_labels = []
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
            if cluster_list_faces:
                feats_tensor = torch.tensor(np.array(cluster_list_faces))
                # ---- POST-CACHE AUGMENTATION ----
                if self.augmentation is not None:
                    feats_tensor = self.augmentation(feats_tensor)
                # ---------------------------------
                nested_list_faces.append(feats_tensor)
                nested_list_labels.append(torch.tensor(np.array(cluster_list_labels)))
        if self.transform:
            nested_list_faces = self.transform(nested_list_faces)
            nested_list_labels = self.transform(nested_list_labels)
        return nested_list_faces, nested_list_labels


# Custom collate function to handle varying sizes of vertices and faces
def custom_collate_fn(batch):
    """
    Pads and stacks a batch of mesh data with varying sizes for use in a DataLoader.
    Args:
        batch (list): List of tuples (nested_list_faces, nested_list_labels) from MeshDataset.__getitem__.
    Returns:
        padded_batches (Tensor): Batched and padded face features, shape (B, P, S, F).
        padded_labels (Tensor): Batched and padded labels, shape (B, P, L).
        masks (Tensor): Mask tensor indicating valid entries, shape (B, P, S).
    """
    max_patch_size = max(len(item[0]) for item in batch)
    max_sequence_length = max(max(patch.size(0) for patch in item[0]) for item in batch)
    max_label_length = max(max(patch.size(0) for patch in item[1]) for item in batch)
    
    padded_batches = []
    padded_labels = []
    masks = []

    for data, label in batch:
        padded_data = [F.pad(patch, (0, 0, 0, max_sequence_length - patch.size(0))) for patch in data]
        padded_data += [torch.zeros(max_sequence_length, data[0].size(1))] * (max_patch_size - len(data))
        padded_batches.append(torch.stack(padded_data))
        
        padded_label = [F.pad(patch, (0, max_label_length - patch.size(0))) for patch in label]
        padded_label += [torch.zeros(max_label_length)] * (max_patch_size - len(label))
        padded_labels.append(torch.stack(padded_label))
        
        batch_mask = torch.zeros((max_patch_size, max_sequence_length), dtype=torch.bool)
        for i, patch in enumerate(data):
            batch_mask[i, :patch.size(0)] = 1
        masks.append(batch_mask)
        
    return torch.stack(padded_batches), torch.stack(padded_labels), torch.stack(masks)




