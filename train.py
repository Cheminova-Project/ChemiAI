import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tools.mesh_dataset import MeshDataset, custom_collate_fn  ,MeshAugmentation
from mesh_texture_dataset import MeshTextureDataset, texture_custom_collate_fn
from tools.model_base import nomeformer
from tools.downst import DownstreamClassifier
from integrated_texture_geometry_model import IntegratedTextureGeometryModel, IntegratedDownstreamClassifier
from tools.helper import init_opt
from tools.check_point import save_checkpoint , load_checkpoint, save_best_checkpoint, load_best_checkpoint_if_exists
from tqdm import tqdm
import os
from ignite.metrics import IoU, ConfusionMatrix  # Metrics for evaluation
from torchmetrics import F1Score, Accuracy  # Additional metrics
import copy
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from loss import MaskedCrossEntropyLoss


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


# ===================== EMA Implementation =====================
class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register model parameters for EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """Get EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'decay': self.decay
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


# ===================== Argument Parsing =====================
# Parse command-line arguments for configuration file
parser = argparse.ArgumentParser(description='Train a model with a given configuration file.')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
args = parser.parse_args()

# ===================== Configuration Loading =====================
# Load configuration from YAML file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# ===================== Extract Config Values =====================
# Paths for data, logs, and checkpoints
mesh_dir = config['paths']['mesh_dir']
label_dir = config['paths'].get('label_dir', None)  # Make optional for PLY cls property usage
json_dir = config['paths']['json_dir']
log_dir = config['paths']['log_dir']
checkpoint_dir = config['paths']['checkpoint_dir']
checkpoint_pertrain = config['paths'].get('checkpoint_pertrain', None)
val_mesh_dir = config['paths']['val_mesh_dir']
val_label_dir = config['paths'].get('val_label_dir', None)  # Make optional for PLY cls property usage
val_json_dir = config['paths']['val_json_dir']

# Texture-specific paths
texture_dir = config['paths'].get('texture_dir', None)
val_texture_dir = config['paths'].get('val_texture_dir', None)

# Training hyperparameters
Training_epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
clusters_per_batch = config['training']['clusters_per_batch']
learning_rate = config['training']['learning_rate']
accumulation_steps = config['training'].get('accumulation_steps', 32)
resume = config['training'].get('resume', False)
use_pretrained = config['training'].get('use_pretrained', False)
# Hardcode to unfreeze all encoder layers regardless of config
unfreeze_all = True
num_unfrozen_encoder_layers = 0

# Weight decay configuration
weight_decay = config['training'].get('weight_decay', 1e-4)  # Default to 1e-4 if not specified

# EMA configuration
ema_decay = config['training'].get('ema_decay', 0.9999)  # Default EMA decay
use_ema = config['training'].get('use_ema', False)  # Enable EMA by default

# Ignore index configuration
ignore_index = config['paths'].get('ignore_index', None)  # Can be None, or a specific class to ignore
ignore_index = ignore_index + 1


# Optional layer-wise learning rate decay configuration
# Example: training.layer_wise_lr_decay: [10, 2]
#   - head_lr_multiplier = 10 (applies to classifier/head)
#   - decay_factor = 2 (per-layer decay from last block backward)
layer_wise_lr_decay = config['training'].get('layer_wise_lr_decay', None)

# Model hyperparameters
# hierarchical configuration
use_hierarchical = config['model'].get('use_hierarchical', True)
# Hardcode feature_dim to 16 instead of reading from config
feature_dim = 16
embedding_dim = config['model']['embedding_dim']
num_heads = config['model']['num_heads']
num_attention_blocks = config['model']['num_attention_blocks']
N_class = config['model']['n_classes']
n_clusters = config['model']['n_clusters']
PE = config['model'].get('use_pe', True)
Gradinat_ac = config['model'].get('gradinat_ac', False)
dropout = config['model'].get('dropout', 0.1) 
include_normals = config['model'].get('include_normals', True)
additional_geometrical_features = config['model'].get('additional_geometrical_features', False)

# Texture-specific configuration
use_texture = config['model'].get('use_texture', True)
texture_embed_dim = config['model'].get('texture_embed_dim', 64)
fusion_method = config['model'].get('fusion_method', 'gated')
max_texture_pixels = config['model'].get('max_texture_pixels', 128)
texture_patch_size = config['model'].get('texture_patch_size', 16)

# ===================== Validation Function =====================
def evaluate(model, val_data_loader, ema=None):
    """Evaluate the model on the validation set and return metrics."""
    # Apply EMA parameters for evaluation if available
    if ema is not None:
        ema.apply_shadow()
    
    model.eval()
    val_running_loss = 0.0
    val_confusion_matrix = ConfusionMatrix(num_classes=N_class)
    
    # Create metrics with dynamic ignore_index
    if ignore_index is not None:
        val_miou_metric = IoU(cm=val_confusion_matrix, ignore_index=ignore_index)
        val_f1_metric = F1Score(task='multiclass', num_classes=N_class, average='none', ignore_index=ignore_index).to(device)
    else:
        val_miou_metric = IoU(cm=val_confusion_matrix)
        val_f1_metric = F1Score(task='multiclass', num_classes=N_class, average='none').to(device)
    
    val_f1_metric.reset()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in val_data_loader:
            if use_texture and texture_dir is not None:
                # Texture dataset returns: (geometry_features, texture_sequences, labels, masks, texture_masks)
                geometry_features, labels, texture_sequences, masks, texture_masks ,_ = data
                geometry_features = geometry_features.to(device)
                texture_sequences = texture_sequences.to(device)
                labels = labels.to(device)
                if mesh_dir == '/bigwork/nhgnheid/EU_data_V2/train' or mesh_dir == '/bigwork/nhgnheid/EU_data_V3/Train':
                    labels = labels + 1
                masks = masks.to(device)
                texture_masks = texture_masks.to(device)
                output = model(geometry_features, texture_sequences, masks, texture_masks)
            else:
                # Geometry-only dataset returns: (batch, labels, masks)
                batch, labels, masks = data
                batch = batch.to(device)
                if mesh_dir == '/bigwork/nhgnheid/EU_data_V2/train' or mesh_dir == '/bigwork/nhgnheid/EU_data_V3/Train':
                    labels = labels + 1
                labels = labels.to(device)
                masks = masks.to(device)
                output = model(batch, masks)
            
            loss = masked_loss_fn(output, labels, masks)
            val_running_loss += loss.item()
            pred = output.reshape(-1, N_class)
            target = labels.reshape(-1)
            mask = masks.view(-1)
            valid_mask = mask == 1
            pred = pred[valid_mask]
            target = target[valid_mask]
            val_confusion_matrix.update((pred, target))
            
            # Get predictions for overall accuracy
            pred_classes = pred.argmax(dim=1)
            val_f1_metric.update(pred_classes, target)
            
            # Collect all predictions and targets for overall accuracy
            all_preds.append(pred_classes)
            all_targets.append(target)
    
    # Compute metrics
    val_f1_scores = val_f1_metric.compute()
    if ignore_index is not None and ignore_index < N_class:
        class_mask = torch.arange(N_class) != ignore_index
        val_mean_f1_score = val_f1_scores[class_mask].mean().item()
    else:
        val_mean_f1_score = val_f1_scores.mean().item()
    
    # Compute overall accuracy using simple correct/total approach
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    val_overall_accuracy = overall_accuracy(all_preds, all_targets, ignore_index=ignore_index)
    
    val_miou = val_miou_metric.compute()
    model.train()
    
    # Restore original parameters if EMA was used
    if ema is not None:
        ema.restore()
    
    # Return confusion matrix as well
    return val_running_loss / len(val_data_loader), val_mean_f1_score, val_overall_accuracy, val_miou, val_f1_scores, val_confusion_matrix

# ===================== Data Augmentation =====================
# Set up data augmentation using parameters from config
augmentation = MeshAugmentation(
    rotation_range=config['augmentation']['rotation_range'],
    scale_range=config['augmentation']['scale_range'],
    noise_std=config['augmentation']['noise_std'],
    flip_probability=config['augmentation']['flip_probability']
)
mode = config['augmentation'].get('mode', 'RSN')

# ===================== Optimizer and Class Weights =====================
# Set up optimizer from config if specified
optimizer_type = None

# Set up class weights for loss function if provided
class_percentages = None
if 'class_weights' in config:
    class_percentages = config['class_weights']['class_percentages']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== Mixed Precision Setup =====================
use_mixed_precision = True  # Enable mixed precision training
scaler = GradScaler() if use_mixed_precision else None
print(f"🚀 Mixed precision training: {'Enabled' if use_mixed_precision else 'Disabled'}")

# Memory monitoring function
def monitor_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"🔍 GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {max_memory:.1f}GB total")
        return allocated, reserved, max_memory
    return 0, 0, 0

# ===================== Training Setup =====================
# Print training configuration
print(f"Training Configuration:")
print(f"  - Learning Rate: {learning_rate}")
print(f"  - Weight Decay: {weight_decay}")
print(f"  - Batch Size: {batch_size}")
print(f"  - Epochs: {Training_epochs}")
print(f"  - Device: {device}")
print(f"  - Number of Classes: {N_class}")
print(f"  - Ignore Index: {ignore_index if ignore_index is not None else 'None (all classes used)'}")
print(f"  - EMA: {'Enabled' if use_ema else 'Disabled'}")
print(f"  - Texture Features: {'Enabled' if use_texture else 'Disabled'}")
if use_ema:
    print(f"  - EMA Decay: {ema_decay}")
if use_texture:
    print(f"  - Texture Embed Dim: {texture_embed_dim}")
    print(f"  - Fusion Method: {fusion_method}")
    print(f"  - Max Texture Pixels: {max_texture_pixels}")


# Instantiate dataset and DataLoader
if use_texture and texture_dir is not None:
    print("Using MeshTextureDataset with texture features")
    if label_dir is not None:
        print(f"Using separate label files from: {label_dir}")
    else:
        print("No separate label directory provided - will read labels from PLY cls property")
    dataset = MeshTextureDataset(
        mesh_dir=mesh_dir,
        label_dir=label_dir,
        texture_dir=texture_dir,
        n_clusters=n_clusters,
        clusters_per_batch=clusters_per_batch,
        PE=PE,
        json_dir=json_dir,
        augmentation=None,
        transform=None,
        include_normals=include_normals,
        additional_geometrical_features=additional_geometrical_features,
        texture_patch_size=texture_patch_size,
        max_texture_pixels=max_texture_pixels,
        pe_bbox_normalized=True
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=texture_custom_collate_fn)
    
    if val_label_dir is not None:
        print(f"Validation: Using separate label files from: {val_label_dir}")
    else:
        print("Validation: No separate label directory provided - will read labels from PLY cls property")
    val_dataset = MeshTextureDataset(
        mesh_dir=val_mesh_dir,
        label_dir=val_label_dir,
        texture_dir=val_texture_dir if val_texture_dir is not None else texture_dir,
        n_clusters=n_clusters,
        clusters_per_batch=clusters_per_batch,
        PE=PE,
        json_dir=val_json_dir,
        augmentation=None,
        transform=None,
        include_normals=include_normals,
        additional_geometrical_features=additional_geometrical_features,
        texture_patch_size=texture_patch_size,
        max_texture_pixels=max_texture_pixels,
        pe_bbox_normalized=True
    )
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=texture_custom_collate_fn)
else:
    print("Using MeshDataset with geometry features only")
    if label_dir is not None:
        print(f"Using separate label files from: {label_dir}")
    else:
        print("No separate label directory provided - will read labels from PLY cls property")
    dataset = MeshDataset(
        mesh_dir=mesh_dir,
        label_dir=label_dir,
        n_clusters=n_clusters,
        clusters_per_batch=clusters_per_batch,
        PE=PE,
        json_dir=json_dir,
        augmentation=None,
        transform=None,
        include_normals=include_normals,
        additional_geometrical_features=additional_geometrical_features,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    if val_label_dir is not None:
        print(f"Validation: Using separate label files from: {val_label_dir}")
    else:
        print("Validation: No separate label directory provided - will read labels from PLY cls property")
    val_dataset = MeshDataset(
        mesh_dir=val_mesh_dir,
        label_dir=val_label_dir,
        n_clusters=n_clusters,
        clusters_per_batch=clusters_per_batch,
        PE=PE,
        json_dir=val_json_dir,
        augmentation=None,
        transform=None,
        include_normals=include_normals,
        additional_geometrical_features=additional_geometrical_features,
    )
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# dataset = MeshDataset(
#     mesh_dir=mesh_dir,
#     label_dir=label_dir,
#     clusters_per_batch=clusters_per_batch,
#     faces_per_cluster=faces_per_cluster,
#     PE=PE,
#     json_dir=json_dir,
#     augmentation= None,
#     classification=False,
# )
# data_loader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     collate_fn=custom_collate_fn,
# )
# val_dataset = MeshDataset(
#     mesh_dir=val_mesh_dir,
#     label_dir=val_label_dir,
#     clusters_per_batch=clusters_per_batch,
#     faces_per_cluster=faces_per_cluster,
#     PE=PE,
#     json_dir=val_json_dir,
#     augmentation=None,
#     classification=False,
# )
# val_data_loader = DataLoader(
#     val_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     collate_fn=custom_collate_fn,
# )

# Load pre-trained encoder
# For hierarchical processing (optional)

if use_texture and texture_dir is not None:
    # Create integrated texture-geometry model
    pretrained_encoder = IntegratedTextureGeometryModel(
        geometry_feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        texture_embed_dim=texture_embed_dim,
        num_heads=num_heads,
        num_attention_blocks=num_attention_blocks,
        dropout=dropout,
        summary_mode='cls',
        use_hierarchical=use_hierarchical,
        fourier=False,
        relative_positional_encoding=False,
        fusion_method=fusion_method,
        max_texture_pixels=max_texture_pixels
    )
    print("Created integrated texture-geometry model")
else:
    # Create geometry-only model
    pretrained_encoder = nomeformer(
        feature_dim=feature_dim,            # your input feature size
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_attention_blocks=num_attention_blocks,
        dropout=dropout,
        summary_mode='cls',       # or 'avg'
        use_hierarchical=use_hierarchical,    # or False
        num_hierarchical_stages=1,
        fourier=False,             # keep as you use now
        relative_positional_encoding=False  # <-- activate RPE
    )
    print("Created geometry-only model")

# pretrained_encoder = nomeformer(
#     feature_dim, embedding_dim, num_heads, num_attention_blocks, dropout, 
#     summary_mode='cls'
# ).to(device)
# Load pre-trained weights
if use_pretrained and checkpoint_pertrain is not None and os.path.exists(checkpoint_pertrain):
    checkpoint = torch.load(checkpoint_pertrain, map_location=device)
    if use_texture and texture_dir is not None:
        # For integrated model, try to load geometry encoder weights
        if 'target_encoder' in checkpoint:
            # Try to load geometry encoder weights only
            try:
                if hasattr(pretrained_encoder, 'geometry_encoder'):
                    pretrained_encoder.geometry_encoder.load_state_dict(checkpoint['target_encoder'])
                    print(f"Loaded pretrained geometry encoder from {checkpoint_pertrain}")
                else:
                    print("Warning: Integrated model doesn't have geometry_encoder attribute")
            except Exception as e:
                print(f"Warning: Could not load geometry encoder weights: {e}")
        else:
            print("Warning: No 'target_encoder' key found in checkpoint")
    else:
        # For geometry-only model
        pretrained_encoder.load_state_dict(checkpoint['target_encoder'])
        print(f"Loaded pretrained encoder from {checkpoint_pertrain}")
else:
    if use_pretrained and checkpoint_pertrain is not None:
        print(f"Warning: pretrained checkpoint not found at {checkpoint_pertrain}, proceeding with random init")
    else:
        print("Skipping pretrained weights — encoder will be randomly initialized")
    # If you want to explicitly re-initialize the encoder weights
    def _init_enc_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    pretrained_encoder.apply(_init_enc_weights)

# Create downstream model
if use_texture and texture_dir is not None:
    model = IntegratedDownstreamClassifier(
        integrated_encoder=pretrained_encoder,
        num_classes=N_class,
        embedding_dim=embedding_dim,
        dropout=dropout,
        freeze_encoder_layers=num_unfrozen_encoder_layers,
        fusion_method=fusion_method
    ).to(device)
    print("Created integrated downstream classifier")
else:
    model = DownstreamClassifier(pretrained_encoder, N_class, embedding_dim, dropout, True, num_unfrozen_encoder_layers).to(device)
    print("Created geometry-only downstream classifier")
# Force unfreeze all encoder layers
for p in model.encoder.parameters():
    p.requires_grad = True
print("All encoder layers are now trainable (hardcoded).")

# Initialize EMA if enabled
ema = None
if use_ema:
    ema = EMA(model, decay=ema_decay)
    print(f"EMA enabled with decay={ema_decay}")
else:
    print("EMA disabled")

# Check if encoder parameters are frozen
for name, param in model.encoder.named_parameters():
    if param.requires_grad:
        print(f"WARNING: Encoder parameter {name} is trainable!")
    else:
        print(f"Encoder parameter {name} is frozen.")
# Define loss function and optimizer
# Create loss function with dynamic ignore_index
if ignore_index is not None:
    masked_loss_fn = MaskedCrossEntropyLoss(ignore_index=ignore_index)
else:
    masked_loss_fn = MaskedCrossEntropyLoss()
def _split_decay_param_groups(all_named_parameters, candidate_params=None):
    """Return (decay, no_decay) param lists using names to exclude biases and norm weights.

    If candidate_params is provided, it filters only those tensors; otherwise, uses all.
    """
    # Map param id to (name, param) for quick reverse lookup
    id_to_named = {id(p): (n, p) for n, p in all_named_parameters}

    def is_no_decay_param(param_name):
        lower_name = param_name.lower()
        if param_name.endswith('.bias'):
            return True
        # Common norm module parameter names
        if any(k in lower_name for k in ['layernorm', 'batchnorm', 'batch_norm', 'groupnorm', 'instancenorm', 'rmsnorm', '.ln', '.bn', '.gn', '.in', 'norm']):
            # Restrict to weights of norm layers; name-based heuristic
            if param_name.endswith('.weight') or param_name.endswith('.bias'):
                return True
        return False

    decay, no_decay = [], []
    if candidate_params is None:
        # Use all params
        for name, param in id_to_named.values():
            if not param.requires_grad:
                continue
            (no_decay if is_no_decay_param(name) else decay).append(param)
    else:
        for p in candidate_params:
            if not p.requires_grad:
                continue
            name = id_to_named.get(id(p), (None, None))[0]
            if name is None:
                # Fallback: if name missing, assume decay for weights, no_decay for 1D tensors
                ((no_decay if p.ndim == 1 else decay).append(p))
            else:
                (no_decay if is_no_decay_param(name) else decay).append(p)
    return decay, no_decay
if layer_wise_lr_decay is not None:
    assert isinstance(layer_wise_lr_decay, (list, tuple)) and len(layer_wise_lr_decay) == 2, \
        "training.layer_wise_lr_decay must be a list like [head_multiplier, decay_factor]"
    head_multiplier, decay_factor = float(layer_wise_lr_decay[0]), float(layer_wise_lr_decay[1])

    param_groups = []

    # 1) Classifier / head parameters
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    if len(head_params) > 0:
        decay_params, no_decay_params = _split_decay_param_groups(model.named_parameters(), head_params)
        if len(decay_params) > 0:
            param_groups.append({
                'params': decay_params,
                'lr': learning_rate * head_multiplier,
                'weight_decay': weight_decay,
            })
        if len(no_decay_params) > 0:
            param_groups.append({
                'params': no_decay_params,
                'lr': learning_rate * head_multiplier,
                'weight_decay': 0.0,
            })

    # 2) Encoder blocks with per-layer decay (from last to first)
    encoder = model.encoder
    used_param_ids = set(id(p) for p in head_params)
    
    # Debug: Check encoder structure
    print(f"Encoder type: {type(encoder)}")
    print(f"Has attention_blocks: {hasattr(encoder, 'attention_blocks')}")
    if hasattr(encoder, 'attention_blocks'):
        print(f"attention_blocks type: {type(encoder.attention_blocks)}")
        print(f"attention_blocks length: {len(encoder.attention_blocks) if hasattr(encoder.attention_blocks, '__len__') else 'N/A'}")
    
    if hasattr(encoder, 'attention_blocks') and isinstance(encoder.attention_blocks, (list, tuple, torch.nn.modules.container.ModuleList)):
        blocks = encoder.attention_blocks
        num_blocks = len(blocks)
        print(f"Processing {num_blocks} attention blocks")
        for block_index, block in enumerate(blocks):
            # Deeper blocks (towards the output) should receive higher LR
            distance_from_last = (num_blocks - 1 - block_index)
            block_lr = learning_rate / (decay_factor ** distance_from_last) if decay_factor != 0 else learning_rate
            block_params = [p for p in block.parameters() if p.requires_grad]
            for p in block_params:
                used_param_ids.add(id(p))
            if len(block_params) > 0:
                decay_params, no_decay_params = _split_decay_param_groups(model.named_parameters(), block_params)
                if len(decay_params) > 0:
                    param_groups.append({
                        'params': decay_params,
                        'lr': block_lr,
                        'weight_decay': weight_decay,
                    })
                if len(no_decay_params) > 0:
                    param_groups.append({
                        'params': no_decay_params,
                        'lr': block_lr,
                        'weight_decay': 0.0,
                    })
                print(f"  Block {block_index}: lr={block_lr}, params={len(block_params)}")

        # 3) Any remaining encoder params (e.g., embeddings, input stems)
        remaining_encoder_params = []
        for p in encoder.parameters():
            if p.requires_grad and id(p) not in used_param_ids:
                remaining_encoder_params.append(p)
        if len(remaining_encoder_params) > 0:
            # Assign the smallest LR (as if before the first block)
            smallest_lr = learning_rate / (decay_factor ** num_blocks) if decay_factor != 0 else learning_rate
            decay_params, no_decay_params = _split_decay_param_groups(model.named_parameters(), remaining_encoder_params)
            if len(decay_params) > 0:
                param_groups.append({
                    'params': decay_params,
                    'lr': smallest_lr,
                    'weight_decay': weight_decay,
                })
            if len(no_decay_params) > 0:
                param_groups.append({
                    'params': no_decay_params,
                    'lr': smallest_lr,
                    'weight_decay': 0.0,
                })
    else:
        # Fallback: no explicit blocks; treat the entire encoder as a single group
        encoder_params = [p for p in encoder.parameters() if p.requires_grad and id(p) not in used_param_ids]
        if len(encoder_params) > 0:
            decay_params, no_decay_params = _split_decay_param_groups(model.named_parameters(), encoder_params)
            if len(decay_params) > 0:
                param_groups.append({
                    'params': decay_params,
                    'lr': learning_rate,
                    'weight_decay': weight_decay,
                })
            if len(no_decay_params) > 0:
                param_groups.append({
                    'params': no_decay_params,
                    'lr': learning_rate,
                    'weight_decay': 0.0,
                })

    # Build optimizer with parameter groups
    optimizer = optim.AdamW(param_groups)
    print(f"Using layer-wise LR decay with head_multiplier={head_multiplier}, decay_factor={decay_factor}.")
    for gi, g in enumerate(param_groups):
        print(f"Param group {gi}: lr={g['lr']}, params={len(g['params'])}")
else:
    # Standard grouping: exclude biases and norm weights from weight decay
    decay_params, no_decay_params = _split_decay_param_groups(model.named_parameters())
    param_groups = []
    if len(decay_params) > 0:
        param_groups.append({'params': decay_params, 'lr': learning_rate, 'weight_decay': weight_decay})
    if len(no_decay_params) > 0:
        param_groups.append({'params': no_decay_params, 'lr': learning_rate, 'weight_decay': 0.0})
    optimizer = optim.AdamW(param_groups)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Training_epochs)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Training_epochs/2, eta_min=1e-6)
#scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)
# Initialize TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# Initialize variables for checkpoint tracking
best_f1 = 0.0

start_epoch = 0
if resume:
    loaded = load_best_checkpoint_if_exists(model, optimizer, checkpoint_dir)
    if loaded is not None:
        model, optimizer, start_epoch, best_f1 = loaded
        # Try to load scheduler state from best checkpoint as well
        best_path = os.path.join(checkpoint_dir, 'best_f1_score.pth')
        try:
            checkpoint = torch.load(best_path, map_location=device)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Successfully loaded scheduler state from best checkpoint")
            if use_ema and 'ema_state_dict' in checkpoint:
                ema.load_state_dict(checkpoint['ema_state_dict'])
                print("Loaded EMA state from best checkpoint")
        except Exception as e:
            print(f"WARNING: Could not load scheduler/EMA from best checkpoint: {e}")
        print(f"Resuming from epoch {start_epoch} with best F1 {best_f1:.4f}")
    else:
        print("No best checkpoint found; starting from scratch.")
best_f1_score = 0
# Initialize Ignite metrics
confusion_matrix = ConfusionMatrix(num_classes=N_class)
if ignore_index is not None:
    miou_metric = IoU(cm=confusion_matrix, ignore_index=ignore_index)
    # Initialize torchmetrics metrics and move them to the correct device
    f1_metric = F1Score(task='multiclass', num_classes=N_class, average='none', ignore_index=ignore_index).to(device)
else:
    miou_metric = IoU(cm=confusion_matrix)
    # Initialize torchmetrics metrics and move them to the correct device
    f1_metric = F1Score(task='multiclass', num_classes=N_class, average='none').to(device)
best_f1 = 0.0
# ===================== Training Loop =====================
print(f"Starting downstream semantic segmentation training for {Training_epochs} epochs")
for epoch in range(start_epoch, Training_epochs):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    # Training
    with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{Training_epochs}', unit='batch') as pbar:
        for i, data in enumerate(data_loader):
            if use_texture and texture_dir is not None:
                # Texture dataset returns: (geometry_features, labels, texture_sequences, masks, texture_masks)
                geometry_features, labels, texture_sequences, masks, texture_masks, _ = data

                geometry_features = geometry_features.to(device)
                texture_sequences = texture_sequences.to(device)
                labels = labels.to(device)
                if mesh_dir == '/bigwork/nhgnheid/EU_data_V2/train' or mesh_dir == '/bigwork/nhgnheid/EU_data_V3/Train':
                    labels = labels + 1
                masks = masks.to(device)
                texture_masks = texture_masks.to(device)
                optimizer.zero_grad()
                
                # Forward pass - get per-face predictions
                # Clear cache before forward pass to free memory
                torch.cuda.empty_cache()
                
                # Use mixed precision for forward pass
                if use_mixed_precision:
                    with autocast():
                        logits = model(geometry_features, texture_sequences, masks, texture_masks)  # [B, N, F, num_classes]
                else:
                    logits = model(geometry_features, texture_sequences, masks, texture_masks)  # [B, N, F, num_classes]
            else:
                # Geometry-only dataset returns: (batch, labels, masks)
                batch, labels, masks = data
                batch = batch.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                # Forward pass - get per-face predictions with mixed precision
                if use_mixed_precision:
                    with autocast():
                        logits = model(batch, masks)  # [B, N, F, num_classes]
                else:
                    logits = model(batch, masks)  # [B, N, F, num_classes]
            
            # Reshape for loss calculation
            B, N, F, num_classes = logits.shape
            
            # Calculate loss (CrossEntropyLoss will ignore -100 labels)
            if use_mixed_precision:
                with autocast():
                    loss = masked_loss_fn(logits, labels, masks)
            else:
                loss = masked_loss_fn(logits, labels, masks)
            
            # Backward pass with mixed precision
            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
            #    torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update EMA after optimizer step
            if ema is not None:
                ema.update()
            
            pred = logits.reshape(-1, N_class)
            target = labels.reshape(-1)
            mask = masks.view(-1)
            valid_mask = mask == 1  # assuming mask is binary with 1 for valid data points
            pred = pred[valid_mask]
            target = target[valid_mask]
            # Update confusion matrix and IoU metric based on initial output
            confusion_matrix.update((pred, target))
            # Update torchmetrics
            f1_metric.update(pred, target)
            f1_scores = f1_metric.compute()
            # Calculate mean excluding ignore_index if specified
            if ignore_index is not None and ignore_index < N_class:
                class_mask = torch.arange(N_class) != ignore_index
                mean_f1_score = f1_scores[class_mask].mean().item()
            else:
                mean_f1_score = f1_scores.mean().item()
            # Compute overall accuracy using simple correct/total approach
            pred_classes = pred.argmax(dim=1)
            overall_acc = overall_accuracy(pred_classes, target, ignore_index=ignore_index)
            if i % 10 == 9:  # Log every 10 batches
                writer.add_scalar('Loss/train', running_loss / 10, epoch * len(data_loader) + i)
                writer.add_scalar('f1_scores/train', f1_scores.mean().item(), epoch * len(data_loader) + i)
                running_loss = 0.0
                total_miou = 0.0
                total_batches = 0.0
            # Update tqdm bar and print statement
            pbar.set_postfix({'Loss': loss.item(), 'mean_f1_score': mean_f1_score, 'Overall Accuracy': overall_acc})
            pbar.update(1)
            print(f"Epoch {epoch + 1}, f1_scores: {f1_scores}")
        # Log torchmetrics results at the end of each epoch
        miou = miou_metric.compute()
        f1_scores = f1_metric.compute()
        # Calculate mean excluding ignore_index if specified
        if ignore_index is not None and ignore_index < N_class:
            class_mask = torch.arange(N_class) != ignore_index
            mean_f1_score = f1_scores[class_mask].mean().item()
        else:
            mean_f1_score = f1_scores.mean().item()
        # For epoch-level accuracy, we need to collect all predictions and targets
        # Since we already have them from the training loop, we'll compute it here
        # Note: This is a simplified approach - in practice you might want to collect all preds/targets during training
        writer.add_scalar('F1/train', mean_f1_score, epoch)
        writer.add_scalar('Accuracy/train', overall_acc, epoch)
        print(f"Epoch {epoch + 1}, f1_scores: {f1_scores}, Mean F1 Score: {mean_f1_score}, Overall accuracy: {overall_acc}, miou: {miou}")
        # Evaluate on validation set
        val_loss, val_mean_f1_score, val_overall_accuracy, val_miou, val_f1_scores, val_confusion_matrix = evaluate(model, val_data_loader, ema)
        scheduler.step()
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/val', val_mean_f1_score, epoch)
        writer.add_scalar('Accuracy/val', val_overall_accuracy, epoch)
        print(f"Validation - Epoch {epoch + 1}, Loss: {val_loss}, F1 Score: {val_f1_scores}, Mean F1 Score: {val_mean_f1_score}, Overall Accuracy: {val_overall_accuracy}, miou: {val_miou}")
        # Save only the best model when validation F1 improves
        if val_mean_f1_score > best_f1_score:
            best_f1_score = val_mean_f1_score
            if ema is not None:
                save_best_checkpoint(model, optimizer, epoch, best_f1_score, checkpoint_dir, ema_state_dict=ema.state_dict(), scheduler=scheduler)
            else:
                save_best_checkpoint(model, optimizer, epoch, best_f1_score, checkpoint_dir, scheduler=scheduler)
            # Write metrics to evaluation.txt when improvement happens
            try:
                evaluation_txt_path = os.path.join(checkpoint_dir, 'evaluation.txt')
                with open(evaluation_txt_path, 'w') as f:
                    # Classwise F1 scores at the top
                    val_f1_scores_list = val_f1_scores.detach().cpu().tolist() if torch.is_tensor(val_f1_scores) else list(val_f1_scores)
                    f.write("classwise_f1_scores:\n")
                    for cls_idx, score in enumerate(val_f1_scores_list):
                        f.write(f"  class_{cls_idx}: {score}\n")
                    f.write("\n")
                    # Remaining aggregated metrics
                    f.write(f"epoch: {epoch}\n")
                    f.write(f"val_loss: {val_loss}\n")
                    f.write(f"val_mean_f1: {val_mean_f1_score}\n")
                    f.write(f"val_overall_accuracy: {val_overall_accuracy}\n")
                    f.write(f"val_miou: {val_miou}\n")
            except Exception as e:
                print(f"WARNING: Failed to write evaluation.txt: {e}")
            print(f"New best validation F1 score: {best_f1_score}. Best checkpoint saved (best_f1_score.pth).")
            print(f"Confusion Matrix at Best Validation:\n{val_confusion_matrix.compute()}")
        # Reset metrics
        confusion_matrix.reset()
        f1_metric.reset()

print('Training complete')
# Close the TensorBoard writer
writer.close()
