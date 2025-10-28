import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tools.model_base import MultiHeadAttention,TransformerBlock
import math
import numpy as np
from tools.model_base import nomeformer

# ===================== Texture Processing Components =====================

class TexturePatchEmbedding(nn.Module):
    """
    Processes texture pixel sequences using local attention to summarize pixels into CLS tokens.
    Each face's pixels are processed independently to create a single embedding per face.
    """
    def __init__(self, max_pixels=256, in_channels=3, embed_dim=128, num_heads=2, dropout=0.1):
        super(TexturePatchEmbedding, self).__init__()
        self.max_pixels = max_pixels
        self.embed_dim = embed_dim
        
        # Project RGB channels to embedding dimension
        self.pixel_projection = nn.Linear(in_channels, embed_dim)  # RGB -> embed_dim
        
        # Local attention block to summarize pixels into CLS token
        self.local_attention = TransformerBlock(
            embedding_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=embed_dim,
            dropout=dropout,
            summary_mode='cls',  # Use CLS token to summarize pixels
            use_relative_positional_encoding=False
        )
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, texture_sequences, texture_masks):
        """
        Args:
            texture_sequences: [B, Clusters, Faces, max_pixels, C] - texture pixel sequences
            texture_masks: [B, Clusters, Faces] - validity masks for textures
        Returns:
            embeddings: [B, Clusters, Faces, embed_dim] - texture embeddings (CLS tokens)
        """

        B, Clusters, Faces, max_pixels, C = texture_sequences.shape

        # # Memory optimization: process in chunks if faces > 1000
        # if Faces > 1000:
        #     return self._forward_chunked(texture_sequences, texture_masks)
        
        # Reshape to process all faces independently: [B*Clusters, Faces, max_pixels, C]
        x = texture_sequences.view(B * Clusters, Faces, max_pixels, C)
        texture_masks = texture_masks.view(B * Clusters, Faces, max_pixels)

        # Project RGB pixels to embedding space: [B*Clusters, Faces, max_pixels, embed_dim]
        x = self.pixel_projection(x.float())  # Ensure float32
        
        # Create pixel masks - assume all pixels are valid for now
        # In practice, you might want to use actual pixel validity masks
        
        # Apply local attention to get CLS token for each face with gradient checkpointing
        # # This will summarize all pixels of each face into a single CLS token
        # if self.training:
        #     # Use gradient checkpointing during training to save memory
        #     _, cls_tokens = checkpoint(self.local_attention, x, pixel_masks, use_reentrant=False)
        # else:
            # _, 
        x,cls_tokens = self.local_attention(x, texture_masks)  # cls_tokens: [B*Clusters, Faces, 1, embed_dim]
        
        # Remove the singleton dimension and reshape back
        cls_tokens = cls_tokens.squeeze(-2)  # [B*Clusters, Faces, embed_dim]
        embeddings = cls_tokens.view(B, Clusters, Faces, self.embed_dim)
        
        # # Apply texture masks - zero out invalid textures
        # embeddings = embeddings * texture_masks.unsqueeze(-1).float()
        
        # Final normalization
        embeddings = self.norm(embeddings)
        
        return embeddings
    
    def _forward_chunked(self, texture_sequences, texture_masks, chunk_size=500):
        """
        Process texture sequences in chunks to save memory.
        """
        B, Clusters, Faces, max_pixels, C = texture_sequences.shape
        
        # Process in chunks along the Faces dimension
        embeddings_list = []
        for start_face in range(0, Faces, chunk_size):
            end_face = min(start_face + chunk_size, Faces)
            
            # Extract chunk
            chunk_sequences = texture_sequences[:, :, start_face:end_face]  # [B, Clusters, chunk_faces, max_pixels, C]
            chunk_masks = texture_masks[:, :, start_face:end_face]  # [B, Clusters, chunk_faces]
            
            # Process chunk
            chunk_B, chunk_Clusters, chunk_Faces, chunk_pixels, chunk_C = chunk_sequences.shape
            x = chunk_sequences.view(chunk_B * chunk_Clusters, chunk_Faces, chunk_pixels, chunk_C)
            x = self.pixel_projection(x.float())
            
            pixel_masks = torch.ones(chunk_B * chunk_Clusters, chunk_Faces, chunk_pixels, device=x.device, dtype=torch.bool)
            
            if self.training:
                _, cls_tokens = checkpoint(self.local_attention, x, pixel_masks, use_reentrant=False)
            else:
                _, cls_tokens = self.local_attention(x, pixel_masks)
            
            cls_tokens = cls_tokens.squeeze(-2)  # [B*Clusters, chunk_Faces, embed_dim]
            chunk_embeddings = cls_tokens.view(chunk_B, chunk_Clusters, chunk_Faces, self.embed_dim)
            
            # Apply masks and normalize
            chunk_embeddings = chunk_embeddings * chunk_masks.unsqueeze(-1).float()
            chunk_embeddings = self.norm(chunk_embeddings)
            
            embeddings_list.append(chunk_embeddings)
        
        # Concatenate all chunks
        embeddings = torch.cat(embeddings_list, dim=2)  # Concatenate along Faces dimension
        return embeddings

class TextureGeometryFusion(nn.Module):
    """
    Fuses texture and geometry embeddings.
    """
    def __init__(self, geometry_dim, texture_dim, output_dim, fusion_method='concat'):
        super(TextureGeometryFusion, self).__init__()
        self.fusion_method = fusion_method
        
        if fusion_method == 'gated':
            # Gated fusion
            self.gate = nn.Sequential(
                nn.Linear(geometry_dim + texture_dim, output_dim),
                nn.Sigmoid()
            )
            self.geometry_proj = nn.Linear(geometry_dim, output_dim)
            self.texture_proj = nn.Linear(texture_dim, output_dim)
        elif fusion_method == 'concat':
            # Concatenation fusion
            self.geometry_proj = nn.Linear(geometry_dim, output_dim)
            self.texture_proj = nn.Linear(texture_dim, output_dim)
            print('fusion method is concat')
            self.fusion_proj = nn.Linear(output_dim, output_dim)
        elif fusion_method == 'add':
            # Addition fusion (requires same dimensions)
            assert geometry_dim == texture_dim, "Add fusion requires same dimensions"
            self.proj = nn.Linear(geometry_dim, output_dim)
        
    def forward(self, geometry_features, texture_emb):
        """
        Args:
            geometry_features: [B, Clusters, Faces, geometry_dim] - raw geometry features
            texture_emb: [B, Clusters, Faces, texture_dim] - texture embeddings
        Returns:
            fused_features: [B, Clusters, Faces, output_dim] - combined features for nomeformer
        """
        B, Clusters, Faces, D = geometry_features.shape
        
        # Both inputs have same spatial dimensions
        # geometry_features: [B, Clusters, Faces, geometry_dim]
        # texture_emb: [B, Clusters, Faces, texture_dim]
        
        # Ensure both tensors are float32
        geometry_features = geometry_features.float()
        texture_emb = texture_emb.float()
        
        if self.fusion_method == 'gated':
            # Gated fusion
            combined = torch.cat([geometry_features, texture_emb], dim=-1)
            gate = self.gate(combined)
            geometry_proj = self.geometry_proj(geometry_features)
            texture_proj = self.texture_proj(texture_emb)
            fused = gate * geometry_proj + (1 - gate) * texture_proj
        elif self.fusion_method == 'concat':
            # Concatenation fusion
            print('fusion method is concat')
            geometry_proj = self.geometry_proj(geometry_features)
            texture_proj = self.texture_proj(texture_emb)
            print(f"Geometry projection shape: {geometry_proj.shape}")
            print(f"Texture projection shape: {texture_proj.shape}")
            combined = torch.cat([geometry_proj, geometry_proj], dim=-1)
            fused = self.fusion_proj(geometry_proj)
        elif self.fusion_method == 'add':
            # Addition fusion
            proj_geom = self.proj(geometry_features)
            proj_text = self.proj(texture_emb)
            fused = proj_geom + proj_text
            
        return fused

# ===================== Integrated Model =====================

class IntegratedTextureGeometryModel(nn.Module):
    """
    Integrated model that processes both texture and geometry features.
    """
    def __init__(self, 
                 geometry_feature_dim,
                 embedding_dim=256,
                 texture_embed_dim=128,
                 num_heads=4,
                 num_attention_blocks=4,
                 dropout=0.1,
                 summary_mode='cls',
                 use_hierarchical=False,
                 fourier=False,
                 relative_positional_encoding=False,
                 fusion_method='concat',
                 max_texture_pixels=256):
        super(IntegratedTextureGeometryModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.texture_embed_dim = texture_embed_dim
        self.max_texture_pixels = max_texture_pixels
        
        # Geometry encoder (existing nomeformer)
        self.encoder = nomeformer(
            feature_dim=embedding_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_attention_blocks=num_attention_blocks,
            dropout=dropout,
            summary_mode=summary_mode,
            use_hierarchical=use_hierarchical,
            fourier=fourier,
            relative_positional_encoding=relative_positional_encoding
        )
        
        # Texture processor
        self.texture_embedding = TexturePatchEmbedding(
            max_pixels=max_texture_pixels,
            in_channels=3,
            embed_dim=texture_embed_dim
        )
        
        # Fusion module - fuses raw geometry features with texture embeddings
        self.fusion = TextureGeometryFusion(
            geometry_dim=geometry_feature_dim,  # Raw geometry features
            texture_dim=texture_embed_dim,
            output_dim=embedding_dim,  # Output ready for nomeformer
            fusion_method=fusion_method
        )
        
        # Final processing layers
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, geometry_features, texture_sequences, masks, texture_masks):
        """
        Args:
            geometry_features: [B, Clusters, Faces, geometry_feature_dim] - geometry features
            texture_sequences: [B, Clusters, Faces, max_pixels, C] - texture pixel sequences
            masks: [B, Clusters, Faces] - geometry masks
            texture_masks: [B, Clusters, Faces] - texture validity masks
        Returns:
            output: [B, Clusters, Faces, T, embedding_dim] - joint representation from nomeformer
        """
        # Step 1: Process texture sequences to get face-level embeddings
        texture_emb = self.texture_embedding(texture_sequences, texture_masks)  # [B, Clusters, Faces, texture_embed_dim]

        # Step 2: Fuse texture embeddings with raw geometry features
        fused_features = self.fusion(geometry_features, texture_emb)  # [B, Clusters, Faces, embedding_dim]

        # Step 3: Pass combined features through nomeformer to learn joint representation
        output = self.encoder(fused_features, masks)  # [B, Clusters, Faces, T, embedding_dim]
        
        # Final processing
        output = self.final_norm(output)
        output = self.output_proj(output)
        
        return output

# ===================== Downstream Classifier for Integrated Model =====================

class IntegratedDownstreamClassifier(nn.Module):
    """
    Downstream classifier for the integrated texture-geometry model.
    """
    def __init__(self, integrated_encoder, num_classes, embedding_dim, dropout=0.1, 
                 freeze_encoder_layers=0, fusion_method='gated'):
        super(IntegratedDownstreamClassifier, self).__init__()
        
        self.encoder = integrated_encoder
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Freeze encoder layers if specified
        if freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
    def _freeze_encoder_layers(self, num_layers):
        """Freeze specified number of encoder layers."""
        if hasattr(self.encoder, 'geometry_encoder') and hasattr(self.encoder.geometry_encoder, 'attention_blocks'):
            blocks = self.encoder.geometry_encoder.attention_blocks
            num_blocks = len(blocks)
            layers_to_freeze = min(num_layers, num_blocks)
            
            for i in range(layers_to_freeze):
                for param in blocks[i].parameters():
                    param.requires_grad = False
                print(f"Froze geometry encoder layer {i}")
        
        # Also freeze texture embedding if specified
        if num_layers >= num_blocks:
            for param in self.encoder.texture_embedding.parameters():
                param.requires_grad = False
            print("Froze texture embedding layers")
    
    def forward(self, geometry_features, texture_sequences, masks, texture_masks):
        """
        Args:
            geometry_features: [B, Clusters, Faces, geometry_feature_dim]
            texture_sequences: [B, Clusters, Faces, max_pixels, C]
            masks: [B, Clusters, Faces]
            texture_masks: [B, Clusters, Faces]
        Returns:
            logits: [B, Clusters, Faces, T, num_classes]
        """
        # Get fused embeddings from integrated encoder
        embeddings = self.encoder(geometry_features, texture_sequences, masks, texture_masks)
        
        # Classify - embeddings shape: [B, Clusters, Faces, T, embedding_dim]
        logits = self.classifier(embeddings)
        
        return logits

# ===================== Helper Functions =====================

def create_integrated_model(geometry_feature_dim,
                           embedding_dim=256,
                           texture_embed_dim=128,
                           num_heads=4,
                           num_attention_blocks=4,
                           dropout=0.1,
                           summary_mode='cls',
                           use_hierarchical=False,
                           fourier=False,
                           relative_positional_encoding=False,
                           fusion_method='concat',
                           max_texture_pixels=256):
    """
    Create an integrated texture-geometry model.
    """
    encoder = IntegratedTextureGeometryModel(
        geometry_feature_dim=geometry_feature_dim,
        embedding_dim=embedding_dim,
        texture_embed_dim=texture_embed_dim,
        num_heads=num_heads,
        num_attention_blocks=num_attention_blocks,
        dropout=dropout,
        summary_mode=summary_mode,
        use_hierarchical=use_hierarchical,
        fourier=fourier,
        relative_positional_encoding=relative_positional_encoding,
        fusion_method=fusion_method,
        max_texture_pixels=max_texture_pixels
    )
    
    return encoder

def load_pretrained_integrated_model(checkpoint_path, geometry_feature_dim, **model_kwargs):
    """
    Load a pretrained integrated model from checkpoint.
    """
    model = create_integrated_model(geometry_feature_dim, **model_kwargs)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, using random initialization")
    
    return model



