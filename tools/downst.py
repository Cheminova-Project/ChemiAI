import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== Downstream Classification Model =====================
class DownstreamClassifier(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 num_classes: int,
                 embedding_dim: int,
                 dropout: float = 0.1,
                 shallow: bool = True,
                 num_unfrozen_encoder_layers: int = 0):
        """
        Args:
            encoder: Pretrained feature encoder (frozen).
            num_classes: Number of output classes.
            embedding_dim: Dimension of the encoder embeddings.
            dropout: Dropout probability for robust head.
            shallow: If True, uses a single linear layer; otherwise uses a deeper, robust head.
            num_unfrozen_encoder_layers: Number of encoder layers to unfreeze.
        """
        super().__init__()
        self.encoder = encoder
        self.shallow = shallow

        # Freeze all encoder parameters by default
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks
        if hasattr(self.encoder, "attention_blocks") and num_unfrozen_encoder_layers > 0:
            blocks = self.encoder.attention_blocks
            for block in blocks[-num_unfrozen_encoder_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Choose classification head based on mode
        if self.shallow:
            # Simple linear classifier
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            # Robust multi-layer head
            hidden1 = embedding_dim // 2
            hidden2 = embedding_dim // 4
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden1),
                nn.LayerNorm(hidden1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.LayerNorm(hidden2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, num_classes)
            )

        # Initialize weights of classifier head
        self.classifier.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input_tensor: torch.Tensor, masks_x: torch.Tensor) -> torch.Tensor:
        # Extract embeddings from the encoder: [B, N, F, embedding_dim]
        embeddings = self.encoder(input_tensor, masks_x)
        # Classify: shallow linear handles last dim; robust sequential handles it too
        logits = self.classifier(embeddings)
        return logits
