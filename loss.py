import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.distributed import AllReduce


def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss
class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with support for masking and optional class weights.
    Args:
        class_weights (Tensor, optional): Class weights for imbalanced data. Shape: (num_classes,)
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient.
    """
    def __init__(self, class_weights=None ,ignore_index = None):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights  # Class weights tensor
        self.ignore_index = ignore_index  # Index to ignore in the target

    def forward(self, prediction, target, mask):
        """
        Computes the masked cross-entropy loss.
        Args:
            prediction (Tensor): Predicted logits of shape (B, ..., num_classes) or (N, num_classes).
            target (Tensor): Ground truth labels of shape (B, ...) or (N,).
            mask (Tensor): Boolean or binary mask of shape (B, ...) or (N,) indicating valid entries (1=use, 0=ignore).
        Returns:
            loss (Tensor): Scalar tensor representing the mean cross-entropy loss over valid entries.
        """
        # Flatten the tensors to ensure they are 1-dimensional
        prediction = prediction.view(-1, prediction.size(-1))
        target = target.view(-1)
        mask = mask.view(-1)
        
        # Apply the mask to filter out the relevant elements
        # Ensure mask is boolean for correct advanced indexing
        if mask.dtype != torch.bool:
            mask = mask != 0
        prediction = prediction[mask]
        target = target[mask]

        # Number of classes inferred from the prediction size
        n_classes = prediction.size(-1)
        
        # Ensure all non-ignored target values are within the valid range
        if self.ignore_index is not None:
            valid_targets = target[target != self.ignore_index]
        else:
            valid_targets = target
        if valid_targets.numel() > 0 and not torch.all((valid_targets >= 0) & (valid_targets < n_classes)):
            print(f"Invalid target values detected. n_classes: {n_classes}")
            print(f"Target values: {target}")
            print(f"Target unique values: {torch.unique(target)}")
            print(f"Target min/max: {target.min()}/{target.max()}")
            print(f"Valid targets: {valid_targets}")
            print(f"Valid targets unique: {torch.unique(valid_targets)}")
            print(f"Valid targets min/max: {valid_targets.min()}/{valid_targets.max()}")
            raise ValueError(f"Target values should be in range [0, {n_classes-1}] (excluding ignore_index={self.ignore_index})")

        # Apply class weights (if provided)
        if self.class_weights is not None:
            class_weights = self.class_weights.to(prediction.device)  # Ensure weights are on the correct device
        else:
            class_weights = None
        
        # Calculate the cross entropy loss with class weights and ignore_index
        # Cast to long as required by CrossEntropyLoss
        target = target.long()
        if self.ignore_index is None:
            loss = F.cross_entropy(prediction, target, weight=class_weights, reduction='mean')
        else:
            loss = F.cross_entropy(prediction, target, weight=class_weights, reduction='mean', ignore_index=self.ignore_index)
        
        return loss

class IJEPALoss(nn.Module):
    def __init__(self):
        super(IJEPALoss, self).__init__()
        # You can add any initialization parameters here if needed
        # For example, if you want to make the loss configurable
        self.reduction = 'mean'  # or 'sum' or 'none'

    def forward(self, z, h):
        """
        Args:
            z: Predictor output (predicted features)
            h: Target encoder output (target features)
        Returns:
            loss: The computed loss value
        """
        # Compute smooth L1 loss (Huber loss)
        loss = F.smooth_l1_loss(z, h, reduction=self.reduction)
        
        # Apply all-reduce for distributed training
        loss = AllReduce.apply(loss)
        
        return loss

    def __call__(self, z, h):
        return self.forward(z, h)
