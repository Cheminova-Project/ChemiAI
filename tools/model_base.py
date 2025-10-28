import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ===================== Weight Initialization =====================
def jepa_init_weights(m):
    """Improved JEPA-style initialization with better scaling."""
    if isinstance(m, nn.Linear):
        # Use fan-in scaling for better gradient flow
        fan_in = m.weight.size(1)
        std = math.sqrt(2.0 / fan_in) * 0.5  # Scale down for stability
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)

def init_attention_weights(m):
    """Specialized initialization for attention layers."""
    if isinstance(m, nn.Linear):
        # Smaller initialization for attention to prevent saturation
        nn.init.xavier_uniform_(m.weight, gain=0.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# ===================== Multi-Head Attention Block =====================
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        super(MultiHeadAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert self.embedding_dim % self.num_heads == 0, f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embedding_dim // self.num_heads
        self.dropout = dropout
        # 'cls' to use a learnable class token, 'avg' to use masked average pooling
        self.summary_mode = summary_mode
        self.use_relative_positional_encoding = use_relative_positional_encoding

        # Linear projections for query, key, and value
        self.query = nn.Linear(embedding_dim, embedding_dim).float()
        self.key = nn.Linear(embedding_dim, embedding_dim).float()
        self.value = nn.Linear(embedding_dim, embedding_dim).float()
        self.norm = nn.LayerNorm(embedding_dim)

        # Learnable token embedding (used when summary_mode == 'cls')
        self.token_embedding = nn.Parameter(torch.randn(1, embedding_dim))

        # RoPE-3D uses positions directly; no learned MLPs



    def forward(self, padded_batches, masks, sum_token_in=None, positions_3d=None):
        # Shapes: padded_batches [B, N, T, D], masks [B, N, T]
        B, N, T, D = padded_batches.shape
        H = self.num_heads
        Hd = self.head_dim
        
        if self.summary_mode == 'cls':
            # Use passed summary token if provided, otherwise initialize from learnable embedding
            cls_token = sum_token_in if sum_token_in is not None else self.token_embedding.expand(B, N, 1, D)
            x = torch.cat((cls_token, padded_batches), dim=-2)  # [B, N, T+1, D]
            L = T + 1

            q = self.query(x.float())
            k = self.key(x.float())
            v = self.value(x.float())

            # Build float attention mask (large_neg = masked/ignore per SDPA semantics)
            prep = torch.ones(B, N, 1, device=masks.device, dtype=masks.dtype)
            key_keep = torch.cat((prep, masks), dim=-1)  # [B, N, L] 1=valid, 0=masked
            large_neg = torch.finfo(torch.float32).min
            # Create 2D attention mask: [B*N, 1, L, L]
            key_mask_1d = torch.where(key_keep == 0, large_neg, 0.0)  # [B, N, L]
            key_mask = key_mask_1d.view(B * N, 1, 1, L).expand(-1, -1, L, -1)  # [B*N, 1, L, L]

            # Reshape to heads: [B*N, H, L, Hd]
            q = q.view(B * N, L, H, Hd).permute(0, 2, 1, 3)
            k = k.view(B * N, L, H, Hd).permute(0, 2, 1, 3)
            v = v.view(B * N, L, H, Hd).permute(0, 2, 1, 3)
            if not self.use_relative_positional_encoding or positions_3d is None:
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=self.dropout)
            else:
                # Apply 3D RoPE to q and k using centroid positions (prepend CLS)
                pos = positions_3d[:, :, :T, :]
                pos = torch.cat((torch.zeros(B, N, 1, 3, device=pos.device, dtype=pos.dtype), pos), dim=-2)
                pos = pos.view(B * N, L, 3)  # [B*N, L, 3]
                # Split head dim into three even chunks for x, y, z
                chunk = (Hd // 6) * 2  # largest even <= Hd/3
                cx = cy = cz = chunk
                def apply_rope(t, coord, start, length):
                    if length <= 0:
                        return t
                    pairs = max(length // 2, 1)
                    inv_freq = 1.0 / (10000 ** (torch.arange(0, pairs, device=t.device, dtype=t.dtype) / pairs))
                    ang = coord.unsqueeze(-1) * inv_freq  # [B*N, L, pairs]
                    cos = ang.cos().unsqueeze(1)
                    sin = ang.sin().unsqueeze(1)
                    seg = t[:, :, :, start:start+length]
                    even, odd = seg[..., 0::2], seg[..., 1::2]
                    seg_rot_even = even * cos - odd * sin
                    seg_rot_odd  = even * sin + odd * cos
                    seg_rot = torch.stack((seg_rot_even, seg_rot_odd), dim=-1).reshape_as(seg)
                    t[:, :, :, start:start+length] = seg_rot
                    return t
                q = apply_rope(q, pos[..., 0], 0, cx)
                k = apply_rope(k, pos[..., 0], 0, cx)
                q = apply_rope(q, pos[..., 1], cx, cy)
                k = apply_rope(k, pos[..., 1], cx, cy)
                q = apply_rope(q, pos[..., 2], cx+cy, cz)
                k = apply_rope(k, pos[..., 2], cx+cy, cz)
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=self.dropout)
            # Merge heads: [B*N, L, D]
            attn = attn.permute(0, 2, 1, 3).contiguous().view(B, N, L, D)

            # Separate class token and residual
            sum_token = attn[:, :, :1, :]
            sum_token = self.norm(sum_token + x[:, :, :1, :])
            out = attn[:, :, 1:, :] + x[:, :, 1:, :]
            return out, sum_token
        else:
            x = padded_batches
            L = T

            q = self.query(x.float())
            k = self.key(x.float())
            v = self.value(x.float())

            # Mask: [B, N, T] -> [B*N, 1, L, L] (large_neg=masked)
            large_neg = torch.finfo(torch.float32).min
            key_mask_1d = torch.where(masks == 0, large_neg, 0.0)  # [B, N, L]
            key_mask = key_mask_1d.view(B * N, 1, 1, L).expand(-1, -1, L, -1)  # [B*N, 1, L, L]

            q = q.view(B * N, L, H, Hd).permute(0, 2, 1, 3)
            k = k.view(B * N, L, H, Hd).permute(0, 2, 1, 3)
            v = v.view(B * N, L, H, Hd).permute(0, 2, 1, 3)

            if not self.use_relative_positional_encoding or positions_3d is None:
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=self.dropout)
            else:
                # Apply 3D RoPE to q and k using centroid positions (no CLS)
                pos = positions_3d.view(B * N, L, 3)
                chunk = (Hd // 6) * 2
                cx = cy = cz = chunk
                def apply_rope(t, coord, start, length):
                    if length <= 0:
                        return t
                    pairs = max(length // 2, 1)
                    inv_freq = 1.0 / (10000 ** (torch.arange(0, pairs, device=t.device, dtype=t.dtype) / pairs))
                    ang = coord.unsqueeze(-1) * inv_freq  # [B*N, L, pairs]
                    cos = ang.cos().unsqueeze(1)
                    sin = ang.sin().unsqueeze(1)
                    seg = t[:, :, :, start:start+length]
                    even, odd = seg[..., 0::2], seg[..., 1::2]
                    seg_rot_even = even * cos - odd * sin
                    seg_rot_odd  = even * sin + odd * cos
                    seg_rot = torch.stack((seg_rot_even, seg_rot_odd), dim=-1).reshape_as(seg)
                    t[:, :, :, start:start+length] = seg_rot
                    return t
                q = apply_rope(q, pos[..., 0], 0, cx)
                k = apply_rope(k, pos[..., 0], 0, cx)
                q = apply_rope(q, pos[..., 1], cx, cy)
                k = apply_rope(k, pos[..., 1], cx, cy)
                q = apply_rope(q, pos[..., 2], cx+cy, cz)
                k = apply_rope(k, pos[..., 2], cx+cy, cz)
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=self.dropout)

            attn = attn.permute(0, 2, 1, 3).contiguous().view(B, N, L, D)
            out = attn + x
            sum_token = torch.zeros_like(out[:, :, :1, :])
            return out, sum_token

# ===================== Global Multi-Head Attention Block =====================
class GlobalMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1, summary_mode='avg', use_relative_positional_encoding=False):
        super(GlobalMultiHeadAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert self.embedding_dim % self.num_heads == 0, f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embedding_dim // self.num_heads
        self.dropout = dropout
        # 'cls' uses provided sum_token; 'avg' recomputes masked average as summary token
        self.summary_mode = summary_mode
        self.use_relative_positional_encoding = use_relative_positional_encoding

        # Linear projections for query, key, and value
        self.query = nn.Linear(embedding_dim, embedding_dim).float()
        self.key = nn.Linear(embedding_dim, embedding_dim).float()
        self.value = nn.Linear(embedding_dim, embedding_dim).float()
        self.norm = nn.LayerNorm(embedding_dim)

        # Learnable token embedding
        self.token_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        # RoPE-3D uses positions directly; no learned MLPs



    def forward(self, padded_batches, sum_token, masks, positions_3d=None):
        # padded_batches: [B, N, T, D], sum_token: [B, N, 1, D], masks: [B, N, T]
        B, N, T, D = padded_batches.shape
        H = self.num_heads
        Hd = self.head_dim

        # Choose summary token for global attention
        if self.summary_mode == 'avg':
            weights = masks.unsqueeze(-1).to(padded_batches.dtype)
            summed = (padded_batches * weights).sum(dim=-2, keepdim=True)
            denom = weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
            summary_token = summed / denom  # [B, N, 1, D]
        else:
            summary_token = sum_token

        # Compute Q from all tokens, K/V from summary tokens
        q = self.query(padded_batches.float())      # [B, N, T, D]
        k = self.key(summary_token.float())         # [B, N, 1, D]
        v = self.value(summary_token.float())       # [B, N, 1, D]

        # Reshape to heads for FlashAttention
        q = q.view(B * N, T, H, Hd).permute(0, 2, 1, 3)   # [B*N, H, T, Hd]
        k = k.view(B * N, 1, H, Hd).permute(0, 2, 1, 3)   # [B*N, H, 1, Hd]
        v = v.view(B * N, 1, H, Hd).permute(0, 2, 1, 3)   # [B*N, H, 1, Hd]
        # Create attention mask for global attention
        # masks: [B, N, T] -> [B*N, 1, T, 1] (large_neg=masked)
        large_neg = torch.finfo(torch.float32).min
        attn_mask = torch.where(masks == 0, large_neg, 0.0).view(B * N, 1, T, 1)
        # Perform attention with proper masking
        if not self.use_relative_positional_encoding or positions_3d is None:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)
        else:
            # 3D RoPE for global attention: rotate q with per-token coords; k with summary coord
            pos = positions_3d  # [B, N, T, 3]
            # summary coord when avg
            if self.summary_mode == 'avg':
                weights = masks.unsqueeze(-1).to(pos.dtype)
                pos_sum = (pos * weights).sum(dim=-2, keepdim=True)
                denom = weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
                pos_summary = pos_sum / denom  # [B, N, 1, 3]
            else:
                pos_summary = pos[:, :, :1, :]
            pos_q = pos.view(B * N, T, 3)
            pos_k = pos_summary.view(B * N, 1, 3)
            # rotate q (T tokens) and k (1 token) with same scheme
            chunk = (Hd // 6) * 2
            cx = cy = cz = chunk
            def apply_rope(t, coord, start, length):
                if length <= 0:
                    return t
                pairs = max(length // 2, 1)
                inv_freq = 1.0 / (10000 ** (torch.arange(0, pairs, device=t.device, dtype=t.dtype) / pairs))
                ang = coord.unsqueeze(-1) * inv_freq  # [B*N, L, pairs]
                cos = ang.cos().unsqueeze(1)
                sin = ang.sin().unsqueeze(1)
                seg = t[:, :, :, start:start+length]
                even, odd = seg[..., 0::2], seg[..., 1::2]
                seg_rot_even = even * cos - odd * sin
                seg_rot_odd  = even * sin + odd * cos
                seg_rot = torch.stack((seg_rot_even, seg_rot_odd), dim=-1).reshape_as(seg)
                t[:, :, :, start:start+length] = seg_rot
                return t
            q = apply_rope(q, pos_q[..., 0], 0, cx)
            k = apply_rope(k, pos_k[..., 0], 0, cx)
            q = apply_rope(q, pos_q[..., 1], cx, cy)
            k = apply_rope(k, pos_k[..., 1], cx, cy)
            q = apply_rope(q, pos_q[..., 2], cx+cy, cz)
            k = apply_rope(k, pos_k[..., 2], cx+cy, cz)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)

        # Merge heads and restore shape
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, N, T, D)

        out = attn + padded_batches
        return out

# ===================== Feed Forward Block =====================
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)  # Add dropout layer
        self.linear2 = nn.Linear(ff_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)  # Apply dropout after the first linear layer
        x = self.linear2(x)
        x = self.norm(x + residual)  # Residual connection + normalization
        return x

# ===================== Transformer Block =====================
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout=dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)

    def forward(self, x, masks, sum_token_in=None, positions_3d=None):
        x, sum_token = self.attention(x, masks, sum_token_in=sum_token_in, positions_3d=positions_3d)
        x = self.feed_forward(x)
        return x, sum_token

# ===================== Global Transformer Block =====================
class GlobalTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        super(GlobalTransformerBlock, self).__init__()
        self.attention = GlobalMultiHeadAttention(embedding_dim, num_heads, dropout=dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)

    def forward(self, x, sum_token, masks, positions_3d=None):
        x = self.attention(x, sum_token, masks, positions_3d=positions_3d)
        x = self.feed_forward(x)
        return x

# ===================== Global-Only Multi-Head Attention Block =====================
class GlobalOnlyMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1, summary_mode='avg', use_gating=False, use_relative_positional_encoding=False):
        super(GlobalOnlyMultiHeadAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert self.embedding_dim % self.num_heads == 0, f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embedding_dim // self.num_heads
        self.dropout = dropout
        self.summary_mode = summary_mode
        self.use_gating = use_gating
        self.use_relative_positional_encoding = use_relative_positional_encoding

        # Linear projections for query, key, and value
        self.query = nn.Linear(embedding_dim, embedding_dim).float()
        self.key = nn.Linear(embedding_dim, embedding_dim).float()
        self.value = nn.Linear(embedding_dim, embedding_dim).float()
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Learnable gating mechanism for combining local and global information
        self.gate_proj = nn.Linear(embedding_dim * 2, embedding_dim).float()
        self.gate_norm = nn.LayerNorm(embedding_dim)

        # RoPE-3D uses positions directly; no learned MLPs

    def forward(self, padded_batches, sum_token, masks, positions_3d=None):
        """
        Global-only attention where only summary tokens (global tokens) interact with each other.
        The individual tokens within clusters don't participate in this attention.
        """
        # Choose summary token for global attention
        if self.summary_mode == 'avg':
            # masks: [B, N, T]
            weights = masks.unsqueeze(-1).to(padded_batches.dtype)
            summed = (padded_batches * weights).sum(dim=-2, keepdim=True)
            denom = weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
            summary_token = summed / denom  # [B, N, 1, D]
        else:
            summary_token = sum_token

        # Only summary tokens participate in attention: Q/K/V from summary tokens
        q = self.query(summary_token.float())  # [B, N, 1, D]
        k = self.key(summary_token.float())    # [B, N, 1, D]
        v = self.value(summary_token.float())  # [B, N, 1, D]
        B, N, Lg, D = q.shape  # Lg = 1
        H = self.num_heads
        Hd = self.head_dim
        
        # Reshape to [B, N, H, Hd] for multi-head attention
        q = q.squeeze(2).view(B, N, H, Hd).permute(0, 2, 1, 3)  # [B, H, N, Hd]
        k = k.squeeze(2).view(B, N, H, Hd).permute(0, 2, 1, 3)  # [B, H, N, Hd]
        v = v.squeeze(2).view(B, N, H, Hd).permute(0, 2, 1, 3)  # [B, H, N, Hd]
        
        # Perform attention (no masking needed for global-only as all summary tokens are valid)
        if not self.use_relative_positional_encoding or positions_3d is None:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)  # [B, H, N, Hd]
        else:
            # 3D RoPE for global-only: rotate q/k using per-cluster summary coord
            if self.summary_mode == 'avg':
                pos = positions_3d  # [B, N, T, 3]
                weights = masks.unsqueeze(-1).to(pos.dtype)
                pos_sum = (pos * weights).sum(dim=-2, keepdim=True)
                denom = weights.sum(dim=-2, keepdim=True).clamp_min(1e-6)
                pos_summary = pos_sum / denom  # [B, N, 1, 3]
            else:
                pos_summary = positions_3d[:, :, :1, :]
            pos_s = pos_summary.squeeze(2)  # [B, N, 3]
            chunk = (Hd // 6) * 2
            cx = cy = cz = chunk
            def apply_rope(t, coord, start, length):
                if length <= 0:
                    return t
                pairs = max(length // 2, 1)
                inv_freq = 1.0 / (10000 ** (torch.arange(0, pairs, device=t.device, dtype=t.dtype) / pairs))
                ang = coord.unsqueeze(-1) * inv_freq  # [B, N, pairs]
                cos = ang.cos().unsqueeze(1)
                sin = ang.sin().unsqueeze(1)
                seg = t[:, :, :, start:start+length]
                even, odd = seg[..., 0::2], seg[..., 1::2]
                seg_rot_even = even * cos - odd * sin
                seg_rot_odd  = even * sin + odd * cos
                seg_rot = torch.stack((seg_rot_even, seg_rot_odd), dim=-1).reshape_as(seg)
                t[:, :, :, start:start+length] = seg_rot
                return t
            q = apply_rope(q, pos_s[..., 0], 0, cx)
            k = apply_rope(k, pos_s[..., 0], 0, cx)
            q = apply_rope(q, pos_s[..., 1], cx, cy)
            k = apply_rope(k, pos_s[..., 1], cx, cy)
            q = apply_rope(q, pos_s[..., 2], cx+cy, cz)
            k = apply_rope(k, pos_s[..., 2], cx+cy, cz)
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        
        # Merge heads back
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, N, D)  # [B, N, D]
        attn = attn.unsqueeze(2)  # [B, N, 1, D]
        # Residual on the summary token
        updated_summary = attn + summary_token
        
        # Broadcast the updated summary information back to all tokens in each cluster
        expanded_summary = updated_summary.expand(-1, -1, padded_batches.size(2), -1)
        
        # Normalize both local and global information to prevent domination
        local_norm = torch.norm(padded_batches, dim=-1, keepdim=True).clamp(min=1e-6)
        global_norm = torch.norm(expanded_summary, dim=-1, keepdim=True).clamp(min=1e-6)
        
        local_normalized = padded_batches / local_norm
        global_normalized = expanded_summary / global_norm
        
        # Combine local and global information according to selected mode
        if self.use_gating:
            combined_features = torch.cat([local_normalized, global_normalized], dim=-1)
            pre_norm_gate = torch.sigmoid(self.gate_proj(combined_features))
            gate = self.gate_norm(pre_norm_gate)
            output = (1 - gate) * padded_batches + gate * expanded_summary
        else:
            output = padded_batches
        
        return output, updated_summary  # [B, N, 1, D]

# ===================== Global-Only Transformer Block =====================
class GlobalOnlyTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, ff_dim, num_heads=4, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        super(GlobalOnlyTransformerBlock, self).__init__()
        self.attention = GlobalOnlyMultiHeadAttention(embedding_dim, num_heads, dropout=dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)

    def forward(self, x, sum_token, masks, positions_3d=None):
        x, updated_sum_token = self.attention(x, sum_token, masks, positions_3d=positions_3d)
        x = self.feed_forward(x)
        return x, updated_sum_token

# ===================== Hierarchical Transformer Block =====================
class HierarchicalTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False, global_only=False):
        """
        Hierarchical processing block that follows the pattern:
        1. Local processing (tokens within clusters interact)
        2. Global-only processing (only global tokens interact with each other)
        3. Local processing again (tokens return to their clusters)
        """
        super(HierarchicalTransformerBlock, self).__init__()
        
        # Local attention blocks (before and after global)
        self.local_attention_1 = TransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.local_attention_2 = TransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        
        # Global-only attention block
        self.global_only_attention = GlobalOnlyTransformerBlock(embedding_dim, ff_dim, num_heads=num_heads, dropout=dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        
        # Feed forward layers
        self.feed_forward_1 = FeedForward(embedding_dim, ff_dim, dropout)
        self.feed_forward_2 = FeedForward(embedding_dim, ff_dim, dropout)
        self.feed_forward_3 = FeedForward(embedding_dim, ff_dim, dropout)

        # If True, this block will only apply global attention stage (skip local stages)
        self.global_only = global_only

    def forward(self, x, masks, sum_token_in=None, positions_3d=None, force_global_only=False):
        # Store input for final residual connection
        input_residual = x
        
        # Optional mode: apply only global attention (useful for enabling this for the first N blocks)
        if self.global_only or force_global_only:
            B, N, T, D = x.shape
            # If we have no incoming summary token and summary_mode is 'cls', synthesize a neutral token;
            # For 'avg' mode the global block computes its own summary, so the value is ignored.
            if sum_token_in is None:
                sum_token_in = torch.zeros(B, N, 1, D, device=x.device, dtype=x.dtype)
            x, final_sum_token = self.global_only_attention(x, sum_token_in, masks, positions_3d=positions_3d)
            # Apply top-level residual connection
            x = x + input_residual
            return x, final_sum_token
        
        # Stage 1: Local attention within clusters
        x, sum_token = self.local_attention_1(x, masks, sum_token_in=sum_token_in, positions_3d=positions_3d)
        local_residual_1 = x
        
        # Stage 2: Global-only attention (only summary tokens interact)
        x, final_sum_token = self.global_only_attention(x, sum_token, masks, positions_3d=positions_3d)
        
        return x, final_sum_token


# ===================== Global-Local Transformer Block =====================
class G_L_TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1, summary_mode='cls', use_relative_positional_encoding=False):
        super(G_L_TransformerBlock, self).__init__()
        self.attention_local = TransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.attention_global = GlobalTransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=use_relative_positional_encoding)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)

    def forward(self, x, masks, sum_token_in=None, positions_3d=None):
        # Store input for final residual connection
        input_residual = x
        
        # Local attention block: self-attention within each cluster
        x, sum_token = self.attention_local(x, masks, sum_token_in=sum_token_in, positions_3d=positions_3d)
        x = self.feed_forward(x)
        
        # Store intermediate state for local residual connection
        local_residual = x
        
        # Global attention block: cross-cluster attention using summary token
        x = self.attention_global(x, sum_token, masks, positions_3d=positions_3d)
        x = self.feed_forward(x)
        
        # Apply residual connections:
        x = x + input_residual
        return x, sum_token

# ===================== Nomeformer Model =====================
class nomeformer(nn.Module):
    def __init__(self, feature_dim, embedding_dim, num_heads, num_attention_blocks, dropout=0.1, summary_mode='cls', use_hierarchical=False, num_hierarchical_stages=1, fourier=False, relative_positional_encoding=False, global_only_middle_n=2):
        """
        Args:
            feature_dim: Input feature dimension
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            num_attention_blocks: Number of transformer blocks
            dropout: Dropout rate
            summary_mode: Summary mode for attention ('cls' or 'avg')
            use_hierarchical: Whether to use hierarchical processing (True) or original G_L blocks (False)
            num_hierarchical_stages: Number of hierarchical processing stages (only used if use_hierarchical=True)
            fourier: Whether to use positional fourier encoding (True) or not (False)
            relative_positional_encoding: If True, use relative 3D positional encoding (bias + value)
        """
        super(nomeformer, self).__init__()
        ff_dim = 2 * embedding_dim
        self.use_hierarchical = use_hierarchical
        self.num_hierarchical_stages = num_hierarchical_stages
        self.fourier = fourier
        self.relative_positional_encoding = relative_positional_encoding
        self.global_only_middle_n = max(int(global_only_middle_n), 0)
        
        # Set up embedding layer based on fourier flag
        if self.fourier:
            # When fourier=True, embedding only processes features after the first 9
            self.embedding = nn.Linear(feature_dim - 9, embedding_dim)
            # Add positional fourier encoding
            self.fourier_encoder = PositionalFourierEncoding(
                in_dim=9,  # First 9 features (XYZ, normals, etc.)
                num_frequencies=8,  # Reduced from 16 to 8
                out_dim=embedding_dim
            )
            # Learnable scale for fourier encoding contribution
            self.pos_scale = nn.Parameter(torch.tensor(0.1))
        else:
            # When fourier=False, embedding processes all features
            self.embedding = nn.Linear(feature_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)  # Add dropout layer after embedding
        self.norm = nn.LayerNorm(embedding_dim)
        
        if use_hierarchical: 
            blocks = []
            total = num_attention_blocks
            # Determine which indices should be global-only (middle K if specified)
            if self.global_only_middle_n > 0 and total > 0:
                k = min(self.global_only_middle_n, total)
                start = max((total - k) // 2, 0)
                end = start + k
                def is_global_only(idx):
                    return start <= idx < end
            else:
                def is_global_only(idx):
                    return False
            for i in range(total):
                blocks.append(
                    HierarchicalTransformerBlock(
                        embedding_dim,
                        num_heads,
                        ff_dim,
                        dropout,
                        summary_mode=summary_mode,
                        use_relative_positional_encoding=self.relative_positional_encoding,
                        global_only=is_global_only(i)
                    )
                )
            self.attention_blocks = nn.ModuleList(blocks)
        else:
            # Stack of original global-local transformer blocks
            self.attention_blocks = nn.ModuleList([
                G_L_TransformerBlock(embedding_dim, num_heads, ff_dim, dropout, summary_mode=summary_mode, use_relative_positional_encoding=self.relative_positional_encoding) 
                for _ in range(num_attention_blocks)
            ])
        
        self.apply(jepa_init_weights)

    def forward(self, input_tensor, masks_x):
        # When fourier=True, use features after the first 9 for embedding
        # When fourier=False, use all features for embedding
        if self.fourier:
            # Extract features after the first 9 (skip position features)
            features_to_embed = input_tensor[:, :, :, 9:].float()  # [B, N, T, feature_dim-9]
            # Extract first 9 features for fourier encoding
            position_features = input_tensor[:, :, :, :9].float()  # [B, N, T, 9]
        else:
            # Use all features for embedding
            features_to_embed = input_tensor.float()
            position_features = None
        
        # Embed input features
        embedded_input = self.embedding(features_to_embed)
        embedded_input = self.dropout(embedded_input)  # Apply dropout to the embedded input
        
        # Apply positional fourier encoding if enabled
        if self.fourier:
            B, N, T, _ = position_features.shape
            # Reshape for fourier encoding: [B*N, T, 9]
            position_flat = position_features.view(B * N, T, 9)
            
            # Apply fourier encoding
            fourier_encoded = self.fourier_encoder(position_flat)  # [B*N, T, embedding_dim]
            
            # Reshape back to original shape: [B, N, T, embedding_dim]
            fourier_encoded = fourier_encoded.view(B, N, T, -1)
            
            # Normalize fourier encoding to match embedded features scale
            fourier_encoded = F.layer_norm(fourier_encoded, fourier_encoded.shape[-1:])
            
            # Add scaled fourier encoding to embedded features
            embedded_input = embedded_input + self.pos_scale * fourier_encoded
        
        output = embedded_input
        
        # Prepare 3D positions for relative positional encoding, if enabled
        if self.relative_positional_encoding:
            if self.fourier:
                # position_features: [B, N, T, 9] -> three vertices (v0,v1,v2) each with (x,y,z)
                Bpf, Npf, Tpf, _ = position_features.shape
                positions_3d = position_features.view(Bpf, Npf, Tpf, 3, 3).mean(dim=-2)  # centroid [B, N, T, 3]
            else:
                # Assume first 9 dims are 3 vertices' XYZ; take centroid
                pos9 = input_tensor[..., :9].float()
                Bpf, Npf, Tpf, _ = pos9.shape
                positions_3d = pos9.view(Bpf, Npf, Tpf, 3, 3).mean(dim=-2)
        else:
            positions_3d = None
        
        # Initialize persistent CLS/summary token once and pass through blocks
        B, N, T, D = output.shape
        persistent_sum_token = None  # None -> first block will initialize from its learnable embedding
        
        # Pass through each attention block
        for attention_block in self.attention_blocks:
            if isinstance(attention_block, G_L_TransformerBlock):
                output, persistent_sum_token = attention_block(output, masks_x, sum_token_in=persistent_sum_token, positions_3d=positions_3d)
            elif isinstance(attention_block, HierarchicalTransformerBlock):
                output, persistent_sum_token = attention_block(output, masks_x, sum_token_in=persistent_sum_token, positions_3d=positions_3d)
            else:
                # Fallback for any block conforming to (x, masks, sum_token_in)
                try:
                    output, persistent_sum_token = attention_block(output, masks_x, sum_token_in=persistent_sum_token)
                except TypeError:
                    output = attention_block(output, masks_x)
        
        return output

class PositionalFourierEncoding(nn.Module):
    def __init__(self, in_dim=9, num_frequencies=16, out_dim=384):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.linear = nn.Linear(2 * in_dim * num_frequencies, out_dim)
        
        # Precompute frequencies
        frequencies = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("frequencies", frequencies)
        
    def forward(self, x):
        # x: [B, N, 9] (includes XYZ, normals, etc.)
        x_exp = x.unsqueeze(-1) * self.frequencies  # [B, N, 9, num_frequencies]
        x_sin = torch.sin(2 * np.pi * x_exp)
        x_cos = torch.cos(2 * np.pi * x_exp)
        x_fourier = torch.cat([x_sin, x_cos], dim=-1)  # [B, N, 9, 2 * num_frequencies]
        x_flat = x_fourier.view(x.shape[0], x.shape[1], -1)  # [B, N, 9 * 2 * num_frequencies]
        return self.linear(x_flat)  # [B, N, out_dim]


