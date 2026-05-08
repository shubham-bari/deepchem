"""RosettaFold-style track update layers.

This module provides reusable building blocks for 1D/MSA, 2D/pair, and
3D/structure tracks. The implementations are designed to preserve canonical
tensor contracts and provide a DeepChem-native baseline architecture.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class _Transition(nn.Module):
    """Feed-forward transition block with residual connection.

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    expansion : int
        Hidden expansion factor.
    dropout : float
        Dropout probability.
    """

    def __init__(self, dim: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a normalized transition with residual update.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with feature dimension in the last axis.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as ``x``.
        """
        return x + self.ffn(self.norm(x))


class MSARowAttentionBlock(nn.Module):
    """Row-wise attention update over residue positions for each MSA row.

    Parameters
    ----------
    msa_dim : int
        Feature dimension of the MSA representation.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(self, msa_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(msa_dim)
        self.attn = nn.MultiheadAttention(msa_dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                msa_repr: torch.Tensor,
                msa_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply row-wise attention.

        Parameters
        ----------
        msa_repr : torch.Tensor
            MSA tensor of shape ``(batch, n_msa, seq_len, msa_dim)``.
        msa_mask : torch.Tensor, optional
            Boolean mask of shape ``(batch, n_msa, seq_len)`` where ``True``
            denotes valid tokens.

        Returns
        -------
        torch.Tensor
            Updated MSA tensor with shape ``(batch, n_msa, seq_len, msa_dim)``.
        """
        bsz, n_msa, seq_len, dim = msa_repr.shape
        x = self.norm(msa_repr).reshape(bsz * n_msa, seq_len, dim)
        key_padding_mask = None
        if msa_mask is not None:
            key_padding_mask = ~msa_mask.reshape(bsz * n_msa, seq_len).bool()
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        out = out.reshape(bsz, n_msa, seq_len, dim)
        return msa_repr + self.dropout(out)


class MSAColumnAttentionBlock(nn.Module):
    """Column-wise attention update over MSA rows for each residue position.

    Parameters
    ----------
    msa_dim : int
        Feature dimension of the MSA representation.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(self, msa_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(msa_dim)
        self.attn = nn.MultiheadAttention(msa_dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                msa_repr: torch.Tensor,
                msa_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply column-wise attention.

        Parameters
        ----------
        msa_repr : torch.Tensor
            MSA tensor of shape ``(batch, n_msa, seq_len, msa_dim)``.
        msa_mask : torch.Tensor, optional
            Boolean mask of shape ``(batch, n_msa, seq_len)`` where ``True``
            denotes valid tokens.

        Returns
        -------
        torch.Tensor
            Updated MSA tensor with shape ``(batch, n_msa, seq_len, msa_dim)``.
        """
        bsz, n_msa, seq_len, dim = msa_repr.shape
        x = self.norm(msa_repr).permute(0, 2, 1,
                                        3).reshape(bsz * seq_len, n_msa, dim)
        key_padding_mask = None
        if msa_mask is not None:
            key_padding_mask = ~msa_mask.permute(0, 2, 1).reshape(
                bsz * seq_len, n_msa).bool()
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        out = out.reshape(bsz, seq_len, n_msa, dim).permute(0, 2, 1, 3)
        return msa_repr + self.dropout(out)


class MSATransition(_Transition):
    """Transition block for MSA representations.

    This class specializes :class:`_Transition` for MSA tensors and keeps the
    same residual feed-forward update semantics.

    Parameters
    ----------
    msa_dim : int
        Feature dimension of MSA representations.
    expansion : int
        Hidden-layer expansion factor for the internal feed-forward block.
    dropout : float
        Dropout probability used in the transition block.
    """

    def __init__(self, msa_dim: int, expansion: int, dropout: float) -> None:
        super().__init__(dim=msa_dim, expansion=expansion, dropout=dropout)

    def forward(self, msa_repr: torch.Tensor) -> torch.Tensor:
        """Apply the MSA transition update.

        Parameters
        ----------
        msa_repr : torch.Tensor
            Tensor of shape ``(batch, n_msa, seq_len, msa_dim)``.

        Returns
        -------
        torch.Tensor
            Updated MSA tensor with the same shape as ``msa_repr``.
        """
        return super().forward(msa_repr)


class PairTransition(_Transition):
    """Transition block for pair representations.

    This class specializes :class:`_Transition` for pairwise residue tensors.

    Parameters
    ----------
    pair_dim : int
        Feature dimension of pair representations.
    expansion : int
        Hidden-layer expansion factor for the internal feed-forward block.
    dropout : float
        Dropout probability used in the transition block.
    """

    def __init__(self, pair_dim: int, expansion: int, dropout: float) -> None:
        super().__init__(dim=pair_dim, expansion=expansion, dropout=dropout)

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Apply the pair transition update.

        Parameters
        ----------
        pair_repr : torch.Tensor
            Tensor of shape ``(batch, seq_len, seq_len, pair_dim)``.

        Returns
        -------
        torch.Tensor
            Updated pair tensor with the same shape as ``pair_repr``.
        """
        return super().forward(pair_repr)


class SeqToPairProjection(nn.Module):
    """Project single-sequence states into pair updates.

    Parameters
    ----------
    seq_dim : int
        Single representation dimension.
    pair_dim : int
        Pair representation dimension.
    """

    def __init__(self, seq_dim: int, pair_dim: int) -> None:
        super().__init__()
        self.left = nn.Linear(seq_dim, pair_dim)
        self.right = nn.Linear(seq_dim, pair_dim)
        self.out = nn.Linear(pair_dim, pair_dim)

    def forward(self, seq_repr: torch.Tensor) -> torch.Tensor:
        """Build pair updates from per-residue sequence states.

        Parameters
        ----------
        seq_repr : torch.Tensor
            Tensor of shape ``(batch, seq_len, seq_dim)``.

        Returns
        -------
        torch.Tensor
            Pair update tensor of shape ``(batch, seq_len, seq_len, pair_dim)``.
        """
        left = self.left(seq_repr)
        right = self.right(seq_repr)
        pair = left.unsqueeze(2) + right.unsqueeze(1)
        return self.out(pair)


class PairToSeqBias(nn.Module):
    """Aggregate pair states into a sequence-level bias signal.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    seq_dim : int
        Sequence representation dimension.
    """

    def __init__(self, pair_dim: int, seq_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim)
        self.proj = nn.Linear(pair_dim, seq_dim)

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Compute sequence bias from pair representations.

        Parameters
        ----------
        pair_repr : torch.Tensor
            Pair tensor of shape ``(batch, seq_len, seq_len, pair_dim)``.

        Returns
        -------
        torch.Tensor
            Sequence bias of shape ``(batch, seq_len, seq_dim)``.
        """
        pooled = self.norm(pair_repr).mean(dim=2)
        return self.proj(pooled)


class TriangleMultiplicationOutgoing(nn.Module):
    """Outgoing triangle multiplication for pair updates.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    """

    def __init__(self, pair_dim: int) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.left = nn.Linear(pair_dim, pair_dim)
        self.right = nn.Linear(pair_dim, pair_dim)
        self.gate = nn.Linear(pair_dim, pair_dim)
        self.out = nn.Linear(pair_dim, pair_dim)

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Apply outgoing triangle multiplication update.

        Parameters
        ----------
        pair_repr : torch.Tensor
            Pair tensor of shape ``(batch, seq_len, seq_len, pair_dim)``.

        Returns
        -------
        torch.Tensor
            Updated pair tensor with identical shape.
        """
        a = self.left(pair_repr)
        b = self.right(pair_repr)
        tri = torch.einsum("bikd,bkjd->bijd", a, b)
        tri = tri / (self.pair_dim**0.5)
        gated = torch.sigmoid(self.gate(pair_repr)) * tri
        return pair_repr + self.out(gated)


class TriangleMultiplicationIncoming(nn.Module):
    """Incoming triangle multiplication for pair updates.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    """

    def __init__(self, pair_dim: int) -> None:
        super().__init__()
        self.pair_dim = pair_dim
        self.left = nn.Linear(pair_dim, pair_dim)
        self.right = nn.Linear(pair_dim, pair_dim)
        self.gate = nn.Linear(pair_dim, pair_dim)
        self.out = nn.Linear(pair_dim, pair_dim)

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Apply incoming triangle multiplication update.

        Parameters
        ----------
        pair_repr : torch.Tensor
            Pair tensor of shape ``(batch, seq_len, seq_len, pair_dim)``.

        Returns
        -------
        torch.Tensor
            Updated pair tensor with identical shape.
        """
        a = self.left(pair_repr)
        b = self.right(pair_repr)
        tri = torch.einsum("bkid,bkjd->bijd", a, b)
        tri = tri / (self.pair_dim**0.5)
        gated = torch.sigmoid(self.gate(pair_repr)) * tri
        return pair_repr + self.out(gated)


class TriangleAttentionStartingNode(nn.Module):
    """Triangle attention around starting nodes.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(self, pair_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim)
        self.attn = nn.MultiheadAttention(pair_dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Apply starting-node triangle attention.

        Parameters
        ----------
        pair_repr : torch.Tensor
            Pair tensor of shape ``(batch, seq_len, seq_len, pair_dim)``.

        Returns
        -------
        torch.Tensor
            Updated pair tensor with identical shape.
        """
        bsz, seq_len, _, dim = pair_repr.shape
        x = self.norm(pair_repr).reshape(bsz * seq_len, seq_len, dim)
        out, _ = self.attn(x, x, x)
        out = out.reshape(bsz, seq_len, seq_len, dim)
        return pair_repr + out


class TriangleAttentionEndingNode(nn.Module):
    """Triangle attention around ending nodes.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(self, pair_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim)
        self.attn = nn.MultiheadAttention(pair_dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """Apply ending-node triangle attention.

        Parameters
        ----------
        pair_repr : torch.Tensor
            Pair tensor of shape ``(batch, seq_len, seq_len, pair_dim)``.

        Returns
        -------
        torch.Tensor
            Updated pair tensor with identical shape.
        """
        bsz, seq_len, _, dim = pair_repr.shape
        x = self.norm(pair_repr).transpose(1,
                                           2).reshape(bsz * seq_len, seq_len,
                                                      dim)
        out, _ = self.attn(x, x, x)
        out = out.reshape(bsz, seq_len, seq_len, dim).transpose(1, 2)
        return pair_repr + out


class StructureTrackBlock(nn.Module):
    """3D structure-track block operating on frame states.

    Parameters
    ----------
    seq_dim : int
        Sequence/state representation dimension.
    pair_dim : int
        Pair representation dimension.
    num_heads : int
        Number of self-attention heads for single-state updates.
    dropout : float
        Dropout probability.
    """

    def __init__(self, seq_dim: int, pair_dim: int, num_heads: int,
                 dropout: float) -> None:
        super().__init__()
        self.seq_norm = nn.LayerNorm(seq_dim)
        self.seq_attn = nn.MultiheadAttention(seq_dim,
                                              num_heads,
                                              dropout=dropout,
                                              batch_first=True)
        self.pair_to_seq = PairToSeqBias(pair_dim, seq_dim)
        self.transition = _Transition(seq_dim, expansion=4, dropout=dropout)
        self.translation_head = nn.Linear(seq_dim, 3)

    def forward(
        self, seq_repr: torch.Tensor, pair_repr: torch.Tensor,
        rotations: torch.Tensor, translations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update sequence state and rigid-frame translations.

        Parameters
        ----------
        seq_repr : torch.Tensor
            Sequence/state tensor of shape ``(batch, seq_len, seq_dim)``.
        pair_repr : torch.Tensor
            Pair tensor of shape ``(batch, seq_len, seq_len, pair_dim)``.
        rotations : torch.Tensor
            Rotation matrices of shape ``(batch, seq_len, 3, 3)``.
        translations : torch.Tensor
            Translation vectors of shape ``(batch, seq_len, 3)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated ``(seq_repr, rotations, translations)``. Rotation matrices
            are currently propagated as residual identity updates while
            translations receive learned deltas.
        """
        bias = self.pair_to_seq(pair_repr)
        x = self.seq_norm(seq_repr + bias)
        attn_out, _ = self.seq_attn(x, x, x)
        x = seq_repr + attn_out
        x = self.transition(x)
        delta_t = self.translation_head(x)
        return x, rotations, translations + delta_t


class RecyclingEmbedder(nn.Module):
    """Embed previous-cycle states for recycling.

    Parameters
    ----------
    seq_dim : int
        Sequence representation dimension.
    msa_dim : int
        MSA representation dimension.
    pair_dim : int
        Pair representation dimension.
    """

    def __init__(self, seq_dim: int, msa_dim: int, pair_dim: int) -> None:
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, seq_dim)
        self.msa_proj = nn.Linear(msa_dim, msa_dim)
        self.pair_proj = nn.Linear(pair_dim, pair_dim)

    def forward(
        self, seq_repr: torch.Tensor, msa_repr: torch.Tensor,
        pair_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project previous representations for the next recycle.

        Parameters
        ----------
        seq_repr : torch.Tensor
            Sequence tensor of shape ``(batch, seq_len, seq_dim)``.
        msa_repr : torch.Tensor
            MSA tensor of shape ``(batch, n_msa, seq_len, msa_dim)``.
        pair_repr : torch.Tensor
            Pair tensor of shape ``(batch, seq_len, seq_len, pair_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Recycled sequence, MSA, and pair representations.
        """
        return self.seq_proj(seq_repr), self.msa_proj(msa_repr), self.pair_proj(
            pair_repr)
