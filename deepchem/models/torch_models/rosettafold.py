"""RosettaFold-style backbone and denoiser modules.

This module defines a DeepChem-native RosettaFold-style architecture with
1D/MSA, 2D/pair, and 3D/structure tracks plus recycling. A denoiser wrapper is
provided for RFdiffusion-style coordinate noise prediction.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from deepchem.models.torch_models.rosettafold_config import RosettaFoldConfig
from deepchem.models.torch_models.rosettafold_layers import (
    MSAColumnAttentionBlock, MSARowAttentionBlock, MSATransition, PairToSeqBias,
    PairTransition, RecyclingEmbedder, SeqToPairProjection, StructureTrackBlock,
    TriangleAttentionEndingNode, TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing)


class _SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion conditioning.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timesteps into sinusoidal embeddings.

        Parameters
        ----------
        t : torch.Tensor
            Integer timesteps of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(batch, dim)``.
        """
        half_dim = self.dim // 2
        step = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -step)
        angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class RosettaFoldTrackBlock(nn.Module):
    """Single 1D/2D/3D joint update block.

    Parameters
    ----------
    config : RosettaFoldConfig
        Model configuration object.
    """

    def __init__(self, config: RosettaFoldConfig) -> None:
        super().__init__()
        c_s = config.input_spec.seq_dim
        c_m = config.input_spec.msa_dim
        c_z = config.input_spec.pair_dim
        self.msa_row = MSARowAttentionBlock(c_m, config.num_heads,
                                            config.dropout)
        self.msa_col = MSAColumnAttentionBlock(c_m, config.num_heads,
                                               config.dropout)
        self.msa_transition = MSATransition(c_m, config.transition_factor,
                                            config.dropout)
        self.seq_to_pair = SeqToPairProjection(c_s, c_z)
        self.pair_transition = PairTransition(c_z, config.transition_factor,
                                              config.dropout)
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z)
        self.tri_attn_start = TriangleAttentionStartingNode(
            c_z, config.num_heads, config.dropout)
        self.tri_attn_end = TriangleAttentionEndingNode(c_z, config.num_heads,
                                                        config.dropout)
        self.pair_to_seq = PairToSeqBias(c_z, c_s)
        self.structure = StructureTrackBlock(c_s, c_z, config.num_heads,
                                             config.dropout)

    def forward(
        self,
        seq_repr: torch.Tensor,
        msa_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """Apply one complete cross-track update.

        Parameters
        ----------
        seq_repr : torch.Tensor
            Sequence tensor of shape ``(batch, seq_len, c_s)``.
        msa_repr : torch.Tensor
            MSA tensor of shape ``(batch, n_msa, seq_len, c_m)``.
        pair_repr : torch.Tensor
            Pair tensor of shape ``(batch, seq_len, seq_len, c_z)``.
        rotations : torch.Tensor
            Rotation matrices of shape ``(batch, seq_len, 3, 3)``.
        translations : torch.Tensor
            Translation vectors of shape ``(batch, seq_len, 3)``.
        msa_mask : torch.Tensor, optional
            Boolean MSA mask of shape ``(batch, n_msa, seq_len)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Updated sequence, MSA, pair, rotations, and translations.
        """
        msa_repr = self.msa_row(msa_repr, msa_mask)
        msa_repr = self.msa_col(msa_repr, msa_mask)
        msa_repr = self.msa_transition(msa_repr)

        pair_repr = pair_repr + self.seq_to_pair(seq_repr)
        pair_repr = self.pair_transition(pair_repr)
        pair_repr = self.tri_mul_out(pair_repr)
        pair_repr = self.tri_mul_in(pair_repr)
        pair_repr = self.tri_attn_start(pair_repr)
        pair_repr = self.tri_attn_end(pair_repr)

        seq_repr = seq_repr + msa_repr.mean(dim=1) + self.pair_to_seq(pair_repr)
        seq_repr, rotations, translations = self.structure(
            seq_repr, pair_repr, rotations, translations)
        return seq_repr, msa_repr, pair_repr, rotations, translations


class RosettaFoldBackbone(nn.Module):
    """RosettaFold-style 3-track backbone with recycling.

    Parameters
    ----------
    config : RosettaFoldConfig, optional
        Model configuration. If ``None``, defaults are used.
    """

    def __init__(self, config: Optional[RosettaFoldConfig] = None) -> None:
        super().__init__()
        self.config = config or RosettaFoldConfig()
        self.config.validate()
        c_s = self.config.input_spec.seq_dim
        c_m = self.config.input_spec.msa_dim
        c_z = self.config.input_spec.pair_dim

        self.seq_pos_emb = nn.Embedding(self.config.max_seq_len, c_s)
        self.msa_row_emb = nn.Embedding(self.config.max_msa_depth, c_m)
        self.blocks = nn.ModuleList([
            RosettaFoldTrackBlock(self.config)
            for _ in range(self.config.num_track_blocks)
        ])
        self.recycle_embedder = RecyclingEmbedder(c_s, c_m, c_z)

    def _init_frames(self, batch_size: int, seq_len: int, device: torch.device,
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create default identity rotations and zero translations.

        Parameters
        ----------
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.
        device : torch.device
            Output device.
        dtype : torch.dtype
            Output dtype.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Rotation matrices of shape ``(batch, seq_len, 3, 3)`` and
            translation vectors of shape ``(batch, seq_len, 3)``.
        """
        rotations = torch.eye(3, device=device, dtype=dtype).reshape(
            1, 1, 3, 3).repeat(batch_size, seq_len, 1, 1)
        translations = torch.zeros(batch_size,
                                   seq_len,
                                   3,
                                   device=device,
                                   dtype=dtype)
        return rotations, translations

    def forward(self,
                batch: Dict[str, torch.Tensor],
                return_aux: bool = False) -> Dict[str, torch.Tensor]:
        """Run the RosettaFold-style backbone.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Input dictionary containing:

            - ``seq_1d`` : shape ``(batch, seq_len, c_s)``
            - ``msa`` : shape ``(batch, n_msa, seq_len, c_m)``
            - ``pair_2d`` : shape ``(batch, seq_len, seq_len, c_z)``
            - ``rotations`` (optional) : shape ``(batch, seq_len, 3, 3)``
            - ``translations`` (optional) : shape ``(batch, seq_len, 3)``
            - ``msa_mask`` (optional) : shape ``(batch, n_msa, seq_len)``
        return_aux : bool, default=False
            If ``True``, include intermediate-state tensors in the return dict.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing final ``seq_repr``, ``msa_repr``,
            ``pair_repr``, ``rotations``, and ``translations``.
        """
        seq_repr = batch["seq_1d"]
        msa_repr = batch["msa"]
        pair_repr = batch["pair_2d"]
        msa_mask = batch.get("msa_mask")
        bsz, seq_len, _ = seq_repr.shape

        pos_ids = torch.arange(seq_len, device=seq_repr.device).unsqueeze(0)
        seq_repr = seq_repr + self.seq_pos_emb(pos_ids)
        row_ids = torch.arange(msa_repr.shape[1],
                               device=msa_repr.device).view(1, -1, 1)
        msa_repr = msa_repr + self.msa_row_emb(row_ids)

        rotations = batch.get("rotations")
        translations = batch.get("translations")
        if rotations is None or translations is None:
            rotations, translations = self._init_frames(bsz, seq_len,
                                                        seq_repr.device,
                                                        seq_repr.dtype)

        aux_states: List[Dict[str, torch.Tensor]] = []
        for _ in range(self.config.num_recycles):
            for block in self.blocks:
                seq_repr, msa_repr, pair_repr, rotations, translations = block(
                    seq_repr, msa_repr, pair_repr, rotations, translations,
                    msa_mask)
            seq_delta, msa_delta, pair_delta = self.recycle_embedder(
                seq_repr, msa_repr, pair_repr)
            seq_repr = seq_repr + seq_delta
            msa_repr = msa_repr + msa_delta
            pair_repr = pair_repr + pair_delta
            if return_aux:
                aux_states.append({
                    "seq_repr": seq_repr,
                    "msa_repr": msa_repr,
                    "pair_repr": pair_repr,
                    "rotations": rotations,
                    "translations": translations
                })

        output = {
            "seq_repr": seq_repr,
            "msa_repr": msa_repr,
            "pair_repr": pair_repr,
            "rotations": rotations,
            "translations": translations
        }
        if return_aux:
            output["aux_states"] = aux_states  # type: ignore[assignment]
        return output


class RosettaFoldDenoiser(nn.Module):
    """RosettaFold-style denoiser head for diffusion training.

    Parameters
    ----------
    config : RosettaFoldConfig, optional
        Model configuration. If ``None``, defaults are used.
    """

    def __init__(self, config: Optional[RosettaFoldConfig] = None) -> None:
        super().__init__()
        self.config = config or RosettaFoldConfig()
        self.backbone = RosettaFoldBackbone(self.config)
        c_s = self.config.input_spec.seq_dim
        c_m = self.config.input_spec.msa_dim
        c_z = self.config.input_spec.pair_dim
        c_coord = self.config.input_spec.coord_dim

        self.coord_embed = nn.Linear(c_coord, c_s)
        self.time_embed = _SinusoidalTimeEmbedding(c_s)
        self.time_proj = nn.Sequential(nn.Linear(c_s, c_s), nn.SiLU(),
                                       nn.Linear(c_s, c_s))
        self.seq_to_msa = nn.Linear(c_s, c_m)
        self.pair_dist_embed = nn.Linear(1, c_z)
        self.noise_head = nn.Sequential(nn.LayerNorm(c_s), nn.Linear(c_s, c_s),
                                        nn.GELU(), nn.Linear(c_s, c_coord))
        nn.init.zeros_(self.noise_head[-1].weight)
        nn.init.zeros_(self.noise_head[-1].bias)

    def _build_pair_from_coords(self, x_t: torch.Tensor) -> torch.Tensor:
        """Build pair features from noisy coordinates.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy coordinates of shape ``(batch, seq_len, coord_dim)``.

        Returns
        -------
        torch.Tensor
            Pair tensor of shape ``(batch, seq_len, seq_len, pair_dim)``.
        """
        coords = x_t[..., :3]
        dist = torch.cdist(coords, coords, p=2).unsqueeze(-1)
        return self.pair_dist_embed(dist)

    def forward_denoise(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            cond_batch: Optional[Dict[str,
                                      torch.Tensor]] = None) -> torch.Tensor:
        """Predict coordinate noise for diffusion.

        Parameters
        ----------
        x_t : torch.Tensor
            Noisy coordinates of shape ``(batch, seq_len, coord_dim)``.
        t : torch.Tensor
            Timesteps of shape ``(batch,)``.
        cond_batch : dict[str, torch.Tensor], optional
            Optional conditioning dictionary. If provided, keys ``seq_1d``,
            ``msa``, ``pair_2d``, ``rotations``, and ``translations`` override
            internally built defaults.

        Returns
        -------
        torch.Tensor
            Predicted coordinate noise of shape ``(batch, seq_len, coord_dim)``.
        """
        cond_batch = cond_batch or {}
        seq_repr = cond_batch.get("seq_1d")
        if seq_repr is None:
            seq_repr = self.coord_embed(x_t)
        time_cond = self.time_proj(self.time_embed(t.long())).unsqueeze(1)
        seq_repr = seq_repr + time_cond

        msa_repr = cond_batch.get("msa")
        if msa_repr is None:
            msa_repr = self.seq_to_msa(seq_repr).unsqueeze(1)

        pair_repr = cond_batch.get("pair_2d")
        if pair_repr is None:
            pair_repr = self._build_pair_from_coords(x_t)

        batch: Dict[str, torch.Tensor] = {
            "seq_1d": seq_repr,
            "msa": msa_repr,
            "pair_2d": pair_repr
        }
        if "rotations" in cond_batch:
            batch["rotations"] = cond_batch["rotations"]
        if "translations" in cond_batch:
            batch["translations"] = cond_batch["translations"]
        if "msa_mask" in cond_batch:
            batch["msa_mask"] = cond_batch["msa_mask"]

        outputs = self.backbone(batch, return_aux=False)
        seq_repr = torch.nan_to_num(outputs["seq_repr"],
                                    nan=0.0,
                                    posinf=1e4,
                                    neginf=-1e4)
        noise = self.noise_head(seq_repr)
        return torch.nan_to_num(noise, nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass with RFdiffusion-compatible list input.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            List where ``inputs[0]`` is ``x_t`` and ``inputs[1]`` is ``t``.

        Returns
        -------
        torch.Tensor
            Predicted noise with shape matching ``inputs[0]``.
        """
        if len(inputs) < 2:
            raise ValueError("Expected at least [x_t, t] in inputs.")
        return self.forward_denoise(inputs[0], inputs[1], cond_batch=None)
