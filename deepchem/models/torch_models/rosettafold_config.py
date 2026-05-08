"""Configuration objects for RosettaFold-style torch models.

This module defines dataclasses used to describe model hyperparameters and
expected tensor dimensions for RosettaFold-style models integrated in
DeepChem.
"""

from dataclasses import dataclass, field


@dataclass
class RosettaFoldInputSpec:
    """Canonical channel specification for RosettaFold-style inputs.

    Parameters
    ----------
    seq_dim : int, default=256
        Feature dimension of per-residue sequence representations in the 1D
        track.
    msa_dim : int, default=256
        Feature dimension of per-token MSA representations.
    pair_dim : int, default=128
        Feature dimension of pairwise residue representations in the 2D track.
    structure_dim : int, default=256
        Feature dimension of per-residue structure states consumed by the 3D
        track.
    coord_dim : int, default=9
        Number of coordinate channels predicted by the denoiser head. For
        backbone coordinates this is typically ``(N, CA, C) x (x, y, z) = 9``.
    """

    seq_dim: int = 256
    msa_dim: int = 256
    pair_dim: int = 128
    structure_dim: int = 256
    coord_dim: int = 9


@dataclass
class RosettaFoldConfig:
    """Configuration for RosettaFold-style track and recycling modules.

    Parameters
    ----------
    input_spec : RosettaFoldInputSpec
        Input/output channel specification object.
    num_heads : int, default=8
        Number of attention heads used in attention-based submodules.
    num_track_blocks : int, default=4
        Number of repeated track-update blocks executed per recycle.
    num_recycles : int, default=2
        Number of recycling iterations.
    transition_factor : int, default=4
        Multiplicative expansion factor for transition feed-forward networks.
    dropout : float, default=0.1
        Dropout probability used by attention and transition layers.
    max_seq_len : int, default=1024
        Maximum supported protein sequence length.
    max_msa_depth : int, default=128
        Maximum supported number of MSA rows.
    epsilon : float, default=1e-8
        Numerical stability constant for normalization paths.
    """

    input_spec: RosettaFoldInputSpec = field(
        default_factory=RosettaFoldInputSpec)
    num_heads: int = 8
    num_track_blocks: int = 4
    num_recycles: int = 2
    transition_factor: int = 4
    dropout: float = 0.1
    max_seq_len: int = 1024
    max_msa_depth: int = 128
    epsilon: float = 1e-8

    def validate(self) -> None:
        """Validate configuration values.

        Raises
        ------
        ValueError
            If one or more fields are outside valid ranges.
        """
        if self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if self.num_track_blocks <= 0:
            raise ValueError("num_track_blocks must be a positive integer.")
        if self.num_recycles <= 0:
            raise ValueError("num_recycles must be a positive integer.")
        if self.transition_factor <= 0:
            raise ValueError("transition_factor must be a positive integer.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")
        if self.max_msa_depth <= 0:
            raise ValueError("max_msa_depth must be a positive integer.")
