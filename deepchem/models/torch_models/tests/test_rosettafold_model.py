"""Unit tests for RosettaFold-style backbone, denoiser, and wrapper model."""

import pytest

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from deepchem.models.torch_models.rosettafold import (RosettaFoldBackbone,
                                                          RosettaFoldDenoiser)
    from deepchem.models.torch_models.rosettafold_config import RosettaFoldConfig
    from deepchem.models.torch_models.rosettafold_model import RosettaFoldModel


def _small_config():
    """Construct a small config for fast unit tests."""
    cfg = RosettaFoldConfig()
    cfg.input_spec.seq_dim = 24
    cfg.input_spec.msa_dim = 24
    cfg.input_spec.pair_dim = 16
    cfg.input_spec.structure_dim = 24
    cfg.input_spec.coord_dim = 9
    cfg.num_heads = 4
    cfg.num_track_blocks = 2
    cfg.num_recycles = 2
    cfg.transition_factor = 2
    cfg.max_seq_len = 64
    cfg.max_msa_depth = 16
    cfg.dropout = 0.0
    cfg.validate()
    return cfg


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestRosettaFoldModelContracts:
    """Contract tests for input and output behavior."""

    def setup_method(self):
        torch.manual_seed(11)
        self.cfg = _small_config()
        self.batch_size = 2
        self.seq_len = 10
        self.n_msa = 3

    def test_backbone_forward_shapes(self):
        model = RosettaFoldBackbone(config=self.cfg)
        batch = {
            "seq_1d":
                torch.randn(self.batch_size, self.seq_len,
                            self.cfg.input_spec.seq_dim),
            "msa":
                torch.randn(self.batch_size, self.n_msa, self.seq_len,
                            self.cfg.input_spec.msa_dim),
            "pair_2d":
                torch.randn(self.batch_size, self.seq_len, self.seq_len,
                            self.cfg.input_spec.pair_dim),
            "msa_mask":
                torch.ones(self.batch_size,
                           self.n_msa,
                           self.seq_len,
                           dtype=torch.bool),
        }
        out = model(batch, return_aux=True)
        assert out["seq_repr"].shape == (self.batch_size, self.seq_len,
                                         self.cfg.input_spec.seq_dim)
        assert out["msa_repr"].shape == (self.batch_size, self.n_msa,
                                         self.seq_len,
                                         self.cfg.input_spec.msa_dim)
        assert out["pair_repr"].shape == (self.batch_size, self.seq_len,
                                          self.seq_len,
                                          self.cfg.input_spec.pair_dim)
        assert out["rotations"].shape == (self.batch_size, self.seq_len, 3, 3)
        assert out["translations"].shape == (self.batch_size, self.seq_len, 3)
        assert len(out["aux_states"]) == self.cfg.num_recycles

    def test_denoiser_forward_denoise_shape(self):
        model = RosettaFoldDenoiser(config=self.cfg)
        x_t = torch.randn(self.batch_size, self.seq_len,
                          self.cfg.input_spec.coord_dim)
        t = torch.randint(0, 1000, (self.batch_size,))
        out = model.forward_denoise(x_t, t)
        assert out.shape == x_t.shape

    def test_denoiser_forward_list_contract(self):
        model = RosettaFoldDenoiser(config=self.cfg)
        x_t = torch.randn(self.batch_size, self.seq_len,
                          self.cfg.input_spec.coord_dim)
        t = torch.randint(0, 1000, (self.batch_size,))
        out = model([x_t, t])
        assert out.shape == x_t.shape

    def test_denoiser_with_conditioning_inputs(self):
        model = RosettaFoldDenoiser(config=self.cfg)
        x_t = torch.randn(self.batch_size, self.seq_len,
                          self.cfg.input_spec.coord_dim)
        t = torch.randint(0, 1000, (self.batch_size,))
        cond_batch = {
            "seq_1d":
                torch.randn(self.batch_size, self.seq_len,
                            self.cfg.input_spec.seq_dim),
            "msa":
                torch.randn(self.batch_size, self.n_msa, self.seq_len,
                            self.cfg.input_spec.msa_dim),
            "pair_2d":
                torch.randn(self.batch_size, self.seq_len, self.seq_len,
                            self.cfg.input_spec.pair_dim),
            "rotations":
                torch.eye(3).reshape(1, 1, 3, 3).repeat(self.batch_size,
                                                        self.seq_len, 1, 1),
            "translations":
                torch.zeros(self.batch_size, self.seq_len, 3),
        }
        out = model.forward_denoise(x_t, t, cond_batch=cond_batch)
        assert out.shape == x_t.shape

    def test_rosettafold_torchmodel_wrapper(self):
        model = RosettaFoldModel(config=self.cfg, batch_size=2)
        x_t = torch.randn(self.batch_size,
                          self.seq_len,
                          self.cfg.input_spec.coord_dim,
                          device=model.device)
        t = torch.randint(0, 1000, (self.batch_size,), device=model.device)
        out = model.model([x_t, t])
        assert out.shape == x_t.shape
