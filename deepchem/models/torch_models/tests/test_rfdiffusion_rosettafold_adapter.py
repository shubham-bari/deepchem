"""Integration tests for RFdiffusion with RosettaFold backend."""

import pytest

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from deepchem.models.torch_models.rfdiffusion import BackboneDiffusion, CosineSchedule


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestRFdiffusionRosettaFoldBackend:
    """Verify RFdiffusion compatibility with RosettaFold denoiser backend."""

    def setup_method(self):
        torch.manual_seed(13)
        self.batch_size = 2
        self.seq_len = 12
        self.coord_dim = 9
        self.x = torch.randn(self.batch_size, self.seq_len, self.coord_dim)
        self.t = torch.randint(0, 100, (self.batch_size,))
        self.backend_cfg = {
            "num_heads": 4,
            "num_track_blocks": 2,
            "num_recycles": 1,
            "transition_factor": 2,
            "dropout": 0.0,
            "max_seq_len": 64,
            "max_msa_depth": 8,
            "input_spec": {
                "seq_dim": 24,
                "msa_dim": 24,
                "pair_dim": 16,
                "structure_dim": 24,
                "coord_dim": self.coord_dim,
            },
        }

    def test_backbone_diffusion_rosettafold_output_shape(self):
        model = BackboneDiffusion(coord_dim=self.coord_dim,
                                  backend="rosettafold",
                                  rosettafold_config=self.backend_cfg)
        out = model([self.x, self.t])
        assert out.shape == self.x.shape

    def test_backend_selection_guard(self):
        with pytest.raises(ValueError):
            BackboneDiffusion(coord_dim=self.coord_dim,
                              backend="invalid_backend")

    def test_schedule_p_sample_with_rosettafold_backend(self):
        model = BackboneDiffusion(coord_dim=self.coord_dim,
                                  backend="rosettafold",
                                  rosettafold_config=self.backend_cfg)
        schedule = CosineSchedule(num_timesteps=50)
        t = torch.randint(0, 49, (self.batch_size,))
        x_prev = schedule.p_sample(model, self.x, t)
        assert x_prev.shape == self.x.shape
        assert torch.isfinite(x_prev).all()

    def test_schedule_sample_with_rosettafold_backend(self):
        model = BackboneDiffusion(coord_dim=self.coord_dim,
                                  backend="rosettafold",
                                  rosettafold_config=self.backend_cfg)
        schedule = CosineSchedule(num_timesteps=8)
        sampled = schedule.sample(model, (1, 6, self.coord_dim),
                                  device=torch.device("cpu"))
        assert sampled.shape == (1, 6, self.coord_dim)
        assert torch.isfinite(sampled).all()
