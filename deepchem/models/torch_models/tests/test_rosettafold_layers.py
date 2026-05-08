"""Unit tests for RosettaFold-style layer blocks."""

import pytest

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    from deepchem.models.torch_models.rosettafold_layers import (
        MSAColumnAttentionBlock, MSARowAttentionBlock, MSATransition,
        PairToSeqBias, PairTransition, SeqToPairProjection, StructureTrackBlock,
        TriangleAttentionEndingNode, TriangleAttentionStartingNode,
        TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
class TestRosettaFoldLayerShapes:
    """Shape and gradient tests for RosettaFold-style layers."""

    def setup_method(self):
        """Create deterministic dummy inputs."""
        torch.manual_seed(7)
        self.batch_size = 2
        self.n_msa = 3
        self.seq_len = 6
        self.c_s = 32
        self.c_m = 32
        self.c_z = 24
        self.msa = torch.randn(self.batch_size,
                               self.n_msa,
                               self.seq_len,
                               self.c_m,
                               requires_grad=True)
        self.seq = torch.randn(self.batch_size,
                               self.seq_len,
                               self.c_s,
                               requires_grad=True)
        self.pair = torch.randn(self.batch_size,
                                self.seq_len,
                                self.seq_len,
                                self.c_z,
                                requires_grad=True)
        self.rot = torch.eye(3).reshape(1, 1, 3,
                                        3).repeat(self.batch_size, self.seq_len,
                                                  1, 1)
        self.trans = torch.randn(self.batch_size, self.seq_len, 3)
        self.msa_mask = torch.ones(self.batch_size,
                                   self.n_msa,
                                   self.seq_len,
                                   dtype=torch.bool)

    def test_msa_row_attention_shape(self):
        layer = MSARowAttentionBlock(self.c_m, num_heads=4, dropout=0.0)
        out = layer(self.msa, self.msa_mask)
        assert out.shape == self.msa.shape

    def test_msa_column_attention_shape(self):
        layer = MSAColumnAttentionBlock(self.c_m, num_heads=4, dropout=0.0)
        out = layer(self.msa, self.msa_mask)
        assert out.shape == self.msa.shape

    def test_msa_transition_shape(self):
        layer = MSATransition(self.c_m, expansion=2, dropout=0.0)
        out = layer(self.msa)
        assert out.shape == self.msa.shape

    def test_seq_to_pair_projection_shape(self):
        layer = SeqToPairProjection(self.c_s, self.c_z)
        out = layer(self.seq)
        assert out.shape == (self.batch_size, self.seq_len, self.seq_len,
                             self.c_z)

    def test_pair_transition_shape(self):
        layer = PairTransition(self.c_z, expansion=2, dropout=0.0)
        out = layer(self.pair)
        assert out.shape == self.pair.shape

    def test_triangle_multiplication_shapes(self):
        outgoing = TriangleMultiplicationOutgoing(self.c_z)
        incoming = TriangleMultiplicationIncoming(self.c_z)
        out1 = outgoing(self.pair)
        out2 = incoming(self.pair)
        assert out1.shape == self.pair.shape
        assert out2.shape == self.pair.shape

    def test_triangle_attention_shapes(self):
        start = TriangleAttentionStartingNode(self.c_z,
                                              num_heads=4,
                                              dropout=0.0)
        end = TriangleAttentionEndingNode(self.c_z, num_heads=4, dropout=0.0)
        out1 = start(self.pair)
        out2 = end(self.pair)
        assert out1.shape == self.pair.shape
        assert out2.shape == self.pair.shape

    def test_pair_to_seq_bias_shape(self):
        layer = PairToSeqBias(self.c_z, self.c_s)
        out = layer(self.pair)
        assert out.shape == (self.batch_size, self.seq_len, self.c_s)

    def test_structure_track_block_shapes(self):
        layer = StructureTrackBlock(self.c_s,
                                    self.c_z,
                                    num_heads=4,
                                    dropout=0.0)
        seq_out, rot_out, trans_out = layer(self.seq, self.pair, self.rot,
                                            self.trans)
        assert seq_out.shape == self.seq.shape
        assert rot_out.shape == self.rot.shape
        assert trans_out.shape == self.trans.shape

    def test_gradient_flow(self):
        layer = PairTransition(self.c_z, expansion=2, dropout=0.0)
        out = layer(self.pair)
        loss = out.sum()
        loss.backward()
        assert self.pair.grad is not None
        assert self.pair.grad.shape == self.pair.shape
