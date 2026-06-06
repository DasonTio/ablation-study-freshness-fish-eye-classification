import torch

from src.secondorder import SecondOrderPool, to_bcm
from src.secondorder_model import PooledHierarchicalClassifier


def test_to_bcm_handles_swin_convnext_and_token_shapes():
    swin = torch.randn(2, 7, 7, 16)
    convnext = torch.randn(2, 16, 7, 7)
    tokens = torch.randn(2, 49, 16)

    assert to_bcm(swin, 16).shape == (2, 16, 49)
    assert to_bcm(convnext, 16).shape == (2, 16, 49)
    assert to_bcm(tokens, 16).shape == (2, 16, 49)


def test_raw_moment_preserves_global_mean_signal_that_centered_cov_removes():
    x = torch.ones(2, 7, 7, 8)
    raw = SecondOrderPool(in_channels=8, proj_dim=None, mode="raw_bilinear")
    centered = SecondOrderPool(in_channels=8, proj_dim=None, mode="centered_cov")

    raw_out = raw(x)
    centered_out = centered(x)

    assert raw_out.abs().sum() > 0
    assert centered_out.abs().sum() < 1e-5


def test_pool_output_dim_is_upper_triangle_after_projection():
    pool = SecondOrderPool(in_channels=16, proj_dim=8, mode="raw_bilinear")
    out = pool(torch.randn(2, 7, 7, 16))

    assert out.shape == (2, 8 * 9 // 2)
    assert pool.out_dim == 8 * 9 // 2


def test_gradient_flows_through_second_order_pool():
    x = torch.randn(2, 7, 7, 8, requires_grad=True)
    pool = SecondOrderPool(in_channels=8, proj_dim=4, mode="raw_bilinear")

    pool(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_pooled_classifier_returns_hierarchical_contract_for_gap_and_second_order():
    for pooling in ["gap", "raw_bilinear", "gap_raw_bilinear", "centered_cov"]:
        model = PooledHierarchicalClassifier(
            backbone_tag="resnet18",
            num_classes=24,
            num_species=8,
            pooling=pooling,
            proj_dim=16,
            pretrained=False,
        )

        out = model(torch.randn(2, 3, 64, 64))

        assert set(out) == {"class", "species", "freshness"}
        assert out["class"].shape == (2, 24)
        assert out["species"].shape == (2, 8)
        assert out["freshness"].shape == (2, 2)
