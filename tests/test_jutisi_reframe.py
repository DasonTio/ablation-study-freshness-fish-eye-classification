from pathlib import Path


def test_paper_builder_declares_final_model_positioning():
    import scripts.build_jutisi_final_paper as paper

    assert paper.TITLE_EN == "Global Ocular Statistics Validate GAP Swin-Tiny for Fish-Eye Freshness Classification"
    assert "GAP Swin-Tiny + 24-class head" in paper.FINAL_MODEL_STATEMENT
    assert "diagnostic" in paper.GLOBAL_SIGNAL_ROLE.lower()


def test_stage_xai_specs_define_correlated_three_stage_panel():
    from scripts.make_stage_xai_figure import stage_specs, strip_compile_prefix

    specs = stage_specs()
    assert [spec.key for spec in specs] == ["clahe", "second_order", "final_gap"]
    assert "CLAHE" in specs[0].title
    assert "Second-Order" in specs[1].title
    assert "Final GAP Swin-Tiny" in specs[2].title
    assert all(Path(spec.checkpoint).exists() for spec in specs)

    stripped = strip_compile_prefix({
        "_orig_mod.backbone.weight": 1,
        "class_head.bias": 2,
    })
    assert stripped == {
        "backbone.weight": 1,
        "class_head.bias": 2,
    }
