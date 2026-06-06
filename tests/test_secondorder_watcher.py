import tempfile
from pathlib import Path

from scripts.local_watch_secondorder import missing_required_files


def test_missing_required_files_for_secondorder_layout():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "checkpoints").mkdir()
        (root / "SECONDORDER_DONE").write_text("")
        (root / "secondorder_results.csv").write_text("")
        (root / "secondorder_predictions.csv").write_text("")
        (root / "secondorder_summary.csv").write_text("")
        (root / "secondorder_significance.json").write_text("")
        (root / "checkpoints" / "A_gap_BEST.pth").write_text("")
        (root / "checkpoints" / "B_raw_bilinear_BEST.pth").write_text("")
        (root / "checkpoints" / "C_gap_raw_bilinear_BEST.pth").write_text("")

        assert missing_required_files(root) == ["checkpoints/D_centered_cov_BEST.pth"]
