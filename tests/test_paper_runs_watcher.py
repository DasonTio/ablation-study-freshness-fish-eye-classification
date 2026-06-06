import tempfile
import unittest
from pathlib import Path

from scripts.local_watch_paper_runs import missing_required_files


class PaperRunsWatcherTest(unittest.TestCase):
    def test_missing_required_files_for_paper_run_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hier6416").mkdir(parents=True)
            (root / "flat6416").mkdir(parents=True)
            (root / "hier6416" / "ckpt").mkdir(parents=True)
            (root / "flat6416" / "ckpt").mkdir(parents=True)
            (root / "PAPER_RUNS_DONE").write_text("")
            (root / "hier6416" / "hierarchical_ordinal_results.csv").write_text("")
            (root / "hier6416" / "hierarchical_ordinal_summary.csv").write_text("")
            (root / "hier6416" / "ckpt" / "hierarchical_ordinal_swin_tiny_hier_ord_6416_BEST.pth").write_text("")
            (root / "flat6416" / "hierarchical_ordinal_results.csv").write_text("")
            (root / "flat6416" / "ckpt" / "hierarchical_ordinal_swin_tiny_flat_6416_BEST.pth").write_text("")

            missing = missing_required_files(root)

            self.assertEqual(missing, ["flat6416/hierarchical_ordinal_summary.csv"])


if __name__ == "__main__":
    unittest.main()
