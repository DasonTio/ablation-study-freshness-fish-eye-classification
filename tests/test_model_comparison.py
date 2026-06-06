import unittest
from pathlib import Path

from src.model_comparison import available_model_specs, get_model_spec


class ModelComparisonRegistryTest(unittest.TestCase):
    def test_available_model_specs_only_returns_existing_checkpoints(self):
        specs = available_model_specs()

        self.assertGreaterEqual(len(specs), 1)
        for spec in specs:
            self.assertTrue(Path(spec.checkpoint_path).exists(), spec.checkpoint_path)

    def test_hierarchical_model_is_registered_with_xai(self):
        spec = get_model_spec("hierarchical_swin_tiny")

        self.assertEqual(spec.kind, "hierarchical")
        self.assertTrue(spec.xai_supported)
        self.assertEqual(spec.display_name, "Hierarchical Ordinal Swin-Tiny")


if __name__ == "__main__":
    unittest.main()
