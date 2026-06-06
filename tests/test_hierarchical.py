import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.hierarchical import (
    FRESHNESS_TO_RANK,
    HierarchicalClassifier,
    HierarchicalFFEDataset,
    build_hierarchical_metadata,
    coral_loss,
    coral_targets,
    parse_ffe_class_name,
)


class HierarchicalLabelsTest(unittest.TestCase):
    def test_parse_class_name_handles_extra_whitespace(self):
        parsed = parse_ffe_class_name("Nibea Albiflora -  Highly Fresh")

        self.assertEqual(parsed.species, "Nibea Albiflora")
        self.assertEqual(parsed.freshness, "Highly Fresh")
        self.assertEqual(parsed.freshness_rank, FRESHNESS_TO_RANK["Highly Fresh"])

    def test_coral_targets_encode_ordered_thresholds(self):
        ranks = torch.tensor([0, 1, 2])

        encoded = coral_targets(ranks, num_classes=3)

        expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        torch.testing.assert_close(encoded, expected)

    def test_coral_loss_is_small_for_correct_rank_order(self):
        logits = torch.tensor([[-4.0, -5.0], [4.0, -4.0], [5.0, 4.0]])
        ranks = torch.tensor([0, 1, 2])

        loss = coral_loss(logits, ranks, num_classes=3)

        self.assertLess(loss.item(), 0.05)

    def test_dataset_returns_class_species_and_freshness_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class_dir = root / "Chanos Chanos - Fresh"
            class_dir.mkdir()
            image_path = class_dir / "IMG_1.jpg"
            Image.new("RGB", (16, 16), color=(120, 100, 80)).save(image_path)

            samples, metadata = build_hierarchical_metadata(root)
            dataset = HierarchicalFFEDataset(samples, transform=None)

            image, target = dataset[0]

            self.assertEqual(image.size, (16, 16))
            self.assertEqual(target["class"].item(), 0)
            self.assertEqual(target["species"].item(), metadata.species_to_idx["Chanos Chanos"])
            self.assertEqual(target["freshness"].item(), FRESHNESS_TO_RANK["Fresh"])

    def test_model_outputs_all_required_heads(self):
        model = HierarchicalClassifier(
            backbone_tag="resnet18",
            num_classes=24,
            num_species=8,
            pretrained=False,
        )

        outputs = model(torch.zeros(2, 3, 64, 64))

        self.assertEqual(outputs["class"].shape, (2, 24))
        self.assertEqual(outputs["species"].shape, (2, 8))
        self.assertEqual(outputs["freshness"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
