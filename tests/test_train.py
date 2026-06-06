import unittest

import torch
import torch.nn as nn

from src.train import build_optimizer


class BuildOptimizerTest(unittest.TestCase):
    def test_includes_frozen_parameters_for_later_unfreeze(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        frozen_param = model[0].weight
        frozen_param.requires_grad = False

        optimizer = build_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=1e-4,
        )

        optimized_params = {
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        }

        self.assertIn(id(frozen_param), optimized_params)


if __name__ == "__main__":
    unittest.main()
