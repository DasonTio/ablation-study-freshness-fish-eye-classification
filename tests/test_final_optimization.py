import copy
import unittest

from src.final_optimization import build_final_experiments, make_optimized_config


class FinalOptimizationTest(unittest.TestCase):
    def test_build_final_experiments_keeps_all_ablation_scenarios(self):
        experiments = build_final_experiments()

        self.assertEqual(
            [exp["name"] for exp in experiments],
            [
                "final_A_resnet50_no_clahe",
                "final_B_resnet50_clahe",
                "final_C_v2s_no_clahe",
                "final_D_v2s_clahe_proposed",
            ],
        )
        self.assertEqual([exp["use_clahe"] for exp in experiments], [False, True, False, True])

    def test_make_optimized_config_does_not_mutate_base_config(self):
        base = {
            "training": {
                "epochs": 50,
                "learning_rate": 1e-4,
                "dropout": 0.4,
                "patience": 10,
            },
            "models": {
                "resnet50": {"freeze_epochs": 5},
                "efficientnetv2s": {"freeze_epochs": 5},
            },
        }
        original = copy.deepcopy(base)

        optimized = make_optimized_config(base)

        self.assertEqual(base, original)
        self.assertEqual(optimized["training"]["epochs"], 150)
        self.assertEqual(optimized["training"]["dropout"], 0.5)
        self.assertEqual(optimized["training"]["patience"], 25)
        self.assertEqual(optimized["training"]["unfreeze_lr"], 3e-5)


if __name__ == "__main__":
    unittest.main()
