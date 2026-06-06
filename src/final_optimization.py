import copy


def build_final_experiments():
    """Same four scenarios as ablation, trained with the optimized protocol."""
    return [
        {"name": "final_A_resnet50_no_clahe", "backbone": "resnet50", "model_name": "resnet50", "use_clahe": False},
        {"name": "final_B_resnet50_clahe", "backbone": "resnet50", "model_name": "resnet50", "use_clahe": True},
        {"name": "final_C_v2s_no_clahe", "backbone": "efficientnetv2s", "model_name": "efficientnetv2s", "use_clahe": False},
        {"name": "final_D_v2s_clahe_proposed", "backbone": "efficientnetv2s", "model_name": "efficientnetv2s", "use_clahe": True},
    ]


def make_optimized_config(base_cfg):
    """Return a stronger, fair final-training config without mutating the ablation config."""
    cfg = copy.deepcopy(base_cfg)
    cfg["training"]["epochs"] = 150
    cfg["training"]["patience"] = 25
    cfg["training"]["dropout"] = 0.5
    cfg["training"]["learning_rate"] = 1e-4
    cfg["training"]["unfreeze_lr"] = 3e-5
    cfg["training"]["label_smoothing"] = 0.05
    cfg["training"]["amp"] = True
    cfg["training"]["channels_last"] = True
    return cfg
