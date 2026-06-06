import torch
import torch.nn as nn
import timm
from torchvision import models


def build_resnet50(num_classes, pretrained=True, dropout=0.4):
    """ResNet50 with ImageNet weights — matches benchmark paper setup."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return model


def build_efficientnetv2s(num_classes, pretrained_tag="tf_efficientnetv2_s.in21k", dropout=0.4):
    """EfficientNetV2-S with ImageNet21k weights — proposed model."""
    model = timm.create_model(
        pretrained_tag,
        pretrained=True,
        num_classes=0,
        global_pool="avg"
    )
    in_features = model.num_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return model


def build_convnext_small(num_classes, dropout=0.4):
    """ConvNeXt-Small with ImageNet-22k → IN-1k weights. Large-kernel depthwise conv
    preserves local contrast signal that CLAHE amplifies.

    timm's ConvNeXt forward routes through `model.head` (NormMlpClassifierHead, which
    does global-pool → norm → flatten → fc). Replace only its `.fc` so pooling/norm are
    kept and our MLP head is actually used. Head params are named `head.*`.
    """
    model = timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=True)
    in_features = model.head.fc.in_features
    model.head.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return model


def freeze_backbone(model, model_name):
    """Freeze all layers except final classifier head."""
    if model_name == "resnet50":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    elif model_name == "efficientnetv2s":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    elif model_name == "convnext_small":
        # timm ConvNeXt head params are named `head.*` (incl. our replaced head.fc)
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
