import torch.nn as nn

class HFClassifierWrapper(nn.Module):
    """Adapter: biến output .logits của transformers thành tensor logits thẳng."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, pixel_values):
        outputs = self.hf_model(pixel_values=pixel_values)
        return outputs.logits

    def state_dict(self, *args, **kwargs):
        return self.hf_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.hf_model.load_state_dict(state_dict, *args, **kwargs)



def _build_hf_model(config: dict, device: str) -> nn.Module:
    from transformers import AutoModelForImageClassification
    hf_model = AutoModelForImageClassification.from_pretrained(
        config["model"]["hf_name"],
        num_labels=config["model"]["num_classes"],
        ignore_mismatched_sizes=True,
    )
    wrapped = HFClassifierWrapper(hf_model)
    return wrapped.to(device)

def build_model(config: dict, device: str) -> nn.Module:
    backend = config["model"].get("backend", "self")

    if backend == "self":
        return _build_self_vit(config, device)
    elif backend == "timm":
        return _build_timm_model(config, device)
    elif backend == "hf":
        return _build_hf_model(config, device)
    else:
        raise ValueError(f"Backend không hỗ trợ: {backend!r} (chỉ nhận 'self' hoặc 'timm')")

def _build_self_vit(config: dict, device: str) -> nn.Module:
    from model.vit import ViT
    return ViT(
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=3,
        n_classes=config["model"]["num_classes"],
        embed_dim=config["model"]["embed_dim"],
        depth=config["model"]["depth"],
        n_heads=config["model"]["num_heads"],
    ).to(device)

def _build_timm_model(config: dict, device: str) -> nn.Module:
    import timm 
    model = timm.create_model(
        config["model"]["timm_name"],
        pretrained=True,
        num_classes=config["model"]["num_classes"],
    )
    return model.to(device)


