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
        model_name = config["model"].get("name", "vit").lower()
        if model_name == "vit":
            return _build_self_vit(config, device)
        elif model_name == "resnet":
            return _build_resnet(config, device)
        elif model_name == "densenet":
            return _build_densenet(config, device)
        else:
            raise ValueError(f"Model không hỗ trợ: {model_name!r} (chỉ nhận 'vit' hoặc 'resnet')")
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

def _build_resnet(config: dict, device: str) -> nn.Module:
    from model.resnet import ResNet, ResidualBlock # Giả sử bạn để BasicBlock trong cùng file
    

    n_blocks_lst = config["model"].get("n_blocks_lst", [2, 2, 2, 2])
    
    return ResNet(
        residual_block=ResidualBlock, 
        n_blocks_lst=n_blocks_lst,
        n_classes=config["model"]["num_classes"]
    ).to(device)
def _build_densenet(config: dict, device: str) -> nn.Module:
    from model.densenet import DenseNet

    num_blocks = config["model"].get("num_blocks", [6, 12, 24, 16])
    growth_rate = config["model"].get("growth_rate", 32)

    return DenseNet(
        num_blocks=num_blocks,
        growth_rate=growth_rate,
        num_classes=config["model"]["num_classes"]
    ).to(device)