import torch
import torch.nn as nn
import timm
from app.core.config import settings

LABEL_MAP = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}


class DRModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


_model: DRModel | None = None
_device: torch.device | None = None


def get_model() -> tuple[DRModel, torch.device]:
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = DRModel(
            settings.BACKBONE, settings.NUM_CLASSES, settings.DROPOUT
        ).to(_device)
        ckpt = torch.load(
            settings.CHECKPOINT_PATH, map_location=_device, weights_only=False
        )
        _model.load_state_dict(ckpt["state_dict"])
        _model.eval()
    return _model, _device
