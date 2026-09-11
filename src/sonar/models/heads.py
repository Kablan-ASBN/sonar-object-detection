"""Domain classification heads used by the adversarial adapters."""

from __future__ import annotations

from torch import Tensor, nn


class DomainDiscriminator(nn.Sequential):
    """MLP scoring one logit per sample: high for source, low for target."""

    def __init__(self, in_features: int, hidden: int = 1024, dropout: float = 0.5) -> None:
        super().__init__(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )


class ProposalDomainHead(nn.Module):
    """Scores domain at every feature location, then averages to one logit per sample."""

    def __init__(self, in_channels: int = 256, hidden: int = 64) -> None:
        super().__init__()
        # 3x3 first so each score sees the same neighbourhood the RPN scores its anchors over;
        # a 1x1 pair would align per-pixel statistics the proposals never look at.
        self.conv = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.logit = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, feature_map: Tensor) -> Tensor:
        hidden = self.act(self.conv(feature_map))
        # Mean rather than max: one strong location should not carry the whole image.
        return self.logit(hidden).mean(dim=(2, 3))
