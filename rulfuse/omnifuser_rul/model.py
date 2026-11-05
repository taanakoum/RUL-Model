"""The OmniFuser-RUL model (docs/OMNIFUSER_RUL_PLAN.md §5).

OmniFuser v1.1 (``OmniFuser.encode``) with the paper's dimensions, no metadata codes and
precomputed photo features, plus one linear head on the last fused step: ŷ/H.

Inputs are switched off at the input level, so every arm keeps the same architecture and
usage stays available when force or photos are off:

* force off: the descriptor columns and the signal flag are zero (as a force-missing
  segment, D8);
* photos off: the photo features are zeroed and every step points at slot 0 (the all-zero
  "no photo yet" composite of ``data.Windows``);
* usage off: the usage columns are zero.

Modality dropout (training only, per sample) uses the same two states: with probability
p/2 the force is missing, with p/2 no photo is visible.
"""

from __future__ import annotations

import torch
from omnifuser.config import ModelConfig
from omnifuser.models import OmniFuser
from torch import Tensor, nn

from rulfuse.omnifuser_rul.data import INPUT_STEPS, Layout


class OmniFuserRUL(nn.Module):
    def __init__(
        self,
        layout: Layout,
        views: int,
        view_width: int,
        use_force: bool = True,
        use_photos: bool = True,
        use_usage: bool = True,
        modality_dropout: float = 0.0,
        input_steps: int = INPUT_STEPS,
    ) -> None:
        super().__init__()
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        self.layout = layout
        self.use_force, self.use_photos, self.use_usage = use_force, use_photos, use_usage
        self.modality_dropout = modality_dropout
        config = ModelConfig(
            input_len=input_steps,
            horizon=1,
            force_channels=layout.width,
            prediction_head="direct",
            metadata="none",
            image_encoder="efficientnet_b0",
            pretrained=True,
            freeze_image_encoder=True,
            image_views=max(views, 1),
        )
        backbone = OmniFuser(config, load_pretrained=False)
        # The features are precomputed and OmniFuser's own heads are replaced, so the unused
        # parts hold no parameters and get no gradient.
        backbone.image_branch.encoder.features = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
        backbone.direct_force = None  # type: ignore[assignment]
        backbone.direct_class = None  # type: ignore[assignment]
        if backbone.image_branch.encoder.feature_width != view_width:
            raise ValueError(
                f"photo features must be {backbone.image_branch.encoder.feature_width} wide"
            )
        self.backbone = backbone
        self.head = nn.Linear(config.d_model, 1)

    def _masked(
        self, force: Tensor, images: Tensor, image_indices: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        force = force.clone()
        image_indices = image_indices.clone()
        layout = self.layout
        if not self.use_force:
            force[..., : layout.descriptors + 1] = 0.0
        if not self.use_usage:
            force[..., layout.usage] = 0.0
        if not self.use_photos:  # whatever the data passes in
            images = torch.zeros_like(images)
            image_indices.zero_()
        if self.training and self.modality_dropout > 0.0:
            draw = torch.rand(force.shape[0], device=force.device)
            no_force = draw < self.modality_dropout / 2
            no_photo = (draw >= self.modality_dropout / 2) & (draw < self.modality_dropout)
            force[no_force, :, : layout.descriptors + 1] = 0.0
            image_indices[no_photo] = 0  # slot 0 is the all-zero composite (data.Windows)
        return force, images, image_indices

    def forward(self, force: Tensor, images: Tensor, image_indices: Tensor) -> Tensor:
        """(B, T, width), (B, M, views x 1280), (B, T) -> (B,) predicted y/H."""
        force, images, image_indices = self._masked(force, images, image_indices)
        fused = self.backbone.encode(force, images, image_indices)
        return self.head(fused[:, -1]).squeeze(-1)
