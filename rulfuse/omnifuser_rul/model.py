"""The OmniFuser-RUL model (docs/OMNIFUSER_RUL_PLAN.md §5; version 3).

OmniFuser v1.1 (``OmniFuser.encode``) with the paper's dimensions, no metadata codes and
precomputed photo features reads the 96 most recent steps. A ``CausalMemory`` (O8, D21)
carries everything older: a GRU over the unit's whole history, in 10-step blocks, of the
same step inputs (force vector, projected photo composite, photo flag). The head reads the
last fused step and the memory state: ŷ/H = Linear([F_T ; m_T]).

Inputs are switched off at the input level, in the window and in the memory alike, so
every arm keeps the same architecture and usage stays available when force or photos are
off:

* force off: the descriptor columns and the signal flag are zero (as a force-missing
  segment, D8);
* photos off: the photo features are zeroed and every step points at row 0 (the all-zero
  "no photo yet" composite);
* usage off: the usage columns are zero.

Modality dropout (training only, per sample) uses the same two states: code 1, the force
is missing; code 2, no photo is visible. ``draw_dropout`` draws the codes, so the training
loop can give each sample the matching memory variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from omnifuser.config import ModelConfig
from omnifuser.models import OmniFuser
from torch import Tensor, nn

from rulfuse.models.memory import CausalMemory
from rulfuse.omnifuser_rul.data import INPUT_STEPS, Layout

NO_DROP, DROP_FORCE, DROP_PHOTO = 0, 1, 2


@dataclass(frozen=True)
class MemorySpec:
    hidden: int = 64
    block: int = 10  # steps per memory block (10 s at 1 s steps)
    cell: str = "gru"
    photo_dim: int = 32  # the memory's own projection of the photo composites


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
        memory: MemorySpec | None = None,
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
        self.memory_spec = memory
        head_width = config.d_model
        if memory is not None:
            self.photo_projection = nn.Linear(max(views, 1) * view_width, memory.photo_dim)
            self.memory = CausalMemory(
                layout.width + memory.photo_dim + 1, memory.hidden, memory.block, memory.cell
            )
            head_width += memory.hidden
        self.head = nn.Linear(head_width, 1)

    # ------------------------------------------------------------------ input switches

    def _switch_force(self, force: Tensor, drop_force: Tensor | None = None) -> Tensor:
        force = force.clone()
        layout = self.layout
        if not self.use_force:
            force[..., : layout.descriptors + 1] = 0.0
        if not self.use_usage:
            force[..., layout.usage] = 0.0
        if drop_force is not None and drop_force.any():
            force[drop_force, ..., : layout.descriptors + 1] = 0.0
        return force

    def draw_dropout(self, batch: int, device) -> Tensor | None:
        """Per-sample codes (NO_DROP, DROP_FORCE, DROP_PHOTO), in training only."""
        if not (self.training and self.modality_dropout > 0.0):
            return None
        draw = torch.rand(batch, device=device)
        codes = torch.full((batch,), NO_DROP, device=device, dtype=torch.long)
        codes[draw < self.modality_dropout / 2] = DROP_FORCE
        codes[(draw >= self.modality_dropout / 2) & (draw < self.modality_dropout)] = DROP_PHOTO
        return codes

    # ------------------------------------------------------------------ memory

    def memory_states(
        self,
        force: Tensor,
        composites: Tensor,
        comp_index: Tensor,
        lengths: Tensor,
        variant: int = NO_DROP,
    ) -> Tensor:
        """(U, n, width), (U, m, F), (U, n), (U,) -> (U, n, hidden) memory at every step of
        each unit. ``variant`` DROP_FORCE / DROP_PHOTO: the memory of the same unit without
        force or without photos (for modality dropout)."""
        if self.memory_spec is None:
            raise RuntimeError("this model has no memory")
        force = self._switch_force(force)
        if variant == DROP_FORCE:
            force[..., : self.layout.descriptors + 1] = 0.0
        photos_on = self.use_photos and variant != DROP_PHOTO
        if photos_on:
            projected = self.photo_projection(composites)  # (U, m, photo_dim)
            photo = torch.gather(
                projected, 1, comp_index.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
            )
            flag = (comp_index > 0).to(force.dtype).unsqueeze(-1)
        else:
            photo = force.new_zeros(*force.shape[:2], self.memory_spec.photo_dim)
            flag = force.new_zeros(*force.shape[:2], 1)
        return self.memory(torch.cat([force, photo, flag], dim=-1), lengths)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        force: Tensor,
        images: Tensor,
        image_indices: Tensor,
        memory_state: Tensor | None = None,
        drop: Tensor | None = None,
    ) -> Tensor:
        """(B, T, width), (B, M, views x 1280), (B, T), memory (B, hidden) -> (B,) ŷ/H."""
        drop_force = None if drop is None else drop == DROP_FORCE
        force = self._switch_force(force, drop_force)
        image_indices = image_indices.clone()
        if not self.use_photos:  # whatever the data passes in
            images = torch.zeros_like(images)
            image_indices.zero_()
        if drop is not None:
            image_indices[drop == DROP_PHOTO] = 0  # row 0 is the all-zero composite
        fused = self.backbone.encode(force, images, image_indices)[:, -1]
        if self.memory_spec is not None:
            if memory_state is None:
                raise ValueError("a model with memory needs memory_state")
            fused = torch.cat([fused, memory_state], dim=-1)
        return self.head(fused).squeeze(-1)
