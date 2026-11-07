"""The OmniFuser-RUL model (docs/OMNIFUSER_RUL_PLAN.md §5; version 6).

OmniFuser v1.1 (``OmniFuser.encode``) with the paper's dimensions, no metadata codes and
precomputed photo features reads the 96 most recent steps. A ``CausalMemory`` (O8, D21)
carries everything older: a GRU over the unit's whole history, one input per block of
``MemorySpec.block`` steps (96 by default: one OmniFuser window), and that input is
**OmniFuser's own encoding of the block**:

* the window of 96 steps ending at the block's last step goes through OmniFuser's force
  encoder (temporal embedding + multi-resolution extractor) and its image branch (photo
  projection + graph attention), the same modules and weights the head's window uses;
* each stream is averaged over the window's steps; the GRU input is [force ; photos]
  (2 × d = 64 numbers);
* step t reads the state after the last block complete at t, so it never depends on a step
  after t. OmniFuser mixes all 96 steps of a window, so a block is encoded only once all
  its steps exist. With 96-step blocks the windows tile the history exactly once.

Gradients reach the encoders through the memory as well as through the head, and training
and inference compute the memory the same way (``memory_states``). The head reads the last
fused step and the memory state: ŷ/H = Linear([F_T ; m_T]).

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
    block: int = INPUT_STEPS  # steps per memory block: one OmniFuser window (96 s at 1 s)
    cell: str = "gru"


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
        self.input_steps = input_steps
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
            self.memory = CausalMemory(2 * config.d_model, memory.hidden, memory.block, memory.cell)
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

    def encoded_streams(
        self, force: Tensor, images: Tensor, image_indices: Tensor
    ) -> tuple[Tensor, Tensor]:
        """(B, T, width), (B, m, F), (B, T) -> OmniFuser's encoded force series and encoded
        photos, each (B, T, d): the two streams ``OmniFuser.encode`` fuses (RevIN, metadata
        and OmniFuser's own modality dropout are off here, so these are exactly its inputs
        to the cross-modal fusion)."""
        backbone = self.backbone
        sensor = backbone.temporal_encoder(backbone.temporal_embedding(force))
        return sensor, backbone.image_branch(images, image_indices)

    def memory_states(
        self,
        force: Tensor,
        composites: Tensor,
        comp_index: Tensor,
        lengths: Tensor,
        variant: int = NO_DROP,
    ) -> Tensor:
        """(U, n, width), (U, m, F), (U, n), (U,) -> (U, n, hidden) memory at every step of
        each unit (inputs in time order, padded at the end). Block j of a unit is the window
        of ``input_steps`` steps ending at its last step, (j + 1)·block − 1, padded at the
        front with empty steps as the head's windows are; its GRU input is the mean over the
        window of OmniFuser's encoded force and encoded photos. Only blocks a unit completes
        are encoded. ``variant`` DROP_FORCE / DROP_PHOTO: the memory of the same unit
        without force or without photos (for modality dropout)."""
        spec = self.memory_spec
        if spec is None:
            raise RuntimeError("this model has no memory")
        units, steps, width = force.shape
        block, window = spec.block, self.input_steps
        n_blocks = steps // block
        force = self._switch_force(force)
        if variant == DROP_FORCE:
            force[..., : self.layout.descriptors + 1] = 0.0
        if not (self.use_photos and variant != DROP_PHOTO):
            composites = torch.zeros_like(composites)
            comp_index = torch.zeros_like(comp_index)
        front = window - 1
        force = torch.cat([force.new_zeros(units, front, width), force], dim=1)
        comp_index = torch.cat([comp_index.new_zeros(units, front), comp_index], dim=1)
        ends = torch.arange(n_blocks, device=force.device) * block + (block - 1)
        positions = ends[:, None] + torch.arange(window, device=force.device)[None]
        d = self.backbone.config.d_model
        encoded = []
        for u in range(units):
            done = int((ends < lengths[u]).sum())  # blocks this unit completes
            rows = force.new_zeros(n_blocks, 2 * d)
            if done:
                sensor, photo = self.encoded_streams(
                    force[u, positions[:done]],
                    composites[u : u + 1].expand(done, -1, -1),
                    comp_index[u, positions[:done]],
                )
                rows = torch.cat([torch.cat([sensor.mean(1), photo.mean(1)], -1), rows[done:]])
            encoded.append(rows)
        return self.memory.forward_blocks(torch.stack(encoded), steps)

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
