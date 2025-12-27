"""Self-supervised learning tasks."""
import copy
import math
from dataclasses import dataclass, field

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trainformer.types import DatasetInfo


@dataclass
class SSL:
    """Self-supervised learning tasks.

    Supports multiple SSL methods: SimCLR, MoCo, DINO, MAE, BYOL.

    Args:
        method: SSL method ('simclr', 'moco', 'dino', 'mae', 'byol')
        backbone: Name of timm backbone
        pretrained: Use pretrained weights (usually False for SSL)
        proj_dim: Projection head dimension
        temperature: Contrastive temperature
        momentum: EMA momentum for momentum methods
    """

    method: str
    backbone: str = "resnet50"
    pretrained: bool = False
    proj_dim: int = 128
    temperature: float = 0.1
    momentum: float = 0.996

    model: nn.Module = field(init=False)

    def __post_init__(self):
        if self.method == "simclr":
            self.model = SimCLR(self.backbone, self.proj_dim, self.temperature, self.pretrained)
        elif self.method == "moco":
            self.model = MoCo(
                self.backbone, self.proj_dim, self.temperature, self.momentum, self.pretrained
            )
        elif self.method == "byol":
            self.model = BYOL(self.backbone, self.proj_dim, self.momentum, self.pretrained)
        elif self.method == "dino":
            self.model = DINO(self.backbone, self.momentum, self.pretrained)
        elif self.method == "mae":
            self.model = MAE(self.backbone, self.pretrained)
        else:
            raise ValueError(f"Unknown SSL method: {self.method}")

    @classmethod
    def simclr(cls, backbone: str = "resnet50", **kw) -> "SSL":
        return cls(method="simclr", backbone=backbone, **kw)

    @classmethod
    def moco(cls, backbone: str = "resnet50", **kw) -> "SSL":
        return cls(method="moco", backbone=backbone, **kw)

    @classmethod
    def byol(cls, backbone: str = "resnet50", **kw) -> "SSL":
        return cls(method="byol", backbone=backbone, **kw)

    @classmethod
    def dino(cls, backbone: str = "vit_small_patch16_224", **kw) -> "SSL":
        return cls(method="dino", backbone=backbone, **kw)

    @classmethod
    def mae(cls, backbone: str = "vit_base_patch16_224", **kw) -> "SSL":
        return cls(method="mae", backbone=backbone, **kw)

    def configure(self, info: DatasetInfo) -> None:
        """SSL doesn't need dataset configuration."""
        pass

    def parameters(self):
        return self.model.parameters()

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        x, _ = batch  # Labels unused in SSL
        return self.model.training_step(x)

    def eval_step(self, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        x, y = batch
        embeddings = self.model.embed(x)
        return {"embeddings": embeddings, "labels": y}

    def on_step_end(self, step: int, max_steps: int):
        if hasattr(self.model, "on_step_end"):
            self.model.on_step_end(step, max_steps)

    def state_dict(self) -> dict:
        if hasattr(self.model, "state_dict_extra"):
            return self.model.state_dict_extra()
        return {}

    def load_state_dict(self, state: dict):
        if hasattr(self.model, "load_state_dict_extra"):
            self.model.load_state_dict_extra(state)


class SimCLR(nn.Module):
    """SimCLR: contrastive learning with two views."""

    def __init__(self, backbone: str, proj_dim: int, temperature: float, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.num_features, self.backbone.num_features),
            nn.ReLU(),
            nn.Linear(self.backbone.num_features, proj_dim),
        )
        self.temperature = temperature
        self.augment = SimCLRAugmentation()

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, x: Tensor) -> Tensor:
        v1, v2 = self.augment(x)
        z1 = F.normalize(self.projector(self.backbone(v1)), dim=-1)
        z2 = F.normalize(self.projector(self.backbone(v2)), dim=-1)
        return nt_xent_loss(z1, z2, self.temperature)


class MoCo(nn.Module):
    """MoCo v2: momentum contrast with queue."""

    def __init__(
        self, backbone: str, proj_dim: int, temperature: float, momentum: float, pretrained: bool
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.projector = self._build_projector(proj_dim)

        # Momentum encoder
        self.backbone_m = copy.deepcopy(self.backbone)
        self.projector_m = copy.deepcopy(self.projector)
        self._freeze(self.backbone_m)
        self._freeze(self.projector_m)

        # Queue
        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, 65536), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.temperature = temperature
        self.momentum = momentum
        self.augment = SimCLRAugmentation()

    def _build_projector(self, proj_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.backbone.num_features, self.backbone.num_features),
            nn.ReLU(),
            nn.Linear(self.backbone.num_features, proj_dim),
        )

    def _freeze(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, x: Tensor) -> Tensor:
        v1, v2 = self.augment(x)

        q = F.normalize(self.projector(self.backbone(v1)), dim=-1)

        with torch.no_grad():
            k = F.normalize(self.projector_m(self.backbone_m(v2)), dim=-1)

        loss = infonce_with_queue(q, k, self.queue, self.temperature)
        self._enqueue(k)
        return loss

    def on_step_end(self, step: int, max_steps: int):
        for p_q, p_k in zip(self.backbone.parameters(), self.backbone_m.parameters()):
            p_k.data = self.momentum * p_k.data + (1 - self.momentum) * p_q.data
        for p_q, p_k in zip(self.projector.parameters(), self.projector_m.parameters()):
            p_k.data = self.momentum * p_k.data + (1 - self.momentum) * p_q.data

    @torch.no_grad()
    def _enqueue(self, keys: Tensor):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        queue_size = self.queue.shape[1]

        if ptr + batch_size > queue_size:
            self.queue[:, ptr:] = keys[: queue_size - ptr].T
            self.queue[:, : batch_size - (queue_size - ptr)] = keys[queue_size - ptr :].T
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T

        self.queue_ptr[0] = (ptr + batch_size) % queue_size

    def state_dict_extra(self) -> dict:
        return {"queue": self.queue, "queue_ptr": self.queue_ptr}

    def load_state_dict_extra(self, state: dict):
        self.queue.copy_(state["queue"])
        self.queue_ptr.copy_(state["queue_ptr"])


class BYOL(nn.Module):
    """BYOL: Bootstrap Your Own Latent."""

    def __init__(self, backbone: str, proj_dim: int, momentum: float, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.projector = self._build_projector(proj_dim)
        self.predictor = self._build_predictor(proj_dim)

        self.backbone_t = copy.deepcopy(self.backbone)
        self.projector_t = copy.deepcopy(self.projector)
        self._freeze(self.backbone_t)
        self._freeze(self.projector_t)

        self.momentum = momentum
        self.augment = SimCLRAugmentation()

    def _build_projector(self, proj_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.backbone.num_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, proj_dim),
        )

    def _build_predictor(self, proj_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(proj_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, proj_dim),
        )

    def _freeze(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, x: Tensor) -> Tensor:
        v1, v2 = self.augment(x)

        z1 = self.projector(self.backbone(v1))
        z2 = self.projector(self.backbone(v2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        with torch.no_grad():
            t1 = self.projector_t(self.backbone_t(v1))
            t2 = self.projector_t(self.backbone_t(v2))

        loss = byol_loss(p1, t2) + byol_loss(p2, t1)
        return loss / 2

    def on_step_end(self, step: int, max_steps: int):
        for p_o, p_t in zip(self.backbone.parameters(), self.backbone_t.parameters()):
            p_t.data = self.momentum * p_t.data + (1 - self.momentum) * p_o.data
        for p_o, p_t in zip(self.projector.parameters(), self.projector_t.parameters()):
            p_t.data = self.momentum * p_t.data + (1 - self.momentum) * p_o.data


class DINO(nn.Module):
    """DINO: self-distillation with no labels."""

    def __init__(self, backbone: str, momentum: float, pretrained: bool):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = DINOHead(self.backbone.num_features, 65536)

        # Teacher
        self.backbone_t = copy.deepcopy(self.backbone)
        self.head_t = copy.deepcopy(self.head)
        self._freeze(self.backbone_t)
        self._freeze(self.head_t)

        # Center for teacher outputs
        self.register_buffer("center", torch.zeros(1, 65536))

        self.momentum = momentum
        self.center_momentum = 0.9
        self.student_temp = 0.1
        self.teacher_temp = 0.04
        self.augment = MultiCropAugmentation()

    def _freeze(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, x: Tensor) -> Tensor:
        views = self.augment(x)  # {"global": [g1, g2], "local": [l1, ..., l8]}

        # Student: all views
        student_out = []
        for crop in views["global"] + views.get("local", []):
            student_out.append(self.head(self.backbone(crop)))

        # Teacher: global only, with centering
        with torch.no_grad():
            teacher_out = []
            for crop in views["global"]:
                out = self.head_t(self.backbone_t(crop))
                teacher_out.append(out - self.center)

            # Update center
            teacher_mean = torch.cat(teacher_out).mean(dim=0, keepdim=True)
            self.center = self.center * self.center_momentum + teacher_mean * (1 - self.center_momentum)

        return dino_loss(student_out, teacher_out, self.student_temp, self.teacher_temp)

    def on_step_end(self, step: int, max_steps: int):
        # Cosine momentum schedule
        m = 1 - (1 - self.momentum) * (1 + math.cos(math.pi * step / max_steps)) / 2

        for p_s, p_t in zip(self.backbone.parameters(), self.backbone_t.parameters()):
            p_t.data = m * p_t.data + (1 - m) * p_s.data
        for p_s, p_t in zip(self.head.parameters(), self.head_t.parameters()):
            p_t.data = m * p_t.data + (1 - m) * p_s.data

    def state_dict_extra(self) -> dict:
        return {"center": self.center}

    def load_state_dict_extra(self, state: dict):
        self.center.copy_(state["center"])


class MAE(nn.Module):
    """MAE: Masked Autoencoder for self-supervised ViT pretraining."""

    def __init__(self, backbone: str, pretrained: bool, mask_ratio: float = 0.75):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.mask_ratio = mask_ratio

        # Get patch embedding dimensions
        if hasattr(self.backbone, "patch_embed"):
            self.patch_size = self.backbone.patch_embed.patch_size[0]
            self.embed_dim = self.backbone.embed_dim
        else:
            # Fallback for non-ViT models
            self.patch_size = 16
            self.embed_dim = self.backbone.num_features

        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.patch_size ** 2 * 3),
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def training_step(self, x: Tensor) -> Tensor:
        # Patchify
        patches = self._patchify(x)
        B, N, D = patches.shape

        # Random masking
        num_masked = int(N * self.mask_ratio)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep visible patches
        ids_keep = ids_shuffle[:, :N - num_masked]

        # Get patch embeddings through backbone's patch_embed
        if hasattr(self.backbone, "patch_embed"):
            patch_embed = self.backbone.patch_embed(x)
            visible_encoded = torch.gather(
                patch_embed, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )
        else:
            # Fallback: use full forward
            visible_encoded = self.backbone(x).unsqueeze(1)

        # Add mask tokens and restore order
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        full = torch.cat([visible_encoded, mask_tokens], dim=1)

        # Ensure we have the right shape for gathering
        if full.shape[1] == N:
            full = torch.gather(full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Decode
        pred = self.decoder(full)

        # Reconstruction loss on masked patches only
        target = patches
        mask = torch.zeros(B, N, device=x.device)
        mask.scatter_(1, ids_shuffle[:, N - num_masked:], 1)

        loss = (pred - target) ** 2
        loss = (loss.mean(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
        return loss

    def _patchify(self, x: Tensor) -> Tensor:
        """Convert image to patches."""
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # B, h, w, p, p, C
        x = x.reshape(B, h * w, p * p * C)
        return x


class DINOHead(nn.Module):
    """DINO projection head with weight normalization."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# ─── Augmentation ───────────────────────────────────────────────


class SimCLRAugmentation(nn.Module):
    """SimCLR-style augmentation (two views)."""

    def __init__(self, size: int = 224):
        super().__init__()
        from torchvision import transforms as T

        self.transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=size // 10 * 2 + 1),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.transform(x), self.transform(x)


class MultiCropAugmentation(nn.Module):
    """DINO multi-crop augmentation (2 global + N local views)."""

    def __init__(self, global_size: int = 224, local_size: int = 96, num_local: int = 6):
        super().__init__()
        from torchvision import transforms as T

        self.global_transform = T.Compose([
            T.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.local_transform = T.Compose([
            T.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.num_local = num_local

    def forward(self, x: Tensor) -> dict[str, list[Tensor]]:
        return {
            "global": [self.global_transform(x), self.global_transform(x)],
            "local": [self.local_transform(x) for _ in range(self.num_local)],
        }


# ─── Loss Functions ─────────────────────────────────────────────


def nt_xent_loss(z1: Tensor, z2: Tensor, temperature: float = 0.1) -> Tensor:
    """NT-Xent loss (SimCLR)."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)

    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature

    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)], dim=0).to(z.device)

    return F.cross_entropy(sim, labels)


def infonce_with_queue(q: Tensor, k: Tensor, queue: Tensor, temperature: float = 0.1) -> Tensor:
    """InfoNCE loss with memory queue (MoCo)."""
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    l_neg = torch.einsum("nc,ck->nk", [q, queue])

    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)


def byol_loss(p: Tensor, z: Tensor) -> Tensor:
    """BYOL regression loss."""
    p = F.normalize(p, dim=-1, p=2)
    z = F.normalize(z, dim=-1, p=2)
    return 2 - 2 * (p * z).sum(dim=-1).mean()


def dino_loss(
    student_out: list[Tensor],
    teacher_out: list[Tensor],
    student_temp: float,
    teacher_temp: float,
) -> Tensor:
    """DINO cross-entropy loss between student and teacher."""
    total_loss = 0.0
    n_loss_terms = 0

    for t_idx, t in enumerate(teacher_out):
        t_soft = F.softmax(t / teacher_temp, dim=-1)

        for s_idx, s in enumerate(student_out):
            if t_idx == s_idx:
                continue  # Skip same view
            s_log_soft = F.log_softmax(s / student_temp, dim=-1)
            total_loss = total_loss + torch.sum(-t_soft * s_log_soft, dim=-1).mean()
            n_loss_terms += 1

    return total_loss / n_loss_terms
