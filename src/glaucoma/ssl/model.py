"""MoCo-v2 model: ConvNeXt-Tiny encoder + MLP projector + FIFO key queue."""

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny

ENCODER_NAME = "convnext_tiny"
FEATURE_DIM = 768


def build_encoder(stochastic_depth_prob=0.0):
    """Image encoder identical in structure to ``ConvNeXtClassifier.encoder``."""
    if (type(stochastic_depth_prob) is bool
            or not isinstance(stochastic_depth_prob, (int, float))
            or not 0 <= stochastic_depth_prob <= 1):
        raise ValueError("stochastic_depth_prob must be in [0, 1]")
    backbone = convnext_tiny(weights=None, stochastic_depth_prob=float(stochastic_depth_prob))
    return nn.Sequential(backbone.features, backbone.avgpool, backbone.classifier[0], nn.Flatten(1))


def build_projector(in_dim=FEATURE_DIM, hidden_dim=2048, out_dim=128):
    if not all(isinstance(value, int) and value > 0 for value in (in_dim, hidden_dim, out_dim)):
        raise ValueError("projector dimensions must be positive integers")
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))


class MoCoV2(nn.Module):
    """Online encoder + momentum encoder and a FIFO queue of negative keys.

    ``forward`` returns ``(logits, targets)`` where ``logits`` has shape
    ``[B, queue_size + 1]`` and ``targets`` is a zero vector (the positive key
    is always column 0). ``return_keys=True`` additionally returns the
    normalised momentum keys so the trainer can enqueue them after a successful
    optimizer step.
    """

    def __init__(self, encoder=None, projector=None, *, queue_size=65536,
                 temperature=0.2, momentum=0.999, stochastic_depth_prob=0.0):
        super().__init__()
        if isinstance(queue_size, bool) or not isinstance(queue_size, int) or queue_size < 1:
            raise ValueError("queue_size must be a positive integer")
        if not 0 < temperature < 1:
            raise ValueError("temperature must be in (0, 1)")
        if not 0 < momentum < 1:
            raise ValueError("momentum must be in (0, 1)")
        self.encoder = encoder if encoder is not None else build_encoder(stochastic_depth_prob)
        self.projector = projector if projector is not None else build_projector()
        self.queue_size = queue_size
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_projector = copy.deepcopy(self.projector)
        for parameter in list(self.momentum_encoder.parameters()) + list(self.momentum_projector.parameters()):
            parameter.requires_grad_(False)
        self.momentum_encoder.eval()
        self.momentum_projector.eval()
        with torch.no_grad():
            dimension = self._projection_dim(self.projector)
        self.register_buffer("queue", torch.zeros(queue_size, dimension), persistent=False)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long), persistent=False)

    @staticmethod
    def _projection_dim(projector):
        for module in reversed(list(projector.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        raise ValueError("projector must end with a Linear layer")

    @property
    def queue_fill(self):
        """Number of valid keys currently in the queue (saturates at queue_size)."""
        return int(min(self.queue_size, self._enqueued.item()))

    # ``_enqueued`` tracks fill for the first pass; it is not a persistent model
    # state, only a monitoring counter, so it is plain Python state.
    @property
    def _enqueued(self):
        if not hasattr(self, "_enqueued_count"):
            self._enqueued_count = torch.zeros(1, dtype=torch.long)
        return self._enqueued_count

    def train(self, mode=True):
        super().train(mode)
        # Momentum branches never train; ConvNeXt has no batch norm running
        # statistics, but keeping them in eval avoids accidental dropout.
        self.momentum_encoder.eval()
        self.momentum_projector.eval()
        return self

    def forward(self, view_q, view_k, return_keys=False):
        query = F.normalize(self.projector(self.encoder(view_q)), dim=1)
        with torch.no_grad():
            key = F.normalize(self.momentum_projector(self.momentum_encoder(view_k)), dim=1)
        logits = self._logits(query, key)
        targets = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        if return_keys:
            return logits, targets, key.detach()
        return logits, targets

    def _logits(self, query, key):
        query = query.float()
        key = key.float()
        positive = torch.bmm(query.unsqueeze(1), key.unsqueeze(2))  # [B, 1, 1]
        negative = torch.mm(query, self.queue.t().to(query.device))  # [B, Q]
        return torch.cat((positive.squeeze(1), negative), dim=1) / self.temperature

    @torch.no_grad()
    def momentum_update(self, momentum=None):
        """EMA the online encoder/projector into the momentum branches."""
        decay = self.momentum if momentum is None else float(momentum)
        if not 0 < decay < 1:
            raise ValueError("momentum must be in (0, 1)")
        for online, target in ((self.encoder, self.momentum_encoder),
                               (self.projector, self.momentum_projector)):
            for online_parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.mul_(decay).add_(online_parameter.detach(), alpha=1 - decay)
            for online_buffer, target_buffer in zip(online.buffers(), target.buffers()):
                target_buffer.copy_(online_buffer)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        keys = keys.detach().float()
        if keys.ndim != 2 or keys.shape[1] != self.queue.shape[1]:
            raise ValueError(f"Expected [N, {self.queue.shape[1]}] keys, got {tuple(keys.shape)}")
        count = keys.shape[0]
        if count > self.queue_size:
            raise ValueError("A batch larger than the queue cannot be enqueued")
        pointer = int(self.queue_ptr.item())
        end = pointer + count
        if end <= self.queue_size:
            self.queue[pointer:end] = keys.to(self.queue.device)
        else:
            overflow = end - self.queue_size
            self.queue[pointer:] = keys[:self.queue_size - pointer].to(self.queue.device)
            self.queue[:overflow] = keys[self.queue_size - pointer:].to(self.queue.device)
        self.queue_ptr[0] = end % self.queue_size
        self._enqueued[0] += count

    @torch.no_grad()
    def encode(self, images):
        """Return frozen image features [N, 768] (no projection head)."""
        return self.encoder(images)

    def reset_queue(self):
        self.queue.zero_()
        self.queue_ptr.zero_()
        self._enqueued[0] = 0
