"""Exponential moving average of model weights; disabled by default.

The shadow copy is kept in float32 on the parameters' device and starts from
the current weights, so enabling EMA never perturbs initialization, optimizer
state or RNG consumption.
"""

import contextlib

import torch


class ModelEMA:
    def __init__(self, model, decay, warmup_steps=0):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError("EMA warmup_steps must be a nonnegative integer")
        self.decay = float(decay)
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self._backup = None
        self.shadow = {name: value.detach().to(dtype=torch.float32).clone()
                       for name, value in model.state_dict().items()
                       if value.is_floating_point()}

    def current_decay(self, step=None):
        """Constant decay unless warmup_steps > 0, then min(decay, (1+step)/(10+step))."""
        if not self.warmup_steps:
            return self.decay
        step = self.num_updates if step is None else step
        return min(self.decay, (1 + step) / (10 + step))

    @torch.no_grad()
    def update(self, model):
        decay = self.current_decay()
        state = model.state_dict()
        missing = [name for name in self.shadow if name not in state]
        if missing:
            raise ValueError(f"EMA shadow does not match model parameters: {missing}")
        for name, shadow in self.shadow.items():
            shadow.mul_(decay).add_(state[name].detach().to(dtype=shadow.dtype), alpha=1 - decay)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model):
        state = model.state_dict()
        missing = [name for name in self.shadow if name not in state]
        if missing:
            raise ValueError(f"EMA shadow does not match model parameters: {missing}")
        for name, shadow in self.shadow.items():
            state[name].copy_(shadow)

    @torch.no_grad()
    def store(self, model):
        if self._backup is not None:
            raise RuntimeError("EMA backup already stored")
        self._backup = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def restore(self, model):
        if self._backup is None:
            raise RuntimeError("No stored EMA backup to restore")
        model.load_state_dict(self._backup, strict=True)
        self._backup = None

    def state_dict(self):
        return {"shadow": {name: value.detach().cpu().clone() for name, value in self.shadow.items()},
                "num_updates": self.num_updates}

    @torch.no_grad()
    def load_state_dict(self, state):
        shadow = state["shadow"]
        if set(shadow) != set(self.shadow):
            raise ValueError("EMA state does not match model parameters")
        self.shadow = {name: shadow[name].detach().to(dtype=torch.float32,
                                                      device=self.shadow[name].device).clone()
                       for name in self.shadow}
        self.num_updates = int(state["num_updates"])


@contextlib.contextmanager
def apply_ema(ema, model):
    """Temporarily swap EMA weights into the model (no-op when ema is None)."""
    if ema is None:
        yield model
        return
    ema.store(model)
    try:
        ema.copy_to(model)
        yield model
    finally:
        ema.restore(model)
