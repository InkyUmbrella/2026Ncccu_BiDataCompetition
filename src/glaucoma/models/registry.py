"""Model factories must return modules implementing forward(image, age) -> [N]."""

_FACTORIES = {}


def register_model(name):
    def register(factory):
        if name in _FACTORIES:
            raise ValueError(f"Model already registered: {name}")
        _FACTORIES[name] = factory
        return factory
    return register


def available_models():
    return sorted(_FACTORIES)


def build_model(config, *, load_pretrained=True):
    if config.name not in _FACTORIES:
        raise ValueError(f"Unknown model {config.name!r}; available: {available_models()}")
    params = dict(config.params)
    if config.weights is not None and load_pretrained:
        params["weights"] = config.weights
    return _FACTORIES[config.name](**params)
