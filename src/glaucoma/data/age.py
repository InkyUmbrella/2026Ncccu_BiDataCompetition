from dataclasses import asdict, dataclass
import math

import numpy as np


def parse_age(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def valid_age(value):
    return math.isfinite(value) and 0 < value <= 100


@dataclass(frozen=True)
class AgeStats:
    median: float
    mean: float
    std: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.median, self.mean, self.std)) or self.std <= 0:
            raise ValueError("Invalid age preprocessing statistics")

    @classmethod
    def fit(cls, ages):
        values = np.asarray([a for a in ages if valid_age(a)], dtype=np.float64)
        if not len(values):
            raise ValueError("Training CSV has no valid ages in (0, 100]")
        return cls(float(np.median(values)), float(values.mean()), float(values.std()) or 1.0)

    def transform(self, age):
        missing = not math.isfinite(age)
        invalid = not missing and not valid_age(age)
        value = self.median if missing or invalid else age
        return [(value - self.mean) / self.std, float(missing), float(invalid)]

    def to_dict(self):
        return asdict(self)
