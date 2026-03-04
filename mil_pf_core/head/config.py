from dataclasses import dataclass


@dataclass
class HeadConfig:
    suspicious_threshold: float = 0.5

    def __post_init__(self):
        if self.suspicious_threshold is None:
            raise ValueError("suspicious_threshold must not be None.")
        if not 0.0 <= self.suspicious_threshold <= 1.0:
            raise ValueError("suspicious_threshold must satisfy 0 <= value <= 1.")
