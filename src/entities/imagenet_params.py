from dataclasses import dataclass


@dataclass()
class ImagenetParams:
    model: str
    threshold: float
    batch_size: int
    device: str
