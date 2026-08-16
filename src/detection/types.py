from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    @property
    def x1(self):
        return self.bbox[0]

    @property
    def y1(self):
        return self.bbox[1]

    @property
    def x2(self):
        return self.bbox[2]

    @property
    def y2(self):
        return self.bbox[3]


@dataclass
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]


@dataclass
class ComplianceResult:
    track_id: int
    helmet: bool
    mask: bool
    vest: bool
    compliant: bool