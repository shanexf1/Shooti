"""Shooti — composition and camera-angle coaching for photographs."""

from .rules import Analysis, Finding, Horizon, analyze
from .subject import Face, Subject, detect_subject

__all__ = [
    "Analysis",
    "Face",
    "Finding",
    "Horizon",
    "Subject",
    "analyze",
    "detect_subject",
]
