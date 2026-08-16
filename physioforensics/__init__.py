"""physio-forensics: generator-agnostic deepfake detection from physiology."""

__version__ = "0.1.0"

from . import features, regions, rppg, synth  # noqa: F401

__all__ = ["features", "regions", "rppg", "synth"]
