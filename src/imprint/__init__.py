"""Imprint: a personality calibration layer for any chat model.

Learns how one user wants an assistant to talk (8 trait dials plus standing
rules) from their own messages, and renders a short plain-text block for the
host to inject into the system prompt.
"""
from .core import Imprint, LearnResult, new_profile
from .embedders import Embedder, MiniLMEmbedder
from .steering import Steer, detect_steering
from .traits import TRAIT_NAMES, band_step, describe

__all__ = ["Imprint", "LearnResult", "new_profile", "Embedder", "MiniLMEmbedder",
           "Steer", "detect_steering", "TRAIT_NAMES", "band_step", "describe"]
__version__ = "0.3.0"
