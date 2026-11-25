"""Target plugin & hardware backend interfaces."""

from .base import ParamInfo, TargetBackend
from .macos_au import MacOSAudioUnitBackend
from .synthetic import SyntheticAmpBackend, SyntheticOverdriveBackend
from .synthetic_fx import (
    SyntheticBbdDelayBackend,
    SyntheticCompressorBackend,
    SyntheticEnvelopeFilterBackend,
    SyntheticHarmonicTremoloBackend,
    SyntheticSpringReverbBackend,
)
from .wav_pair import WavPairBackend

__all__ = [
    "ParamInfo",
    "TargetBackend",
    "MacOSAudioUnitBackend",
    "WavPairBackend",
    "SyntheticAmpBackend",
    "SyntheticOverdriveBackend",
    "SyntheticSpringReverbBackend",
    "SyntheticHarmonicTremoloBackend",
    "SyntheticBbdDelayBackend",
    "SyntheticEnvelopeFilterBackend",
    "SyntheticCompressorBackend",
]
