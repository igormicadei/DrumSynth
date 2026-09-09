"""Fitting a drum to recorded samples, per ARCHITECTURE.md §8.

Staged, and each stage freezes what the last one settled:

    1. f_static and t60 — measured, not searched
    2. excitation per velocity, independently
    3. inspect the table. THIS is the experiment
    4. a smooth curve through the per-velocity values
    5. joint refinement with the mapping in place

One drum at a time, deliberately: f_static and t60 belong to a specific
physical drum, so there is nothing a second drum could contribute except a way
to get them confused.
"""

from .backend import BatchLoss, Device, DeviceChoice, LossBackend
from .client import TrainingRun
from .objective import LinearVoiceBasis, SpectralTarget, TensionTrajectory
from .stages import (
    ExcitationFit,
    ExcitationStage,
    ExcitationTiltModel,
    Generation,
    Inspection,
    InspectionStage,
    JointStage,
    ModalFit,
    ModalStage,
    TensionFit,
    TensionStage,
    VelocityCurve,
    VelocityCurveStage,
)
from .targets import DrumCatalogue, FitTarget, Layer, TargetBuilder
from .trainer import DrumTrainer, FitEvaluator, FitResult, TrainingSettings
from .worker import FitEvent, FitWorker

__all__ = [
    "DrumCatalogue",
    "TargetBuilder",
    "FitTarget",
    "Layer",
    "ModalStage",
    "ModalFit",
    "ExcitationStage",
    "ExcitationFit",
    "ExcitationTiltModel",
    "TensionFit",
    "TensionStage",
    "InspectionStage",
    "Inspection",
    "VelocityCurveStage",
    "VelocityCurve",
    "JointStage",
    "Generation",
    "DrumTrainer",
    "TrainingSettings",
    "FitResult",
    "FitEvaluator",
    "LinearVoiceBasis",
    "SpectralTarget",
    "TensionTrajectory",
    "TrainingRun",
    "Device",
    "DeviceChoice",
    "BatchLoss",
    "LossBackend",
    "FitWorker",
    "FitEvent",
]
