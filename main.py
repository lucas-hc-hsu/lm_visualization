"""
Transformer Architecture Comparison Animation
Encoder-Decoder vs Decoder-Only

Main entry point for rendering all scenes.

Usage:
    # Render individual scenes
    manim -pql main.py TransformerComparison  # Quick test

    # Render specific phase scenes
    manim -pql scenes/architecture.py EncoderDecoderArchitecture
    manim -pql scenes/architecture.py DecoderOnlyArchitecture
    manim -pql scenes/training.py EncoderDecoderTraining
    manim -pql scenes/training.py DecoderOnlyTraining
    manim -pql scenes/training.py TrainingComparison
    manim -pql scenes/inference.py EncoderDecoderInference
    manim -pql scenes/inference.py DecoderOnlyInference
    manim -pql scenes/inference.py InferenceEfficiencyComparison
    manim -pql scenes/summary.py SummaryComparisonTable
    manim -pql scenes/summary.py FinalIntegration

    # Render high quality
    manim -pqh main.py TransformerComparison
"""

from manim import *
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all scene classes
from scenes.architecture import (
    EncoderDecoderArchitecture,
    DecoderOnlyArchitecture,
    ArchitectureComparison,
)
from scenes.training import (
    EncoderDecoderTraining,
    DecoderOnlyTraining,
    TrainingComparison,
)
from scenes.inference import (
    EncoderDecoderInference,
    DecoderOnlyInference,
    InferenceEfficiencyComparison,
)
from scenes.summary import (
    SummaryComparisonTable,
    FinalIntegration,
    TransformerComparison,
)
from components.transformer_block import TransformerBlock
from components.attention_matrix import AttentionMatrix


# Re-export for manim CLI
__all__ = [
    # Phase 2: Architecture
    "EncoderDecoderArchitecture",
    "DecoderOnlyArchitecture",
    "ArchitectureComparison",
    # Phase 3: Training
    "EncoderDecoderTraining",
    "DecoderOnlyTraining",
    "TrainingComparison",
    # Phase 4: Inference
    "EncoderDecoderInference",
    "DecoderOnlyInference",
    "InferenceEfficiencyComparison",
    # Phase 5: Summary
    "SummaryComparisonTable",
    "FinalIntegration",
    "TransformerComparison",
]
