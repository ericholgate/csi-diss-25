"""
Character embedding learning models package.

Contains the neural network architecture, training system, and experiment
management for the "Did I Say This" character embedding learning task.
"""

from .architecture import DidISayThisConfig, DidISayThisModel
from .trainer import TrainingConfig, DidISayThisTrainer as Trainer
from .experiment import ExperimentManager

__all__ = [
    'DidISayThisConfig',
    'DidISayThisModel',
    'TrainingConfig',
    'Trainer',
    'ExperimentManager'
]