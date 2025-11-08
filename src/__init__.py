"""
IAM ML Framework - Machine Learning for Identity and Access Management

A configuration-driven framework for analyzing IAM data using ML.
"""

__version__ = '1.0.0'
__author__ = 'IAM ML Framework'

from .database import DatabaseConnector
from .preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .insights import InsightsGenerator
from .visualization import Visualizer

__all__ = [
    'DatabaseConnector',
    'DataPreprocessor',
    'ModelTrainer',
    'InsightsGenerator',
    'Visualizer'
]

