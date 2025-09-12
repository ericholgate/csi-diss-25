"""
CSI data processing package.

Contains modules for loading, preprocessing, and creating PyTorch datasets
from CSI episode TSV files for character embedding learning.
"""

from .models import Character, Sentence, Episode
from .preprocessing import load_csi_data_complete, normalize_character_name
from .dataset import DidISayThisDataset

__all__ = [
    'Character',
    'Sentence', 
    'Episode',
    'load_csi_data_complete',
    'normalize_character_name',
    'DidISayThisDataset'
]