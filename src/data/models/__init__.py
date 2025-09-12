"""
Data models for CSI character embedding analysis.

This module provides core data structures for handling CSI episode transcripts:
- Character: Represents a character with raw and normalized name forms
- Sentence: Represents a sentence with speaker, text, and metadata
- Episode: Represents an episode containing sentences and characters
"""

from .character import Character
from .sentence import Sentence
from .episode import Episode

__all__ = ['Character', 'Sentence', 'Episode']