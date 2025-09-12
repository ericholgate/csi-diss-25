"""
Episode model for CSI character embedding analysis.

Represents an episode containing sentences and characters, with methods
for character sampling and episode-level analysis.
"""

import random
from dataclasses import dataclass
from typing import List, Set, Dict, Any
from collections import Counter
from .character import Character
from .sentence import Sentence


@dataclass
class Episode:
    """
    Represents a CSI episode containing sentences and characters.
    
    Provides functionality for character frequency analysis and sampling
    for negative example generation in the "Did I Say This" task.
    """
    
    episode_id: str
    sentences: List[Sentence] = None
    characters: List[Character] = None
    
    def __post_init__(self):
        """Initialize empty lists if not provided."""
        if self.sentences is None:
            self.sentences = []
        if self.characters is None:
            self.characters = []
    
    def get_total_sentences(self) -> int:
        """Get total number of sentences in this episode."""
        return len(self.sentences)
    
    def get_total_characters(self) -> int:
        """Get total number of unique characters in this episode."""
        return len(self.characters)
    
    def get_character_frequency(self) -> Dict[Character, int]:
        """
        Get frequency count of how often each character speaks.
        
        Returns:
            Dictionary mapping Character to speak count
        """
        frequency = Counter()
        for sentence in self.sentences:
            frequency[sentence.speaker] += 1
        return dict(frequency)
    
    def sample_characters_by_frequency(self, exclude: Set[Character] = None, 
                                     n_samples: int = 1) -> List[Character]:
        """
        Sample characters weighted by their speaking frequency.
        
        Characters who speak more often are more likely to be sampled.
        
        Args:
            exclude: Set of characters to exclude from sampling
            n_samples: Number of characters to sample
        
        Returns:
            List of sampled Character objects
        """
        if exclude is None:
            exclude = set()
        
        # Get frequency counts
        frequency = self.get_character_frequency()
        
        # Filter out excluded characters
        available_chars = [char for char in frequency.keys() if char not in exclude]
        
        if not available_chars:
            return []
        
        if len(available_chars) <= n_samples:
            return available_chars
        
        # Create weighted sampling based on frequency
        weights = [frequency[char] for char in available_chars]
        
        # Sample without replacement
        sampled = random.choices(
            population=available_chars,
            weights=weights,
            k=min(n_samples, len(available_chars))
        )
        
        return sampled
    
    def get_sentences_by_speaker(self, speaker: Character) -> List[Sentence]:
        """
        Get all sentences spoken by a specific character.
        
        Args:
            speaker: Character to find sentences for
        
        Returns:
            List of sentences spoken by the character
        """
        return [s for s in self.sentences if s.speaker == speaker]
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about this episode.
        
        Returns:
            Dictionary with episode statistics
        """
        frequency = self.get_character_frequency()
        
        stats = {
            'episode_id': self.episode_id,
            'total_sentences': len(self.sentences),
            'total_characters': len(self.characters),
            'avg_sentences_per_character': len(self.sentences) / len(self.characters) if self.characters else 0,
            'most_active_character': max(frequency, key=frequency.get) if frequency else None,
            'character_frequency_distribution': {
                char.normalized_name: count for char, count in frequency.items()
            }
        }
        
        return stats
    
    def __len__(self) -> int:
        """Return number of sentences in episode."""
        return len(self.sentences)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Episode({self.episode_id}, {len(self.sentences)} sentences, {len(self.characters)} characters)"