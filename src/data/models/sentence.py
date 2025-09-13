"""
Sentence model for CSI character embedding analysis.

Represents a sentence with speaker, text content, and associated metadata
including gold labels and timing information.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from data.models.character import Character


@dataclass
class Sentence:
    """
    Represents a sentence spoken by a character in a CSI episode.
    
    Contains the aggregated text from word-level TSV data along with
    metadata for analysis and gold standard labels.
    """
    
    case_id: str
    sent_id: str
    speaker: Character
    text: str
    gold_labels: Dict[str, Any]
    human_guess: Optional[str] = None
    timing_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate sentence data after initialization."""
        if not self.text.strip():
            raise ValueError("Sentence text cannot be empty")
        if not isinstance(self.speaker, Character):
            raise ValueError("Speaker must be a Character instance")
    
    def get_sentence_key(self) -> str:
        """
        Get unique key for this sentence within an episode.
        
        Returns:
            String key in format "caseID:sentID"
        """
        return f"{self.case_id}:{self.sent_id}"
    
    def has_gold_label(self, label_type: str) -> bool:
        """
        Check if sentence has a specific type of gold label.
        
        Args:
            label_type: Type of label to check ('killer_gold', 'suspect_gold', etc.)
        
        Returns:
            True if sentence has the specified gold label type
        """
        return label_type in self.gold_labels and self.gold_labels[label_type] is not None
    
    def get_word_count(self) -> int:
        """Get word count of sentence text."""
        return len(self.text.split())
    
    def __len__(self) -> int:
        """Return character count of sentence text."""
        return len(self.text)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Sentence({self.get_sentence_key()}, speaker='{self.speaker.normalized_name}', words={self.get_word_count()})"