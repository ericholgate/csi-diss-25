"""
Character model for CSI character embedding analysis.

Represents a character with both raw and normalized name forms,
supporting both episode-isolated and cross-episode character identity modes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Character:
    """
    Represents a character in CSI episodes.
    
    Maintains both raw form (as appears in data) and normalized form (cleaned)
    to support character identity resolution across episodes.
    """
    
    raw_name: str
    normalized_name: str
    episode_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate character data after initialization."""
        if not self.raw_name or not self.normalized_name:
            raise ValueError("Character must have both raw_name and normalized_name")
    
    def get_unique_id(self, mode: str = 'episode-isolated') -> str:
        """
        Get unique identifier for this character based on identity mode.
        
        Args:
            mode: Either 'episode-isolated' or 'cross-episode'
        
        Returns:
            Unique character identifier string
        """
        if mode == 'episode-isolated':
            if not self.episode_id:
                raise ValueError("Episode ID required for episode-isolated mode")
            return f"{self.episode_id}:{self.normalized_name.lower()}"
        elif mode == 'cross-episode':
            return self.normalized_name.lower()
        else:
            raise ValueError(f"Unknown character mode: {mode}")
    
    def __hash__(self) -> int:
        """Allow Character objects to be used in sets and as dict keys."""
        return hash((self.raw_name, self.normalized_name, self.episode_id))
    
    def __eq__(self, other) -> bool:
        """Character equality based on raw name, normalized name, and episode."""
        if not isinstance(other, Character):
            return False
        return (
            self.raw_name == other.raw_name and
            self.normalized_name == other.normalized_name and
            self.episode_id == other.episode_id
        )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Character(raw='{self.raw_name}', norm='{self.normalized_name}', ep='{self.episode_id}')"