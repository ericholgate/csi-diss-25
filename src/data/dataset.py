"""
PyTorch Dataset and DataLoader implementation for the 'Did I Say This' character embedding task.

This module provides the Dataset class that generates temporally-aligned positive and negative 
training examples for character embedding learning, maintaining chronological order across 
episodes and within episodes for proper temporal learning progression.
"""

import random
import logging
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from .models import Episode, Character, Sentence
from .preprocessing import load_csi_data_complete

logger = logging.getLogger(__name__)


class DidISayThisDataset(Dataset):
    """
    PyTorch Dataset for the 'Did I Say This' character embedding task.
    
    Generates temporally-aligned positive/negative example pairs while maintaining
    chronological order for proper character embedding learning as episodes progress.
    
    Key Design:
    - Episodes processed in chronological order (s01e07, s01e08, etc.)
    - Sentences within episodes maintain original order
    - Each positive example immediately followed by corresponding negative example
    - Character embeddings learn temporal progression through the series
    """
    
    def __init__(self, 
                 episodes: List[Episode],
                 character_mode: str = 'episode-isolated',
                 negative_ratio: float = 1.0,
                 tokenizer_name: str = 'bert-base-uncased',
                 max_length: int = 512,
                 include_mentioned_characters: bool = False,
                 max_mentioned_characters: int = 3,
                 seed: Optional[int] = None,
                 # Track killer reveal holdout parameters for metadata
                 _holdout_killer_reveal: Optional[bool] = None,
                 _killer_reveal_holdout_percentage: Optional[float] = None,
                 _total_sentences_before_holdout: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            episodes: List of Episode objects (should be chronologically sorted)
            character_mode: Either 'episode-isolated' or 'cross-episode'
            negative_ratio: Ratio of negative to positive examples (default 1.0 = balanced)
            tokenizer_name: HuggingFace tokenizer to use
            max_length: Maximum sequence length for tokenization
            include_mentioned_characters: Whether to extract character embeddings for characters mentioned in sentences
            max_mentioned_characters: Maximum number of mentioned characters to include (padded/truncated)
            seed: Random seed for reproducibility (only affects negative sampling)
        """
        self.episodes = episodes
        self.character_mode = character_mode
        self.negative_ratio = negative_ratio
        self.max_length = max_length
        self.include_mentioned_characters = include_mentioned_characters
        self.max_mentioned_characters = max_mentioned_characters
        
        self._seed_used = seed
        
        # Store killer reveal holdout parameters for comprehensive metadata
        self._holdout_killer_reveal = _holdout_killer_reveal
        self._killer_reveal_holdout_percentage = _killer_reveal_holdout_percentage
        self._total_sentences_before_holdout = _total_sentences_before_holdout
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Build character vocabulary and examples
        self.character_to_id, self.id_to_character = self._build_character_vocabulary()
        
        # Build character name lookup for mentioned character detection
        if self.include_mentioned_characters:
            self.character_name_to_id = self._build_character_name_lookup()
        
        self.examples = self._generate_examples()
        
        logger.info(f"Dataset initialized with {len(self.examples)} temporally-aligned examples, "
                   f"{len(self.character_to_id)} unique characters")
    
    @classmethod
    def from_data_directory(cls, 
                           data_directory: Path,
                           character_mode: str = 'episode-isolated',
                           holdout_killer_reveal: bool = False,
                           killer_reveal_holdout_percentage: float = 0.1,
                           **kwargs) -> 'DidISayThisDataset':
        """
        Convenience constructor that loads data from TSV directory.
        
        Args:
            data_directory: Path to directory containing TSV files
            character_mode: Either 'episode-isolated' or 'cross-episode'
            **kwargs: Additional arguments for DidISayThisDataset
            
        Returns:
            DidISayThisDataset instance
        """
        data_components = load_csi_data_complete(data_directory, character_mode)
        episodes = data_components['episodes']
        
        # Track original sentence count before any filtering
        total_sentences_before_holdout = sum(len(episode.sentences) for episode in episodes)
        
        # Apply killer reveal holdout if requested
        if holdout_killer_reveal:
            episodes = cls._filter_killer_reveal_sentences(episodes, killer_reveal_holdout_percentage)
        
        return cls(
            episodes=episodes, 
            character_mode=character_mode, 
            _holdout_killer_reveal=holdout_killer_reveal,
            _killer_reveal_holdout_percentage=killer_reveal_holdout_percentage if holdout_killer_reveal else None,
            _total_sentences_before_holdout=total_sentences_before_holdout,
            **kwargs
        )
    
    @staticmethod
    def _filter_killer_reveal_sentences(episodes: List['Episode'], holdout_percentage: float) -> List['Episode']:
        """
        Filter out the last N% of sentences from each episode to prevent killer reveal contamination.
        
        Args:
            episodes: List of Episode objects to filter
            holdout_percentage: Percentage of sentences to hold out from the end (0.0-1.0)
            
        Returns:
            List of filtered Episode objects with reduced sentences
        """
        if holdout_percentage <= 0.0 or holdout_percentage >= 1.0:
            logger.warning(f"Invalid holdout_percentage {holdout_percentage}, must be between 0.0 and 1.0. Returning episodes unchanged.")
            return episodes
        
        filtered_episodes = []
        total_original_sentences = 0
        total_filtered_sentences = 0
        
        for episode in episodes:
            # Count sentences in this episode
            sentence_count = len(episode.sentences)
            total_original_sentences += sentence_count
            
            if sentence_count == 0:
                # Keep episodes with no sentences as-is
                filtered_episodes.append(episode)
                continue
            
            # Calculate how many sentences to keep (exclude the last holdout_percentage)
            sentences_to_keep = max(1, int(sentence_count * (1.0 - holdout_percentage)))
            total_filtered_sentences += sentences_to_keep
            
            # Create new episode with filtered sentences
            filtered_sentences = episode.sentences[:sentences_to_keep]
            
            # Create new episode object with same metadata but filtered sentences
            filtered_episode = Episode(
                episode_id=episode.episode_id,
                sentences=filtered_sentences,
                characters=episode.characters  # Keep same character list
            )
            
            filtered_episodes.append(filtered_episode)
        
        logger.info(f"Killer reveal holdout applied: {total_original_sentences} → {total_filtered_sentences} sentences "
                   f"({holdout_percentage:.1%} holdout, {total_original_sentences - total_filtered_sentences} sentences removed)")
        
        return filtered_episodes

    def _build_character_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build character vocabulary mapping character identifiers to integer IDs."""
        unique_characters = set()
        
        for episode in self.episodes:
            for character in episode.characters:
                char_id = character.get_unique_id(self.character_mode)
                unique_characters.add(char_id)
        
        # Sort for deterministic ordering
        sorted_characters = sorted(unique_characters)
        
        character_to_id = {char_id: idx for idx, char_id in enumerate(sorted_characters)}
        id_to_character = {idx: char_id for char_id, idx in character_to_id.items()}
        
        return character_to_id, id_to_character

    def _generate_examples(self) -> List[Dict[str, Any]]:
        """Generate temporally-aligned positive and negative examples for training."""
        examples = []
        global_position = 0
        pair_id = 0
        
        # Process episodes in chronological order (already sorted by load_csi_data)
        for episode in self.episodes:
            episode_examples = self._generate_episode_examples_temporally_aligned(
                episode, global_position, pair_id
            )
            examples.extend(episode_examples)
            
            # Update counters
            global_position += len(episode.sentences)
            pair_id += len(episode.sentences)  # Each sentence generates one pair
        
        logger.info(f"Generated {len(examples)} temporally-aligned examples "
                   f"({sum(1 for ex in examples if ex['label'] == 1)} positive, "
                   f"{sum(1 for ex in examples if ex['label'] == 0)} negative) "
                   f"across {len(self.episodes)} episodes")
        
        return examples

    def _generate_episode_examples_temporally_aligned(self, 
                                                     episode: Episode, 
                                                     global_start_position: int,
                                                     pair_start_id: int) -> List[Dict[str, Any]]:
        """Generate temporally-aligned examples for a single episode."""
        examples = []
        
        # Process sentences in their original order
        for local_position, sentence in enumerate(episode.sentences):
            global_position = global_start_position + local_position
            current_pair_id = pair_start_id + local_position
            
            # Generate positive example
            positive_example = self._create_positive_example(
                sentence, episode, global_position, current_pair_id
            )
            examples.append(positive_example)
            
            # Generate negative example(s) immediately after positive
            num_negatives = max(1, int(self.negative_ratio))  # At least 1 negative per positive
            
            for neg_idx in range(num_negatives):
                negative_example = self._create_negative_example(
                    sentence, episode, global_position, current_pair_id, neg_idx
                )
                if negative_example:  # Only add if successfully created
                    examples.append(negative_example)
        
        return examples

    def _create_positive_example(self, 
                                sentence: Sentence, 
                                episode: Episode,
                                global_position: int,
                                pair_id: int) -> Dict[str, Any]:
        """Create a positive example (character actually said the sentence)."""
        char_id = sentence.speaker.get_unique_id(self.character_mode)
        speaker_id = self.character_to_id[char_id]
        
        return {
            'sentence_text': sentence.text,
            'character_id': speaker_id,
            'label': 1,
            'episode_id': episode.episode_id,
            'temporal_position': global_position,
            'pair_id': pair_id,
            'example_type': 'positive',
            'metadata': {
                'sentence_key': sentence.get_sentence_key(),
                'character_name': sentence.speaker.normalized_name,
                'case_id': sentence.case_id,
                'sent_id': sentence.sent_id,
                'human_guess': sentence.human_guess,
                'gold_labels': sentence.gold_labels
            }
        }

    def _create_negative_example(self, 
                                sentence: Sentence, 
                                episode: Episode,
                                global_position: int,
                                pair_id: int,
                                neg_idx: int = 0) -> Optional[Dict[str, Any]]:
        """Create a negative example (different character, same sentence)."""
        # Sample a different character from the same episode (weighted by frequency)
        exclude_chars = {sentence.speaker}
        sampled_chars = episode.sample_characters_by_frequency(
            exclude=exclude_chars, 
            n_samples=1
        )
        
        if not sampled_chars:
            logger.warning(f"Could not sample negative character for {episode.episode_id}:{sentence.get_sentence_key()}")
            return None
        
        false_speaker = sampled_chars[0]
        char_id = false_speaker.get_unique_id(self.character_mode)
        false_speaker_id = self.character_to_id[char_id]
        
        return {
            'sentence_text': sentence.text,
            'character_id': false_speaker_id,
            'label': 0,
            'episode_id': episode.episode_id,
            'temporal_position': global_position,
            'pair_id': pair_id,
            'example_type': 'negative',
            'negative_idx': neg_idx,
            'metadata': {
                'sentence_key': sentence.get_sentence_key(),
                'actual_speaker': sentence.speaker.normalized_name,
                'false_speaker': false_speaker.normalized_name,
                'case_id': sentence.case_id,
                'sent_id': sentence.sent_id,
                'human_guess': sentence.human_guess,
                'gold_labels': sentence.gold_labels
            }
        }

    def get_character_vocabulary_size(self) -> int:
        """Get the size of the character vocabulary."""
        return len(self.character_to_id)

    def __len__(self) -> int:
        """Return the total number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example by index.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing tokenized inputs and labels
        """
        example = self.examples[idx]
        
        # Tokenize the sentence
        encoding = self.tokenizer(
            example['sentence_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'character_id': torch.tensor(example['character_id'], dtype=torch.long),
            'label': torch.tensor(example['label'], dtype=torch.float),
            'temporal_position': torch.tensor(example['temporal_position'], dtype=torch.long),
            'pair_id': torch.tensor(example['pair_id'], dtype=torch.long),
            'metadata': example['metadata']
        }
        
        return result

    def save_dataset(self, save_path: Path, include_examples: bool = True, 
                     compress: bool = True) -> None:
        """Save the complete dataset as a self-contained package."""
        import pickle
        import gzip
        
        save_path = Path(save_path)
        
        # Create comprehensive dataset package
        dataset_package = {
            # Core configuration - COMPLETELY COMPREHENSIVE
            'metadata': {
                # Dataset creation configuration
                'character_mode': self.character_mode,
                'negative_ratio': self.negative_ratio,
                'max_length': self.max_length,
                'include_mentioned_characters': self.include_mentioned_characters,
                'max_mentioned_characters': self.max_mentioned_characters,
                'seed_used': self._seed_used,
                'tokenizer_name': getattr(self.tokenizer, 'name_or_path', 'unknown'),
                
                # Killer reveal holdout configuration
                'holdout_killer_reveal': self._holdout_killer_reveal,
                'killer_reveal_holdout_percentage': self._killer_reveal_holdout_percentage,
                'total_sentences_before_holdout': self._total_sentences_before_holdout,
                'sentences_removed_by_holdout': (self._total_sentences_before_holdout - sum(len(ep.sentences) for ep in self.episodes)) if self._total_sentences_before_holdout else 0,
                
                # Dataset results
                'total_examples': len(self.examples),
                'total_episodes': len(self.episodes),
                'total_sentences_after_processing': sum(len(ep.sentences) for ep in self.episodes),
                'character_vocabulary_size': self.get_character_vocabulary_size(),
                'positive_examples': sum(1 for ex in self.examples if ex['label'] == 1),
                'negative_examples': sum(1 for ex in self.examples if ex['label'] == 0),
                
                # Creation context
                'creation_timestamp': datetime.now().isoformat(),
                'creation_working_directory': str(Path().cwd()),
                'python_version': sys.version,
                'dataset_creation_method': 'from_data_directory'
            },
            
            # Character vocabulary
            'character_vocabulary': {
                'character_to_id': self.character_to_id,
                'id_to_character': self.id_to_character
            }
        }
        
        # Optionally include full examples (can be very large)
        if include_examples:
            dataset_package['examples'] = self.examples
        else:
            # Include just a sample for verification
            dataset_package['example_sample'] = self.examples[:100] if self.examples else []
            dataset_package['examples_excluded'] = True
        
        # Save the package
        if compress:
            save_file = save_path.with_suffix('.pkl.gz')
            with gzip.open(save_file, 'wb') as f:
                pickle.dump(dataset_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            save_file = save_path.with_suffix('.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(dataset_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Complete dataset package saved to {save_file}")
        logger.info(f"Package size: {save_file.stat().st_size / (1024*1024):.1f} MB")
        logger.info(f"Includes examples: {include_examples}, Compressed: {compress}")

    @classmethod
    def load_dataset(cls, load_path: Path, verify_integrity: bool = True) -> 'DidISayThisDataset':
        """Load a complete dataset from a saved package."""
        import pickle
        import gzip
        
        load_path = Path(load_path)
        
        # Try to load compressed first, then uncompressed
        if load_path.with_suffix('.pkl.gz').exists():
            load_file = load_path.with_suffix('.pkl.gz')
            with gzip.open(load_file, 'rb') as f:
                dataset_package = pickle.load(f)
        elif load_path.with_suffix('.pkl').exists():
            load_file = load_path.with_suffix('.pkl')
            with open(load_file, 'rb') as f:
                dataset_package = pickle.load(f)
        else:
            raise FileNotFoundError(f"No dataset package found at {load_path}")
        
        logger.info(f"Loading dataset package from {load_file}")
        
        # Verify package structure
        required_keys = ['metadata', 'character_vocabulary']
        missing_keys = set(required_keys) - set(dataset_package.keys())
        if missing_keys:
            raise ValueError(f"Dataset package missing required keys: {missing_keys}")
        
        # Extract metadata
        metadata = dataset_package['metadata']
        
        # Create a minimal instance for loaded data
        dummy_instance = cls.__new__(cls)
        
        # Set basic attributes from metadata
        dummy_instance.character_mode = metadata['character_mode']
        dummy_instance.negative_ratio = metadata['negative_ratio']
        dummy_instance.max_length = metadata['max_length']
        dummy_instance.include_mentioned_characters = metadata['include_mentioned_characters']
        dummy_instance.max_mentioned_characters = metadata['max_mentioned_characters']
        dummy_instance._seed_used = metadata['seed_used']
        
        # Restore character vocabulary
        dummy_instance.character_to_id = dataset_package['character_vocabulary']['character_to_id']
        dummy_instance.id_to_character = dataset_package['character_vocabulary']['id_to_character']
        
        # Restore examples (if available)
        if 'examples' in dataset_package:
            dummy_instance.examples = dataset_package['examples']
            logger.info(f"Restored {len(dummy_instance.examples)} examples")
        elif 'example_sample' in dataset_package:
            dummy_instance.examples = dataset_package['example_sample']
            logger.warning(f"Only {len(dummy_instance.examples)} sample examples available (full examples were excluded)")
        else:
            dummy_instance.examples = []
            logger.warning("No examples available in loaded dataset")
        
        # Create minimal episode objects from episode_info if available
        dummy_instance.episodes = []
        
        # Initialize tokenizer (will need to be re-downloaded)
        try:
            from transformers import AutoTokenizer
            dummy_instance.tokenizer = AutoTokenizer.from_pretrained(metadata['tokenizer_name'])
        except Exception as e:
            logger.warning(f"Could not restore tokenizer {metadata['tokenizer_name']}: {e}")
            class DummyTokenizer:
                name_or_path = metadata['tokenizer_name']
            dummy_instance.tokenizer = DummyTokenizer()
        
        logger.info(f"Dataset package loaded successfully")
        logger.info(f"Configuration: {metadata['character_mode']} mode, seed {metadata['seed_used']}")
        logger.info(f"Vocabulary: {len(dummy_instance.character_to_id)} characters")
        
        return dummy_instance
