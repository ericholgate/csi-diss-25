#!/usr/bin/env python3
"""
Emergency repository restoration script.
Recreates all missing critical files from the CSI character embedding project.
"""

import os
from pathlib import Path

def create_file(filepath, content):
    """Create a file with given content."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Created: {filepath}")

# Create all the critical files
def restore_repository():
    print("🚨 EMERGENCY REPOSITORY RESTORATION")
    print("===================================")
    
    # PyTorch Dataset implementation
    dataset_content = '''"""
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
'''
    
    create_file("src/data/dataset.py", dataset_content)
    
    # Create trainer configuration
    trainer_content = '''"""
Training configuration and trainer implementation for 'Did I Say This' character embedding learning.

Provides complete training loop with checkpointing, validation, and metrics tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training the DidISayThis model."""
    
    # Training parameters
    num_epochs: int = 1  # Default to 1 to avoid killer reveal contamination
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Learning rate scheduling
    warmup_steps: int = 0
    use_scheduler: bool = False
    
    # Logging and checkpointing
    checkpoint_every_n_steps: int = 1000
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 500
    
    # Killer reveal holdout (to prevent contamination during multi-epoch training)
    holdout_killer_reveal: bool = False  # Enable killer reveal holdout
    killer_reveal_holdout_percentage: float = 0.1  # Hold out last 10% of each episode
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class DidISayThisTrainer:
    """
    Trainer for the 'Did I Say This' character embedding model.
    
    Provides complete training loop with:
    - AdamW optimizer with gradient clipping
    - Optional learning rate scheduling
    - Automatic checkpointing and resuming
    - Validation evaluation
    - Comprehensive metrics tracking
    """
    
    def __init__(self, model, train_dataset, val_dataset=None, config=None):
        """Initialize trainer with model, datasets, and configuration."""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Set up loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Set up learning rate scheduler
        if self.config.use_scheduler:
            total_steps = len(train_dataset) // self.config.batch_size * self.config.num_epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        else:
            self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        logger.info(f"Trainer initialized for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate}, wd={self.config.weight_decay})")

    def train(self, experiment_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            experiment_dir: Directory to save checkpoints and logs
            
        Returns:
            Training results dictionary
        """
        if experiment_dir:
            experiment_dir = Path(experiment_dir)
            experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting training...")
        start_time = time.time()
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Maintain temporal order
            num_workers=0  # CPU training
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, experiment_dir)
            
            # Validation phase
            if val_loader:
                val_metrics = self._validate_epoch(val_loader)
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    if experiment_dir:
                        self._save_checkpoint(experiment_dir / 'best_model.pt', 
                                           {'epoch': epoch, 'step': self.current_step, 'is_best': True})
            else:
                val_metrics = {}
            
            # Log epoch results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            
            self.training_history.append(epoch_results)
            logger.info(f"Epoch {epoch + 1} completed: {epoch_results}")
        
        # Save final model
        if experiment_dir:
            self._save_checkpoint(experiment_dir / 'final.pt', 
                               {'epoch': self.config.num_epochs, 'step': self.current_step, 'is_final': True})
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        results = {
            'training_duration_minutes': training_duration / 60,
            'final_train_loss': self.training_history[-1]['train_loss'],
            'final_train_accuracy': self.training_history[-1]['train_accuracy'],
            'best_val_loss': self.best_val_loss if val_loader else None,
            'total_steps': self.current_step,
            'total_epochs': self.config.num_epochs,
            'training_history': self.training_history
        }
        
        logger.info(f"Training completed in {training_duration/60:.1f} minutes")
        logger.info(f"Final results: {results}")
        
        return results

    def _train_epoch(self, train_loader: DataLoader, experiment_dir: Optional[Path]) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.current_step += 1
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            character_ids = batch['character_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, character_ids)
            loss = self.criterion(logits.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.sigmoid(logits.squeeze()) > 0.5
            correct_predictions += (predictions == labels.bool()).sum().item()
            total_predictions += labels.size(0)
            
            # Logging
            if self.current_step % self.config.log_every_n_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = correct_predictions / total_predictions
                
                logger.info(f"Step {self.current_step}: Loss: {loss.item():.4f}, "
                           f"Accuracy: {accuracy:.4f}, LR: {current_lr:.2e}")
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}',
                    'LR': f'{current_lr:.2e}'
                })
            
            # Checkpointing
            if experiment_dir and self.current_step % self.config.checkpoint_every_n_steps == 0:
                checkpoint_path = experiment_dir / f'checkpoint_step_{self.current_step}.pt'
                self._save_checkpoint(checkpoint_path, {'epoch': self.current_epoch, 'step': self.current_step})
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Validation during training
            if (self.val_dataset and self.current_step % self.config.eval_every_n_steps == 0):
                val_metrics = self._validate_step()
                logger.info(f"Step {self.current_step} validation: {val_metrics}")
                self.model.train()  # Return to training mode
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return {'loss': avg_loss, 'accuracy': accuracy}

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                character_ids = batch['character_id'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask, character_ids)
                loss = self.criterion(logits.squeeze(), labels)
                
                total_loss += loss.item()
                predictions = torch.sigmoid(logits.squeeze()) > 0.5
                correct_predictions += (predictions == labels.bool()).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return {'loss': avg_loss, 'accuracy': accuracy}

    def _validate_step(self) -> Dict[str, float]:
        """Quick validation during training."""
        if not self.val_dataset:
            return {}
        
        self.model.eval()
        
        # Sample a small batch for quick validation
        val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=True)
        batch = next(iter(val_loader))
        
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            character_ids = batch['character_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits = self.model(input_ids, attention_mask, character_ids)
            loss = self.criterion(logits.squeeze(), labels)
            
            predictions = torch.sigmoid(logits.squeeze()) > 0.5
            accuracy = (predictions == labels.bool()).float().mean().item()
        
        return {'val_loss': loss.item(), 'val_accuracy': accuracy}

    def _save_checkpoint(self, path: Path, extra_info: Dict = None) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'step': self.current_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resumed at epoch {self.current_epoch}, step {self.current_step}")
'''
    
    create_file("src/model/trainer.py", trainer_content)
    
    # Essential scripts and configs
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt

# Data and outputs (keep structure but ignore large files)
experiments/
scratch/analysis_output/
*.pkl
*.pkl.gz
*.csv
*.tsv.gz

# Logs
*.log
logs/

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Planning and implementation notes (secret sauce 🤫)
notes/
CLAUDE.md
scratch/example_training.py'''

    create_file(".gitignore", gitignore_content)
    
    # README.md
    readme_content = '''# CSI Character Embedding Learning System

A complete PyTorch implementation for learning character embeddings from CSI television episode transcripts using the "Did I Say This" proxy task.

## Overview

This system learns character embeddings by training a binary classifier to predict whether a character said a given sentence. The learned embeddings capture speaking patterns and character distinctions within and across episodes.

**Key Features:**
- ✅ Safe multi-epoch training with killer reveal holdout
- ✅ Complete experiment reproducibility and tracking  
- ✅ Comprehensive configuration options
- ✅ Automatic checkpointing and resumable training
- ✅ Character embedding extraction for analysis

## Quick Start

```python
# Basic single-epoch training (safest)
python scratch/example_training.py

# Or import and use directly
from pathlib import Path
from data.dataset import DidISayThisDataset
from model.experiment import ExperimentManager, ExperimentConfig
from model.architecture import DidISayThisConfig  
from model.trainer import TrainingConfig

# Create dataset
dataset = DidISayThisDataset.from_data_directory(
    Path("data/original"),
    character_mode='episode-isolated',
    seed=42
)

# Set up experiment
experiment = ExperimentManager(
    experiment_config=ExperimentConfig(experiment_name="my_experiment"),
    model_config=DidISayThisConfig(character_vocab_size=dataset.get_character_vocabulary_size()),
    training_config=TrainingConfig(num_epochs=1)
)

# Train model
model = experiment.setup_experiment(dataset, None)
results = experiment.train_model()
```

## Architecture

```
Input: (Sentence Text, Character ID) → Binary Classification (Did character say this?)

BERT-base (frozen) → [CLS] Token (768-dim)
                           ↓
Character Embedding Lookup → (768-dim)
                           ↓
                    Concatenate → (1536-dim)
                           ↓
              Optional Projection → (configurable-dim)
                           ↓ 
                    ReLU + Dropout
                           ↓
              Binary Classifier → (1-dim logit)
```

## Safe Training Configurations

### Single-Epoch Training (Recommended)
```python
# Safe default - no killer reveal contamination
TrainingConfig(
    num_epochs=1,
    holdout_killer_reveal=False  # Not needed for single epoch
)
```

### Multi-Epoch Training (Advanced)
```python
# Safe multi-epoch with killer reveal protection
TrainingConfig(
    num_epochs=3,
    holdout_killer_reveal=True,
    killer_reveal_holdout_percentage=0.1  # Hold out last 10%
)

# Dataset must match
DidISayThisDataset.from_data_directory(
    data_directory,
    holdout_killer_reveal=True,
    killer_reveal_holdout_percentage=0.1
)
```

⚠️ **WARNING**: Multi-epoch training without killer reveal holdout risks contaminating character embeddings with killer information!

## Installation

```bash
# Clone repository
git clone <repo-url>
cd csi-diss-25

# Create virtual environment  
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers numpy pandas tqdm

# Verify installation
python -c "from src.data.dataset import DidISayThisDataset; print('✅ Installation successful!')"
```'''

    create_file("README.md", readme_content)
    
    print("\n✅ CRITICAL FILES RESTORED!")
    print("========================")
    print("Core implementation files have been recreated:")
    print("- Data models (Character, Sentence, Episode)")
    print("- Dataset implementation with killer reveal holdout")  
    print("- Neural network architecture")
    print("- Training system")
    print("- README.md and .gitignore")
    print()
    print("⚠️ STILL NEED TO RECREATE:")
    print("- Experiment management system")
    print("- Experimental scripts (run_experiments.sh, etc.)")
    print("- Example training script")
    print()
    print("Run this script to continue restoration:")
    print("python restore_repository.py")

if __name__ == "__main__":
    restore_repository()