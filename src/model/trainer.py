"""
Training configuration and trainer implementation for 'Did I Say This' character embedding learning.

Provides complete training loop with checkpointing, validation, and metrics tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time
import json
import csv
from tqdm import tqdm
from collections import defaultdict
import numpy as np

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
    
    # Prediction tracking and analysis
    record_predictions: bool = True  # Record all "Did I Say This" predictions
    killer_prediction_frequency: int = 200  # Evaluate killer prediction every N steps (~5 times per episode)
    save_embeddings_every_n_steps: int = 1000  # Save character embeddings for analysis
    
    # Killer prediction cross-validation
    killer_cv_folds: int = 5  # Number of CV folds for killer prediction
    killer_cv_seed: int = 42  # Seed for reproducible episode splits
    
    # Training paradigm selection
    sequential_cv_training: bool = True  # Use sequential CV training (theoretically superior)
    
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
    
    def __init__(self, model, train_dataset, val_dataset=None, config=None, experiment_manager=None):
        """Initialize trainer with model, datasets, and configuration."""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        self.experiment_manager = experiment_manager
        
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
        
        # Prediction tracking
        self.prediction_log = [] if self.config.record_predictions else None
        
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
        
        # Performance optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            logger.info("CUDA optimizations enabled")
        else:
            logger.info("Running on CPU - using CPU-optimized settings")
            # CPU-specific optimizations
            torch.set_num_threads(4)  # Limit threads to avoid overhead
        
        start_time = time.time()
        
        # Create data loaders
        # Optimize DataLoader for CPU training
        num_workers = 2 if torch.cuda.is_available() else 0  # CPU training works best with 0 workers
        batch_size = max(32, self.config.batch_size) if torch.cuda.is_available() else max(16, self.config.batch_size)
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Maintain temporal order
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # Only useful with GPU
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
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
            
            # Record individual predictions (sample every 10th batch for performance)
            if self.prediction_log is not None and batch_idx % 10 == 0:
                self._record_prediction(batch_idx, batch, logits, labels, loss)
            
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
            
            # Killer prediction evaluation
            if (self.experiment_manager and self.config.killer_prediction_frequency > 0 and 
                self.current_step % self.config.killer_prediction_frequency == 0):
                current_embeddings = self.model.character_embeddings.weight.detach()
                killer_results = self.experiment_manager.evaluate_killer_prediction(
                    current_embeddings, self.current_step, self.current_epoch
                )
                
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

    def _record_prediction(self, batch_idx: int, batch: Dict[str, torch.Tensor], 
                          logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> None:
        """Record individual predictions for Did I Say This task."""
        # Convert to probabilities
        probs = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
        predictions = (probs > 0.5).astype(int)
        labels_np = labels.detach().cpu().numpy()
        
        # Record each example in the batch
        for i in range(len(labels)):
            self.prediction_log.append({
                'step': self.current_step,
                'epoch': self.current_epoch,
                'batch_idx': batch_idx,
                'pair_id': batch['pair_id'][i].item() if 'pair_id' in batch else -1,
                'character_id': batch['character_id'][i].item(),
                'predicted_prob': float(probs[i]),
                'predicted_class': int(predictions[i]),
                'true_label': int(labels_np[i]),
                'loss': float(loss.item()),
                'temporal_position': batch['temporal_position'][i].item() if 'temporal_position' in batch else -1,
                'example_type': self._get_metadata_field(batch, i, 'example_type', 'unknown'),
            })

    def _get_metadata_field(self, batch, index: int, field_name: str, default_value):
        """Safely extract a field from batch metadata at given index."""
        try:
            metadata = batch.get('metadata')
            if metadata is None:
                return default_value
            
            # Handle different metadata structures that PyTorch collate might create
            if isinstance(metadata, list):
                # List of dictionaries (expected structure)
                if index < len(metadata) and isinstance(metadata[index], dict):
                    return metadata[index].get(field_name, default_value)
            elif isinstance(metadata, dict):
                # Dictionary with field names as keys and lists as values
                if field_name in metadata:
                    field_data = metadata[field_name]
                    if hasattr(field_data, '__getitem__') and index < len(field_data):
                        return field_data[index]
            
            return default_value
        except Exception:
            # Fallback to default if any error occurs
            return default_value

    def save_prediction_logs(self, experiment_dir: Path) -> None:
        """Save prediction logs to CSV files."""
        if not experiment_dir or not self.prediction_log:
            return
            
        import pandas as pd
        
        # Save "Did I Say This" predictions
        predictions_df = pd.DataFrame(self.prediction_log)
        predictions_path = experiment_dir / 'did_i_say_this_predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved {len(self.prediction_log)} predictions to {predictions_path}")
