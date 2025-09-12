"""
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
