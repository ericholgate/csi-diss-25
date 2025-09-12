"""
Experiment management system for "Did I Say This" character embedding learning.

Provides comprehensive experiment tracking, reproducibility, and resumable training
for systematic comparison of different configurations.
"""

import json
import os
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict

import torch

from .trainer import DidISayThisTrainer as Trainer, TrainingConfig
from .architecture import DidISayThisModel
from ..data.dataset import DidISayThisDataset

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manages complete experiment lifecycle with full reproducibility tracking.
    
    Features:
    - Dataset integrity verification via hashing
    - Complete configuration serialization
    - Automatic checkpoint management
    - Resumable training with state preservation
    - Comprehensive metadata tracking
    """
    
    def __init__(self, experiment_name: str, experiment_dir: str, config: TrainingConfig):
        self.experiment_name = experiment_name
        self.experiment_dir = Path(experiment_dir)
        self.config = config
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging for experiment
        self._setup_experiment_logging()
        
        # Metadata tracking
        self.metadata = {
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'config': asdict(config),
            'dataset_hash': None,
            'git_commit': self._get_git_commit(),
            'python_version': self._get_python_version(),
            'pytorch_version': torch.__version__,
            'status': 'created'
        }
        
        logger.info(f"Initialized experiment: {experiment_name}")
    
    def _setup_experiment_logging(self):
        """Setup dedicated logging for this experiment."""
        log_file = self.experiment_dir / f"{self.experiment_name}.log"
        
        # Create file handler for experiment
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Format for experiment logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.experiment_dir.parent)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _get_python_version(self) -> str:
        """Get Python version string."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _calculate_dataset_hash(self, dataset: DidISayThisDataset) -> str:
        """Calculate hash of dataset for integrity verification."""
        # Hash based on episodes and configuration
        hash_data = {
            'num_episodes': len(dataset.episodes),
            'character_mode': dataset.character_mode,
            'killer_reveal_holdout': dataset.killer_reveal_holdout,
            'holdout_percentage': dataset.holdout_percentage,
            'episode_ids': [ep.episode_id for ep in dataset.episodes]
        }
        
        # Add sentence counts per episode for verification
        for episode in dataset.episodes:
            hash_data[f"{episode.episode_id}_sentences"] = len(episode.sentences)
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def save_metadata(self):
        """Save experiment metadata to JSON file."""
        metadata_file = self.experiment_dir / "experiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Saved experiment metadata to {metadata_file}")
    
    def run_experiment(self, dataset: DidISayThisDataset, model: DidISayThisModel) -> Dict[str, Any]:
        """
        Run complete experiment with full tracking.
        
        Args:
            dataset: Prepared dataset
            model: Model to train
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Starting experiment: {self.experiment_name}")
        
        # Update metadata with dataset info
        self.metadata['dataset_hash'] = self._calculate_dataset_hash(dataset)
        self.metadata['dataset_info'] = {
            'num_episodes': len(dataset.episodes),
            'total_samples': len(dataset),
            'character_mode': dataset.character_mode,
            'killer_reveal_holdout': dataset.killer_reveal_holdout,
            'holdout_percentage': dataset.holdout_percentage
        }
        self.metadata['status'] = 'running'
        self.save_metadata()
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=self.config,
            experiment_dir=str(self.experiment_dir)
        )
        
        # Run training
        try:
            results = trainer.train(dataset)
            
            # Update metadata with results
            self.metadata['status'] = 'completed'
            self.metadata['completed_at'] = datetime.now().isoformat()
            self.metadata['final_results'] = results
            
            logger.info(f"Experiment completed successfully: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            self.metadata['failed_at'] = datetime.now().isoformat()
            raise
        finally:
            self.save_metadata()
        
        return results
    
    def resume_training(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint to resume from, or None for latest
            
        Returns:
            Training results
        """
        logger.info(f"Resuming experiment: {self.experiment_name}")
        
        # Find checkpoint to resume from
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if not checkpoint_path:
            raise ValueError("No checkpoint found to resume from")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Recreate model and dataset from checkpoint
        model_config = checkpoint['model_config']
        model = DidISayThisModel(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Recreate dataset (this should be deterministic)
        dataset = DidISayThisDataset.from_checkpoint_info(checkpoint['dataset_info'])
        
        # Verify dataset integrity
        current_hash = self._calculate_dataset_hash(dataset)
        if current_hash != checkpoint['dataset_hash']:
            logger.warning("Dataset hash mismatch - data may have changed")
        
        # Create trainer with resumed state
        trainer = Trainer(
            model=model,
            config=self.config,
            experiment_dir=str(self.experiment_dir)
        )
        
        # Resume training
        self.metadata['status'] = 'resumed'
        self.metadata['resumed_at'] = datetime.now().isoformat()
        self.metadata['resumed_from'] = checkpoint_path
        self.save_metadata()
        
        try:
            results = trainer.resume_training(checkpoint_path)
            
            self.metadata['status'] = 'completed'
            self.metadata['completed_at'] = datetime.now().isoformat()
            self.metadata['final_results'] = results
            
            logger.info(f"Resumed experiment completed: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"Resumed experiment failed: {e}")
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            self.metadata['failed_at'] = datetime.now().isoformat()
            raise
        finally:
            self.save_metadata()
        
        return results
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint in experiment directory."""
        checkpoint_pattern = self.experiment_dir / "checkpoint_epoch_*.pt"
        checkpoints = list(self.experiment_dir.glob("checkpoint_epoch_*.pt"))
        
        if not checkpoints:
            return None
        
        # Sort by modification time (most recent first)
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoints[0])
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get current experiment status and progress."""
        try:
            with open(self.experiment_dir / "experiment_metadata.json", 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            return {'status': 'not_found'}
        
        # Check for checkpoints
        checkpoints = list(self.experiment_dir.glob("checkpoint_epoch_*.pt"))
        
        status_info = {
            'status': metadata.get('status', 'unknown'),
            'created_at': metadata.get('created_at'),
            'num_checkpoints': len(checkpoints),
            'latest_checkpoint': str(max(checkpoints, key=lambda x: x.stat().st_mtime)) if checkpoints else None
        }
        
        # Add completion info if available
        if 'completed_at' in metadata:
            status_info['completed_at'] = metadata['completed_at']
        if 'final_results' in metadata:
            status_info['final_results'] = metadata['final_results']
        
        return status_info
    
    @classmethod
    def load_experiment(cls, experiment_dir: str) -> 'ExperimentManager':
        """
        Load existing experiment from directory.
        
        Args:
            experiment_dir: Directory containing experiment
            
        Returns:
            ExperimentManager instance
        """
        experiment_dir = Path(experiment_dir)
        
        # Load metadata
        metadata_file = experiment_dir / "experiment_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Recreate config
        config = TrainingConfig(**metadata['config'])
        
        # Create experiment manager
        experiment_name = metadata['experiment_name']
        manager = cls(experiment_name, str(experiment_dir), config)
        manager.metadata = metadata
        
        return manager
    
    @staticmethod
    def list_experiments(experiments_root: str) -> List[Dict[str, Any]]:
        """
        List all experiments in the experiments directory.
        
        Args:
            experiments_root: Root directory containing experiments
            
        Returns:
            List of experiment summaries
        """
        experiments_root = Path(experiments_root)
        experiments = []
        
        for exp_dir in experiments_root.iterdir():
            if exp_dir.is_dir() and (exp_dir / "experiment_metadata.json").exists():
                try:
                    manager = ExperimentManager.load_experiment(str(exp_dir))
                    status = manager.get_experiment_status()
                    experiments.append(status)
                except Exception as e:
                    logger.warning(f"Failed to load experiment {exp_dir}: {e}")
        
        return experiments