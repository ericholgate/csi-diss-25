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

from model.trainer import DidISayThisTrainer as Trainer, TrainingConfig
from model.architecture import DidISayThisModel
from data.dataset import DidISayThisDataset

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
            'holdout_killer_reveal': dataset._holdout_killer_reveal,
            'killer_reveal_holdout_percentage': dataset._killer_reveal_holdout_percentage,
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
            'holdout_killer_reveal': dataset._holdout_killer_reveal,
            'killer_reveal_holdout_percentage': dataset._killer_reveal_holdout_percentage
        }
        self.metadata['status'] = 'running'
        self.save_metadata()
        
        # Choose training paradigm based on configuration
        if hasattr(self.config, 'sequential_cv_training') and self.config.sequential_cv_training:
            return self._run_sequential_cv_experiment(dataset, model)
        else:
            return self._run_standard_experiment(dataset, model)
    
    def _run_standard_experiment(self, dataset: DidISayThisDataset, model: DidISayThisModel) -> Dict[str, Any]:
        """Run standard (parallel CV) experiment."""
        # Initialize killer prediction setup if enabled
        if self.config.killer_prediction_frequency > 0:
            initial_embeddings = model.character_embeddings.weight.detach()
            self.initialize_killer_prediction_setup(dataset, initial_embeddings)
        
        # Create trainer with experiment manager
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            val_dataset=None,  # No validation - killer prediction CV is our real evaluation
            config=self.config,
            experiment_manager=self
        )
        
        # Run training
        try:
            results = trainer.train(self.experiment_dir)
            
            # Save prediction logs
            trainer.save_prediction_logs(self.experiment_dir)
            self.save_prediction_logs()
            
            # Update metadata with results
            self.metadata['status'] = 'completed'
            self.metadata['completed_at'] = datetime.now().isoformat()
            self.metadata['final_results'] = results
            
            logger.info(f"Standard experiment completed successfully: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.metadata['status'] = 'failed'
            self.metadata['error'] = str(e)
            self.metadata['failed_at'] = datetime.now().isoformat()
            raise
        finally:
            self.save_metadata()
        
        return results
    
    def _run_sequential_cv_experiment(self, dataset: DidISayThisDataset, model: DidISayThisModel) -> Dict[str, Any]:
        """
        Run sequential CV training experiment.
        
        For each fold:
        1. Train character embeddings on training episodes (4/5 of data)
        2. Train killer classifier on learned training embeddings
        3. Train character embeddings on test episodes (1/5 of data)  
        4. Apply killer classifier to test embeddings
        5. Record accuracy metrics
        """
        logger.info(f"Starting sequential CV training with {self.config.killer_cv_folds} folds")
        
        # Get CV splits from dataset
        cv_data = dataset.get_killer_cv_splits(
            n_folds=self.config.killer_cv_folds,
            seed=self.config.killer_cv_seed
        )
        
        cv_splits = cv_data['splits']
        episode_killer_labels = cv_data['episode_killer_labels']
        episode_characters = cv_data['episode_characters']
        character_to_id = cv_data['character_to_id']
        
        # Check for existing progress to resume from
        progress_file = self.experiment_dir / 'sequential_cv_progress.json'
        start_fold = 0
        fold_results = []
        sequential_results = []
        
        if progress_file.exists():
            logger.info("Found existing sequential CV progress, checking for resume...")
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            print(f"📊 Progress data status: {progress_data.get('status', 'unknown')}")
            print(f"📊 Progress data completed_folds: {progress_data.get('completed_folds', 0)}")
            print(f"📊 Total fold results: {len(progress_data.get('fold_results', []))}")
            
            if progress_data.get('status') == 'completed':
                logger.info("Sequential CV already completed!")
                return progress_data.get('final_results', {})
            
            # Resume from where we left off
            completed_folds = progress_data.get('completed_folds', 0)
            total_folds = len(cv_splits)
            
            # If all folds are completed but status isn't 'completed', complete the experiment
            if completed_folds >= total_folds:
                print("🎯 All folds appear completed, attempting to finalize results...")
                sequential_results = progress_data.get('fold_results', [])
                fold_results = [r for r in sequential_results if 'error' not in r]
                
                if fold_results:
                    print(f"✅ Found {len(fold_results)} successful folds, finalizing...")
                    # Skip the fold loop and go directly to result aggregation
                    start_fold = total_folds  # This will skip all folds
                else:
                    print("❌ No successful fold results found, will restart all folds")
                    start_fold = 0
                    fold_results = []
                    sequential_results = []
            elif completed_folds > 0:
                logger.info(f"Resuming sequential CV from fold {completed_folds}")
                start_fold = completed_folds
                sequential_results = progress_data.get('fold_results', [])
                fold_results = [r for r in sequential_results if 'error' not in r]
        
        for fold_idx, (train_episodes, test_episodes) in enumerate(cv_splits):
            # Skip already completed folds
            if fold_idx < start_fold:
                logger.info(f"Skipping already completed fold {fold_idx}")
                continue
                
            print(f"🔄 Starting fold {fold_idx + 1}/{len(cv_splits)}")
            print(f"   Train episodes: {len(train_episodes)}, Test episodes: {len(test_episodes)}")
            logger.info(f"Starting fold {fold_idx + 1}/{len(cv_splits)}")
            logger.info(f"Train episodes: {train_episodes}, Test episodes: {test_episodes}")
            
            try:
                fold_result = self._run_sequential_fold(
                    fold_idx, train_episodes, test_episodes,
                    dataset, model, cv_data
                )
                fold_results.append(fold_result)
                sequential_results.append({
                    'fold': fold_idx,
                    'train_episodes': train_episodes,
                    'test_episodes': test_episodes,
                    **fold_result
                })
                
                # Save progress after each successful fold
                self._save_sequential_cv_progress(fold_idx + 1, len(cv_splits), sequential_results, 'in_progress')
                logger.info(f"Fold {fold_idx} completed and progress saved")
                
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                logger.error(f"Fold {fold_idx} failed: {e}")
                logger.error(f"Full traceback: {error_traceback}")
                print(f"❌ Fold {fold_idx} failed with error: {e}")
                print(f"Traceback: {error_traceback}")
                fold_results.append({'fold': fold_idx, 'error': str(e), 'traceback': error_traceback})
                sequential_results.append({'fold': fold_idx, 'error': str(e), 'traceback': error_traceback})
                
                # Save progress even for failed folds  
                self._save_sequential_cv_progress(fold_idx, len(cv_splits), sequential_results, 'in_progress')
                
                # Continue with next fold rather than stopping entire experiment
        
        # Aggregate results across folds
        print(f"📊 Total fold_results: {len(fold_results)}")
        for i, result in enumerate(fold_results):
            if 'error' in result:
                print(f"   Fold {i}: ERROR - {result.get('error', 'unknown error')}")
            else:
                print(f"   Fold {i}: SUCCESS")
        
        successful_folds = [r for r in fold_results if 'error' not in r]
        print(f"📊 Successful folds: {len(successful_folds)}")
        
        if not successful_folds:
            print("❌ No successful folds found!")
            raise RuntimeError("All CV folds failed")
        
        # Calculate summary metrics
        import numpy as np
        cv_accuracy_scores = [f['killer_test_accuracy'] for f in successful_folds if 'killer_test_accuracy' in f]
        
        results = {
            'training_paradigm': 'sequential_cv',
            'cv_folds_completed': len(successful_folds),
            'cv_folds_total': len(cv_splits),
            'cv_accuracy_mean': float(np.mean(cv_accuracy_scores)) if cv_accuracy_scores else 0.0,
            'cv_accuracy_std': float(np.std(cv_accuracy_scores)) if cv_accuracy_scores else 0.0,
            'fold_results': sequential_results,
            'successful_folds': len(successful_folds)
        }
        
        # Update metadata with results
        self.metadata['status'] = 'completed'
        self.metadata['completed_at'] = datetime.now().isoformat()
        self.metadata['final_results'] = results
        self.save_metadata()
        
        # Save final sequential CV progress
        self._save_sequential_cv_progress(len(cv_splits), len(cv_splits), sequential_results, 'completed')
        
        logger.info(f"Sequential CV experiment completed: {len(successful_folds)}/{len(cv_splits)} folds successful")
        logger.info(f"CV Accuracy: {results['cv_accuracy_mean']:.3f} ± {results['cv_accuracy_std']:.3f}")
        
        return results
    
    def _save_sequential_cv_progress(self, completed_folds: int, total_folds: int, 
                                   fold_results: List[Dict[str, Any]], status: str) -> None:
        """Save sequential CV progress for resume capability."""
        progress_file = self.experiment_dir / 'sequential_cv_progress.json'
        progress_data = {
            'completed_folds': completed_folds,
            'total_folds': total_folds,
            'fold_results': fold_results,
            'status': status,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        logger.debug(f"Sequential CV progress saved: {completed_folds}/{total_folds} folds")
    
    def _run_sequential_fold(self, fold_idx: int, train_episodes: List[str], test_episodes: List[str],
                           dataset: DidISayThisDataset, model_template: DidISayThisModel, 
                           cv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run training for a single CV fold in sequential paradigm."""
        logger.info(f"Fold {fold_idx}: Training on {len(train_episodes)} episodes, testing on {len(test_episodes)}")
        
        fold_dir = self.experiment_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Train embeddings on training episodes
            training_results, trained_embeddings = self._train_fold_embeddings(
                fold_idx, train_episodes, dataset, model_template, fold_dir, "training"
            )
            
            # Step 2: Train killer classifier on training embeddings
            killer_classifier, classifier_info = self._train_killer_classifier(
                fold_idx, train_episodes, trained_embeddings, cv_data
            )
            
            if killer_classifier is None:
                return {'fold': fold_idx, 'error': 'insufficient_training_data', **classifier_info}
            
            # Step 3: Train embeddings on test episodes  
            test_training_results, test_embeddings = self._train_fold_embeddings(
                fold_idx, test_episodes, dataset, model_template, fold_dir, "test_training"
            )
            
            # Step 4: Apply killer classifier to test embeddings
            test_results = self._evaluate_killer_classifier(
                fold_idx, test_episodes, test_embeddings, killer_classifier, cv_data
            )
            
            if 'error' in test_results:
                return {'fold': fold_idx, **test_results}
            
            # Combine results
            fold_result = {
                'fold': fold_idx,
                'train_episodes': train_episodes,
                'test_episodes': test_episodes,
                'training_results': training_results,
                'test_training_results': test_training_results,
                'killer_classifier_training': classifier_info,
                **test_results
            }
            
            # Save fold results
            self._save_fold_results(fold_dir, fold_result)
            
            logger.info(f"Fold {fold_idx}: Killer prediction accuracy = {test_results['killer_test_accuracy']:.3f}")
            return fold_result
            
        except Exception as e:
            logger.error(f"Fold {fold_idx} failed: {e}")
            return {'fold': fold_idx, 'error': str(e)}
    
    def _train_fold_embeddings(self, fold_idx: int, episodes: List[str], dataset: DidISayThisDataset,
                              model_template: DidISayThisModel, fold_dir: Path, phase: str) -> tuple:
        """Train character embeddings on given episodes."""
        print(f"🔧 Training fold {fold_idx} embeddings ({phase})")
        print(f"   Episodes: {episodes}")
        print(f"   Total episodes in dataset: {len(dataset.episodes)}")
        
        # Create episode subset
        try:
            episode_dataset = dataset.create_episode_subset(episodes)
            print(f"✅ Created episode subset with {len(episode_dataset)} samples")
            logger.info(f"Fold {fold_idx} {phase}: Created dataset with {len(episode_dataset)} samples")
        except Exception as e:
            print(f"❌ Failed to create episode subset: {e}")
            raise
        
        # Create fresh model
        import copy
        model = copy.deepcopy(model_template)
        torch.nn.init.normal_(model.character_embeddings.weight, std=0.02)
        
        # Configure training (no killer prediction during fold training)
        fold_config = copy.deepcopy(self.config)
        fold_config.killer_prediction_frequency = 0
        
        # Train
        trainer = Trainer(model=model, train_dataset=episode_dataset, val_dataset=None,
                         config=fold_config, experiment_manager=None)
        
        results = trainer.train(fold_dir / phase)
        embeddings = model.character_embeddings.weight.detach().cpu().numpy()
        
        logger.info(f"Fold {fold_idx} {phase}: Training completed - "
                   f"loss={results['final_train_loss']:.4f}, acc={results['final_train_accuracy']:.4f}")
        
        training_summary = {
            'train_loss': results['final_train_loss'],
            'train_accuracy': results['final_train_accuracy'], 
            'training_duration': results['training_duration_minutes']
        }
        
        return training_summary, embeddings
    
    def _train_killer_classifier(self, fold_idx: int, train_episodes: List[str], 
                                embeddings: 'np.ndarray', cv_data: Dict[str, Any]) -> tuple:
        """Train killer classifier on embeddings from training episodes."""
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        # Prepare training data
        X_train, y_train = [], []
        episode_killer_labels = cv_data['episode_killer_labels']
        episode_characters = cv_data['episode_characters'] 
        character_to_id = cv_data['character_to_id']
        
        for ep_id in train_episodes:
            if ep_id in episode_characters:
                for char_id in episode_characters[ep_id]:
                    if char_id in character_to_id:
                        embedding_idx = character_to_id[char_id]
                        X_train.append(embeddings[embedding_idx])
                        is_killer = episode_killer_labels[ep_id].get(char_id, False)
                        y_train.append(1 if is_killer else 0)
        
        classifier_info = {
            'train_characters': len(X_train),
            'train_killers': sum(y_train) if y_train else 0
        }
        
        if len(X_train) == 0 or len(set(y_train)) <= 1:
            logger.warning(f"Fold {fold_idx}: Insufficient training data for killer classifier")
            return None, classifier_info
        
        # Train classifier
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        classifier = LogisticRegression(
            random_state=cv_data['classifier_seed'],
            max_iter=1000,
            class_weight='balanced'
        )
        classifier.fit(X_train, y_train)
        
        logger.info(f"Fold {fold_idx}: Trained killer classifier on {len(X_train)} characters ({sum(y_train)} killers)")
        
        return classifier, classifier_info
    
    def _evaluate_killer_classifier(self, fold_idx: int, test_episodes: List[str],
                                  embeddings: 'np.ndarray', classifier, cv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply killer classifier to test embeddings and calculate metrics."""
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Prepare test data
        X_test, y_test, test_char_info = [], [], []
        episode_killer_labels = cv_data['episode_killer_labels']
        episode_characters = cv_data['episode_characters']
        character_to_id = cv_data['character_to_id']
        
        for ep_id in test_episodes:
            if ep_id in episode_characters:
                for char_id in episode_characters[ep_id]:
                    if char_id in character_to_id:
                        embedding_idx = character_to_id[char_id]
                        X_test.append(embeddings[embedding_idx])
                        is_killer = episode_killer_labels[ep_id].get(char_id, False)
                        y_test.append(1 if is_killer else 0)
                        test_char_info.append({
                            'character_id': char_id,
                            'episode_id': ep_id,
                            'is_killer': is_killer
                        })
        
        if len(X_test) == 0:
            logger.warning(f"Fold {fold_idx}: No test characters found")
            return {'error': 'no_test_characters', 'test_episodes': test_episodes}
        
        # Apply classifier
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1] if len(classifier.classes_) > 1 else np.zeros_like(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        return {
            'killer_test_accuracy': float(accuracy),
            'killer_test_precision': float(precision),
            'killer_test_recall': float(recall),
            'killer_test_f1': float(f1),
            'test_characters': len(X_test),
            'test_killers': int(sum(y_test)),
            'test_predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist(),
                'character_info': test_char_info
            }
        }
    
    def _save_fold_results(self, fold_dir: Path, fold_result: Dict[str, Any]) -> None:
        """Save fold results to JSON file."""
        import json
        fold_results_path = fold_dir / 'fold_results.json'
        with open(fold_results_path, 'w') as f:
            json.dump(fold_result, f, indent=2)
    
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

    def initialize_killer_prediction_setup(self, dataset: DidISayThisDataset, 
                                          initial_embeddings: torch.Tensor) -> None:
        """
        Initialize killer prediction cross-validation setup with fixed classifiers.
        
        Args:
            dataset: Dataset containing episodes and labels
            initial_embeddings: Initial character embeddings to train classifiers on
        """
        if not self.config.killer_prediction_frequency:
            return
            
        # Get CV splits from dataset using experiment seed
        self.killer_cv_data = dataset.get_killer_cv_splits(
            n_folds=self.config.killer_cv_folds,
            seed=self.config.killer_cv_seed
        )
        
        # Train fixed classifiers per fold using initial embeddings
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        embeddings_np = initial_embeddings.detach().cpu().numpy()
        cv_splits = self.killer_cv_data['splits']
        episode_killer_labels = self.killer_cv_data['episode_killer_labels']
        episode_characters = self.killer_cv_data['episode_characters']
        character_to_id = self.killer_cv_data['character_to_id']
        classifier_seed = self.killer_cv_data['classifier_seed']
        
        self.killer_classifiers = []
        self.killer_test_data = []  # Store test character info for each fold
        
        for fold_idx, (train_episodes, test_episodes) in enumerate(cv_splits):
            # Prepare training data for this fold
            X_train, y_train = [], []
            
            for ep_id in train_episodes:
                if ep_id in episode_characters:
                    for char_id in episode_characters[ep_id]:
                        if char_id in character_to_id:
                            embedding_idx = character_to_id[char_id]
                            X_train.append(embeddings_np[embedding_idx])
                            is_killer = episode_killer_labels[ep_id].get(char_id, False)
                            y_train.append(1 if is_killer else 0)
            
            # Prepare test data mapping for this fold
            test_chars = []
            for ep_id in test_episodes:
                if ep_id in episode_characters:
                    for char_id in episode_characters[ep_id]:
                        if char_id in character_to_id:
                            embedding_idx = character_to_id[char_id]
                            is_killer = episode_killer_labels[ep_id].get(char_id, False)
                            test_chars.append({
                                'character_id': char_id,
                                'embedding_idx': embedding_idx,
                                'is_killer': is_killer,
                                'episode_id': ep_id
                            })
            
            # Train classifier if we have sufficient data
            if len(X_train) > 0 and len(set(y_train)) > 1:
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                clf = LogisticRegression(
                    random_state=classifier_seed,
                    max_iter=1000,
                    class_weight='balanced'
                )
                clf.fit(X_train, y_train)
                
                self.killer_classifiers.append(clf)
                self.killer_test_data.append(test_chars)
                
                logger.info(f"Fold {fold_idx}: trained classifier on {len(X_train)} characters "
                           f"({sum(y_train)} killers)")
            else:
                logger.warning(f"Fold {fold_idx}: insufficient data for classifier training")
                self.killer_classifiers.append(None)
                self.killer_test_data.append([])
        
        # Initialize tracking
        self.killer_prediction_log = []
        
        logger.info(f"Initialized killer prediction with {len([c for c in self.killer_classifiers if c is not None])} "
                   f"fixed classifiers (seed={classifier_seed})")

    def evaluate_killer_prediction(self, current_embeddings: torch.Tensor, 
                                  step: int, epoch: int) -> Dict[str, Any]:
        """
        Evaluate killer prediction using fixed classifiers on current embeddings.
        
        Args:
            current_embeddings: Current character embedding matrix [vocab_size, embed_dim]
            step: Current training step
            epoch: Current training epoch
            
        Returns:
            Dictionary with CV results and metrics
        """
        if not hasattr(self, 'killer_classifiers') or not self.killer_classifiers:
            return {}
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import numpy as np
        
        # Convert to numpy
        embeddings_np = current_embeddings.detach().cpu().numpy()
        
        fold_results = []
        
        for fold_idx, (classifier, test_chars) in enumerate(zip(self.killer_classifiers, self.killer_test_data)):
            if classifier is None or not test_chars:
                continue
                
            try:
                # Get current embeddings for test characters
                X_test = np.array([embeddings_np[char['embedding_idx']] for char in test_chars])
                y_test = np.array([1 if char['is_killer'] else 0 for char in test_chars])
                
                # Apply fixed classifier to current embeddings
                y_pred = classifier.predict(X_test)
                y_pred_proba = classifier.predict_proba(X_test)[:, 1] if len(classifier.classes_) > 1 else np.zeros_like(y_pred)
                
                fold_results.append({
                    'fold': fold_idx,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
                    'n_test': len(y_test),
                    'test_killers': int(sum(y_test)),
                    'test_episodes': [char['episode_id'] for char in test_chars]
                })
                
            except Exception as e:
                logger.warning(f"Fold {fold_idx} evaluation failed: {e}")
                continue
        
        # Aggregate results across folds
        if not fold_results:
            logger.warning("No valid CV folds completed")
            return {}
        
        results = {
            'step': step,
            'epoch': epoch,
            'cv_accuracy_mean': float(np.mean([f['accuracy'] for f in fold_results])),
            'cv_accuracy_std': float(np.std([f['accuracy'] for f in fold_results])),
            'cv_precision_mean': float(np.mean([f['precision'] for f in fold_results])),
            'cv_recall_mean': float(np.mean([f['recall'] for f in fold_results])),
            'cv_f1_mean': float(np.mean([f['f1'] for f in fold_results])),
            'cv_folds_completed': len(fold_results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store detailed results
        detailed_results = {
            **results,
            'fold_details': fold_results
        }
        
        # Log to experiment
        self.killer_prediction_log.append(detailed_results)
        
        logger.info(f"Step {step} killer prediction CV: "
                   f"acc={results['cv_accuracy_mean']:.3f}±{results['cv_accuracy_std']:.3f}, "
                   f"f1={results['cv_f1_mean']:.3f}")
        
        return results

    def save_prediction_logs(self) -> None:
        """Save all prediction logs to CSV files."""
        if not hasattr(self, 'killer_prediction_log') or not self.killer_prediction_log:
            return
            
        import pandas as pd
        
        # Save killer prediction results  
        killer_df = pd.DataFrame([
            {k: v for k, v in result.items() if k != 'fold_details'} 
            for result in self.killer_prediction_log
        ])
        
        killer_path = self.experiment_dir / 'killer_predictions.csv' 
        killer_df.to_csv(killer_path, index=False)
        logger.info(f"Saved {len(self.killer_prediction_log)} killer prediction evaluations to {killer_path}")
        
        # Save detailed fold results
        fold_details = []
        for result in self.killer_prediction_log:
            for fold_detail in result.get('fold_details', []):
                fold_details.append({
                    'step': result['step'],
                    'epoch': result['epoch'],
                    'timestamp': result['timestamp'],
                    **fold_detail
                })
        
        if fold_details:
            fold_df = pd.DataFrame(fold_details)
            fold_path = self.experiment_dir / 'killer_predictions_by_fold.csv'
            fold_df.to_csv(fold_path, index=False)
            logger.info(f"Saved detailed fold results to {fold_path}")

    def get_holistic_results(self) -> Dict[str, Any]:
        """
        Get holistic experiment results combining all evaluation metrics.
        
        Returns:
            Comprehensive experiment results dictionary
        """
        results = {
            'experiment_config': {
                'experiment_name': self.experiment_name,
                'training_config': asdict(self.config),
                **self.metadata
            },
            'final_metrics': {},
            'temporal_evolution': {},
            'killer_prediction_summary': {}
        }
        
        # Add final training metrics if available
        if 'final_results' in self.metadata:
            training_results = self.metadata['final_results']
            results['final_metrics'].update({
                'did_i_say_this_final_accuracy': training_results.get('final_train_accuracy'),
                'did_i_say_this_final_loss': training_results.get('final_train_loss'),
                'training_duration_minutes': training_results.get('training_duration_minutes'),
                'total_steps': training_results.get('total_steps')
            })
        
        # Add killer prediction metrics if available
        if hasattr(self, 'killer_prediction_log') and self.killer_prediction_log:
            final_killer_result = self.killer_prediction_log[-1]
            results['final_metrics'].update({
                'killer_prediction_cv_accuracy_final': final_killer_result['cv_accuracy_mean'],
                'killer_prediction_cv_std_final': final_killer_result['cv_accuracy_std'],
                'killer_prediction_cv_f1_final': final_killer_result['cv_f1_mean']
            })
            
            # Temporal evolution
            results['temporal_evolution'] = {
                'steps': [r['step'] for r in self.killer_prediction_log],
                'killer_cv_accuracy_mean': [r['cv_accuracy_mean'] for r in self.killer_prediction_log],
                'killer_cv_accuracy_std': [r['cv_accuracy_std'] for r in self.killer_prediction_log],
                'killer_cv_f1_mean': [r['cv_f1_mean'] for r in self.killer_prediction_log]
            }
            
            # Summary statistics
            accuracies = [r['cv_accuracy_mean'] for r in self.killer_prediction_log]
            results['killer_prediction_summary'] = {
                'n_evaluations': len(self.killer_prediction_log),
                'accuracy_trend': 'improving' if accuracies[-1] > accuracies[0] else 'declining',
                'max_accuracy': max(accuracies),
                'min_accuracy': min(accuracies),
                'accuracy_improvement': accuracies[-1] - accuracies[0]
            }
        
        return results