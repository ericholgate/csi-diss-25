#!/usr/bin/env python3
"""
Quick test run using a small subset of data to verify the pipeline works end-to-end.
This should complete in a few minutes even on CPU.
"""

import sys
import os
from pathlib import Path
import time
import json

# Set up Python path for absolute imports
project_root = Path(__file__).parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")

try:
    print("Importing modules...")
    from data.dataset import DidISayThisDataset
    from model.architecture import DidISayThisModel, DidISayThisConfig
    from model.trainer import TrainingConfig
    from model.experiment import ExperimentManager
    import torch
    print("✓ All imports successful!")
    
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print("\n=== Small Dataset Test Run ===")
print("This test uses only 3 episodes and minimal training to verify the pipeline works.")
print()

try:
    start_time = time.time()
    
    # Create dataset from small test data
    print("Creating small dataset...")
    dataset = DidISayThisDataset.from_data_directory(
        Path('data/test_small'),
        character_mode='episode-isolated',  # Use episode-isolated for simpler test
        seed=42
    )
    
    print(f"✓ Small dataset created: {len(dataset)} examples")
    print(f"✓ Character vocabulary size: {dataset.get_character_vocabulary_size()}")
    print(f"✓ Episodes: {len(dataset.episodes)}")
    
    # Quick training config for fast testing
    model_config = DidISayThisConfig(
        bert_model_name='bert-base-uncased',
        character_vocab_size=dataset.get_character_vocabulary_size(),
        freeze_bert=True,
        use_projection_layer=True,
        projection_dim=256,  # Smaller for faster test
        dropout_rate=0.1
    )
    
    training_config = TrainingConfig(
        num_epochs=1,  # Just 1 epoch for quick test
        batch_size=8,  # Small batch size for compatibility
        learning_rate=2e-5,
        checkpoint_every_n_steps=50,
        log_every_n_steps=10,
        
        # Quick killer prediction evaluation
        killer_prediction_frequency=50,
        killer_cv_folds=3,  # Reduced folds for faster testing
        killer_cv_seed=42,
        
        # Test sequential CV training
        sequential_cv_training=True
    )
    
    print(f"✓ Training config: {training_config.num_epochs} epoch, batch_size={training_config.batch_size}")
    
    # Create model
    model = DidISayThisModel(model_config)
    if torch.cuda.is_available():
        model = model.cuda()
        print("✓ Model moved to GPU")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created with {trainable_params:,} trainable parameters")
    
    # Estimate runtime
    steps_per_epoch = len(dataset) // training_config.batch_size
    print(f"✓ Estimated steps: {steps_per_epoch} steps/epoch")
    
    if torch.cuda.is_available():
        estimated_time = steps_per_epoch * 0.05  # ~0.05 seconds per step on GPU
        print(f"✓ Estimated time (GPU): {estimated_time/60:.1f} minutes")
    else:
        estimated_time = steps_per_epoch * 0.5   # ~0.5 seconds per step on CPU
        print(f"✓ Estimated time (CPU): {estimated_time/60:.1f} minutes")
    
    # Create test experiment
    experiment_name = "test_small_pipeline"
    experiment_dir = project_root / 'experiments' / experiment_name
    
    # Clean up any existing test experiment
    if experiment_dir.exists():
        import shutil
        shutil.rmtree(experiment_dir)
    
    print(f"✓ Running experiment: {experiment_name}")
    
    # Run experiment
    experiment_manager = ExperimentManager(experiment_name, str(experiment_dir), training_config)
    results = experiment_manager.run_experiment(dataset, model)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n🎉 TEST COMPLETED SUCCESSFULLY! 🎉")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Results: {results}")
    print(f"Experiment saved to: {experiment_dir}")
    
    # Verify key output files exist
    print("\nVerifying output files:")
    expected_files = [
        'experiment_metadata.json',
        'model_checkpoints',
        'predictions'
    ]
    
    for file_name in expected_files:
        file_path = experiment_dir / file_name
        if file_path.exists():
            if file_path.is_dir():
                file_count = len(list(file_path.iterdir()))
                print(f"✓ {file_name}/ directory exists with {file_count} files")
            else:
                file_size = file_path.stat().st_size
                print(f"✓ {file_name} exists ({file_size} bytes)")
        else:
            print(f"⚠️  {file_name} not found")
    
    print(f"\n✅ Pipeline test complete! Ready for full AWS GPU runs.")
    print(f"Full dataset would be ~{39/3:.0f}x larger with {39} episodes instead of 3.")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)