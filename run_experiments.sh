#!/bin/bash

# CSI Character Embedding Experiments
# Comprehensive experimental comparison of different configurations
# 
# Usage: bash run_experiments.sh
# 
# This script will run experiments in tmux sessions so they continue
# even if SSH connection is lost.

set -e  # Exit on error

echo "=== CSI Character Embedding Experiment Suite ==="
echo "Starting comprehensive experimental comparison..."
echo

# Check if we're in the right directory
if [ ! -f "src/data/dataset.py" ]; then
    echo "Error: Please run this script from the csi-diss-25 root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment 'venv' not found. Please create it first."
    exit 1
fi

# Create experiments directory if it doesn't exist
mkdir -p experiments
mkdir -p experiment_logs

# Function to create a tmux session and run experiment
run_experiment() {
    local session_name=$1
    local exp_name=$2
    local character_mode=$3
    local num_epochs=$4
    local seed=$5
    local batch_size=$6
    local learning_rate=$7
    local sequential_cv=$8
    
    echo "Starting experiment: $exp_name"
    echo "  Character mode: $character_mode, Epochs: $num_epochs, Sequential CV: $sequential_cv, Seed: $seed"
    
    # Create tmux session
    tmux new-session -d -s "$session_name" "bash"
    
    # Send commands to the tmux session
    tmux send-keys -t "$session_name" "cd $(pwd)" Enter
    tmux send-keys -t "$session_name" "source venv/bin/activate" Enter
    tmux send-keys -t "$session_name" "export PYTHONPATH=$(pwd)/src:$PYTHONPATH" Enter
    
    # Create the Python script for this experiment
    local script_content="
import sys
import os
# Set up Python path for absolute imports
project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Change to project root directory 
os.chdir(project_root)

from pathlib import Path
from data.dataset import DidISayThisDataset
from model.architecture import DidISayThisModel, DidISayThisConfig
from model.trainer import TrainingConfig
from model.experiment import ExperimentManager
import torch
import time

print('=== Experiment: $exp_name ===')
print('Configuration:')
print(f'  Character mode: $character_mode')
print(f'  Epochs: $num_epochs')
print(f'  Sequential CV: $sequential_cv')
print(f'  Seed: $seed')
print(f'  Batch size: $batch_size')
print(f'  Learning rate: $learning_rate')
print()

try:
    start_time = time.time()
    
    # Create dataset
    print('Creating dataset...')
    dataset = DidISayThisDataset.from_data_directory(
        Path('data/original'),
        character_mode='$character_mode',
        seed=$seed
    )
    
    print(f'Dataset created: {len(dataset)} examples')
    print(f'Character vocabulary size: {dataset.get_character_vocabulary_size()}')
    
    # Calculate expected training steps for sequential CV
    if '$sequential_cv' == 'true':
        # Sequential CV trains 5 separate models per fold (5 folds = 10 models total)
        steps_per_fold_model = len(dataset) // 5 // $batch_size  # Approximate steps per fold
        total_models = 10  # 5 folds × 2 models per fold (train + test)
        total_steps = steps_per_fold_model * total_models * $num_epochs
        print(f'Sequential CV plan: ~{steps_per_fold_model} steps × {total_models} models × {$num_epochs} epochs = ~{total_steps} total steps')
        print(f'Estimated time: {total_steps * 0.8 / 60:.1f}-{total_steps * 1.5 / 60:.1f} minutes (CV overhead)')
    else:
        # Standard training
        steps_per_epoch = len(dataset) // $batch_size
        total_steps = steps_per_epoch * $num_epochs
        print(f'Standard training: {steps_per_epoch} steps/epoch × {$num_epochs} epochs = {total_steps} total steps')
        print(f'Estimated time: {total_steps * 0.5 / 60:.1f}-{total_steps * 1.0 / 60:.1f} minutes')
    
    # Configure model and training
    model_config = DidISayThisConfig(
        bert_model_name='bert-base-uncased',
        character_vocab_size=dataset.get_character_vocabulary_size(),
        freeze_bert=True,
        use_projection_layer=True,  # Phase 1: use projection for consistency
        projection_dim=512,
        dropout_rate=0.1
    )
    
    training_config = TrainingConfig(
        num_epochs=$num_epochs,
        batch_size=$batch_size,
        learning_rate=$learning_rate,
        checkpoint_every_n_steps=500,
        log_every_n_steps=25,
        
        # Killer prediction evaluation (key feature)
        killer_prediction_frequency=200,  # Evaluate every 200 steps
        killer_cv_folds=5,
        killer_cv_seed=42,
        
        # Sequential vs parallel CV training
        sequential_cv_training=($sequential_cv == 'true')
    )
    
    # Check if experiment already exists
    experiment_dir = Path('experiments') / '$exp_name'
    
    if experiment_dir.exists():
        print(f'Experiment directory exists: {experiment_dir}')
        
        # Check if experiment completed successfully  
        metadata_file = experiment_dir / 'experiment_metadata.json'
        progress_file = experiment_dir / 'sequential_cv_progress.json'
        
        # Check completion status
        if metadata_file.exists():
            with open(metadata_file) as f:
                import json
                metadata = json.load(f)
            
            if metadata.get('status') == 'completed':
                print('✅ Experiment already completed successfully!')
                print(f'Previous results: {metadata.get(\"final_results\", \"N/A\")}')
                print('Skipping re-training of completed experiment.')
                exit(0)
        
        # Check sequential CV progress for resuming
        if progress_file.exists() and '$sequential_cv' == 'true':
            with open(progress_file) as f:
                import json
                progress = json.load(f)
            
            if progress.get('status') == 'completed':
                print('✅ Sequential CV already completed!')
                exit(0)
            elif progress.get('completed_folds', 0) > 0:
                completed = progress.get('completed_folds', 0)
                total = progress.get('total_folds', 5)
                print(f'🔄 Found partial sequential CV progress: {completed}/{total} folds completed')
                print('Will resume from where we left off...')
                # Don't delete directory - let experiment manager handle resume
            else:
                print('🔄 Found sequential CV setup but no progress. Starting fresh...')
                import shutil
                shutil.rmtree(experiment_dir)
        else:
            print('🔄 Incomplete experiment found. Starting fresh...')
            import shutil
            shutil.rmtree(experiment_dir)
    
    # Create and run new experiment
    print('🆕 Starting new experiment...')
    
    # Create model
    model = DidISayThisModel(model_config)
    print(f'Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    
    # Create experiment manager and run
    experiment_manager = ExperimentManager('$exp_name', str(experiment_dir), training_config)
    results = experiment_manager.run_experiment(dataset, model)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f'Training completed successfully!')
    print(f'Duration: {duration/60:.1f} minutes')
    print(f'Final results: {results}')
    print(f'Experiment saved to: {experiment_manager.experiment_dir}')
    
except Exception as e:
    print(f'EXPERIMENT FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
    
    # Write Python script to experiment logs directory
    local script_file="experiment_logs/exp_${session_name}.py"
    echo "$script_content" > "$script_file"
    
    # Run the experiment from project root
    tmux send-keys -t "$session_name" "python $script_file 2>&1 | tee experiment_logs/${exp_name}.log" Enter
    
    echo "  → Experiment $exp_name started in tmux session: $session_name"
}

# Define experimental configurations
echo "Defining experimental configurations..."

# Phase 1: Sequential CV Validation (4 experiments)
declare -a EXPERIMENTS=(
    # Format: session_name|exp_name|character_mode|num_epochs|seed|batch_size|learning_rate|sequential_cv
    
    # Test sequential CV training paradigm
    "exp01|seq_cv_ep_iso_s42|episode-isolated|1|42|16|1e-4|true"
    "exp02|seq_cv_cross_ep_s42|cross-episode|1|42|16|1e-4|true"
    
    # Compare with parallel CV training  
    "exp03|parallel_cv_ep_iso_s42|episode-isolated|1|42|16|1e-4|false"
    "exp04|parallel_cv_cross_ep_s42|cross-episode|1|42|16|1e-4|false"
)

echo "Total experiments planned: ${#EXPERIMENTS[@]}"
echo

# Kill any existing experimental tmux sessions
echo "Cleaning up any existing experimental tmux sessions..."
for i in {1..4}; do
    session_name=$(printf "exp%02d" $i)
    tmux kill-session -t "$session_name" 2>/dev/null || true
done

echo "Starting experiments..."
echo

# Start all experiments
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -ra PARAMS <<< "$exp"
    
    session_name=${PARAMS[0]}
    exp_name=${PARAMS[1]}
    character_mode=${PARAMS[2]}
    num_epochs=${PARAMS[3]}
    seed=${PARAMS[4]}
    batch_size=${PARAMS[5]}
    learning_rate=${PARAMS[6]}
    sequential_cv=${PARAMS[7]}
    
    run_experiment "$session_name" "$exp_name" "$character_mode" "$num_epochs" \
                  "$seed" "$batch_size" "$learning_rate" "$sequential_cv"
    
    # Small delay between starting experiments to avoid resource conflicts
    sleep 5
done

echo
echo "All experiments started!"
echo
echo "To monitor experiments:"
echo "  tmux list-sessions                    # List all sessions"
echo "  tmux attach-session -t exp01         # Attach to experiment 1"
echo "  tmux detach-session                  # Detach (Ctrl+B, then D)"
echo "  tail -f experiment_logs/[exp_name].log  # Monitor log files"
echo
echo "To check progress:"
echo "  ls experiments/                       # List completed experiments"
echo "  du -sh experiments/*/                # Check experiment sizes"
echo
echo "Phase 1 Experimental Design Summary:"
echo "  • 4 total experiments (2 character modes × 2 training paradigms)"
echo "  • Episode-isolated vs Cross-episode character modes"
echo "  • Sequential CV vs Parallel CV training paradigms"
echo "  • Single epoch training with killer prediction evaluation (200 steps)"
echo "  • Fixed architecture: projection layer (512-dim), seed=42"
echo "  • All experiments use batch_size=16, learning_rate=1e-4 for CPU efficiency"
echo
echo "Expected runtime per experiment:"
echo "  • Sequential CV: ~3-6 hours (trains 10 models per experiment)"
echo "  • Parallel CV: ~30-60 minutes (trains 1 model per experiment)"
echo "Monitor with: watch 'tmux list-sessions | grep -c exp'"