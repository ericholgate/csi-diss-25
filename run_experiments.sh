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
    local holdout_killer=$4
    local holdout_pct=$5
    local num_epochs=$6
    local use_projection=$7
    local projection_dim=$8
    local seed=$9
    local batch_size=${10}
    local learning_rate=${11}
    
    echo "Starting experiment: $exp_name"
    
    # Create tmux session
    tmux new-session -d -s "$session_name" "bash"
    
    # Send commands to the tmux session
    tmux send-keys -t "$session_name" "cd $(pwd)" Enter
    tmux send-keys -t "$session_name" "source venv/bin/activate" Enter
    tmux send-keys -t "$session_name" "export PYTHONPATH=$(pwd)/src:$PYTHONPATH" Enter
    
    # Create the Python script for this experiment
    local script_content="
import sys
sys.path.append('src')
from pathlib import Path
from data.dataset import DidISayThisDataset
from model.architecture import DidISayThisModel, DidISayThisConfig
from model.trainer import TrainingConfig
from model.experiment import ExperimentManager, ExperimentConfig
import torch
import time

print('=== Experiment: $exp_name ===')
print('Configuration:')
print(f'  Character mode: $character_mode')
print(f'  Killer holdout: $holdout_killer')
print(f'  Holdout percentage: $holdout_pct')
print(f'  Epochs: $num_epochs')
print(f'  Use projection: $use_projection')
print(f'  Projection dim: $projection_dim')
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
        holdout_killer_reveal=$holdout_killer,
        killer_reveal_holdout_percentage=$holdout_pct,
        seed=$seed
    )
    
    print(f'Dataset created: {len(dataset)} examples')
    
    # Calculate expected training steps for progress tracking
    steps_per_epoch = len(dataset) // $batch_size
    total_steps = steps_per_epoch * $num_epochs
    print(f'Training plan: {steps_per_epoch} steps/epoch × {$num_epochs} epochs = {total_steps} total steps')
    print(f'Estimated time: {total_steps * 0.5 / 60:.1f}-{total_steps * 1.0 / 60:.1f} minutes (0.5-1.0 sec/step)')
    
    # Configure experiment
    experiment_config = ExperimentConfig(
        experiment_name='$exp_name',
        description='Systematic comparison of character embedding configurations',
        tags=['systematic', 'comparison', '$character_mode', 'seed$seed'],
        random_seed=$seed,
        save_dataset_with_experiment=True,
        save_final_embeddings=True
    )
    
    model_config = DidISayThisConfig(
        bert_model_name='bert-base-uncased',
        character_vocab_size=dataset.get_character_vocabulary_size(),
        freeze_bert=True,
        use_projection_layer=$use_projection,
        projection_dim=$projection_dim if $use_projection else None,
        dropout_rate=0.1,
        learning_rate=$learning_rate
    )
    
    training_config = TrainingConfig(
        num_epochs=$num_epochs,
        batch_size=$batch_size,
        learning_rate=$learning_rate,
        checkpoint_every_n_steps=500,
        log_every_n_steps=25,  # More frequent logging for better progress tracking
        eval_every_n_steps=250,
        holdout_killer_reveal=$holdout_killer,
        killer_reveal_holdout_percentage=$holdout_pct
    )
    
    # Check if experiment already exists and can be resumed
    experiment_dir = Path('experiments') / '$exp_name'
    can_resume = False
    
    if experiment_dir.exists():
        print(f'Experiment directory exists: {experiment_dir}')
        
        # Check if experiment completed successfully
        if (experiment_dir / 'final_results.json').exists():
            print('✅ Experiment already completed successfully!')
            print('Loading previous results...')
            
            with open(experiment_dir / 'final_results.json') as f:
                import json
                previous_results = json.load(f)
            
            print(f'Previous results: {previous_results.get(\"training_results\", \"N/A\")}')
            print(f'Experiment saved to: {experiment_dir}')
            print('Skipping re-training of completed experiment.')
            exit(0)
        
        # Check if experiment can be resumed
        elif (experiment_dir / 'experiment_config.json').exists():
            print('🔄 Incomplete experiment found. Checking if resumable...')
            
            # Look for checkpoint files
            checkpoint_files = list(experiment_dir.glob('*.pt'))
            if checkpoint_files:
                print(f'Found checkpoint files: {[f.name for f in checkpoint_files]}')
                can_resume = True
            else:
                print('No checkpoint files found. Starting fresh...')
                # Clean up incomplete experiment
                import shutil
                shutil.rmtree(experiment_dir)
        else:
            print('Incomplete experiment directory found. Cleaning up...')
            import shutil
            shutil.rmtree(experiment_dir)
    
    if can_resume:
        print('📂 Resuming from existing experiment...')
        
        # Load existing experiment
        experiment_manager = ExperimentManager.load_experiment(experiment_dir)
        
        # Load associated datasets
        print('Loading saved datasets...')
        train_dataset, val_dataset = experiment_manager.load_datasets()
        print(f'Loaded datasets: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}')
        
        # Set up experiment (this will restore from saved state)
        print('Restoring model from checkpoint...')
        model = experiment_manager.setup_experiment(train_dataset, val_dataset)
        
        # Find the most recent checkpoint to resume from
        final_checkpoint = experiment_dir / 'final.pt'
        best_checkpoint = experiment_dir / 'best_model.pt'
        
        checkpoint_to_resume = None
        if final_checkpoint.exists():
            checkpoint_to_resume = 'final'
            print('Resuming from final checkpoint...')
        elif best_checkpoint.exists():
            checkpoint_to_resume = 'best_model'
            print('Resuming from best model checkpoint...')
        
        if checkpoint_to_resume:
            print('🔄 Resuming training from checkpoint...')
            results = experiment_manager.resume_training(checkpoint_to_resume)
        else:
            print('⚠️ No suitable checkpoint found for resuming. Starting fresh training...')
            results = experiment_manager.train_model()
    
    else:
        print('🆕 Starting new experiment...')
        
        # Set up experiment
        print('Setting up experiment manager...')
        experiment_manager = ExperimentManager(
            experiment_config=experiment_config,
            model_config=model_config,
            training_config=training_config
        )
        
        model = experiment_manager.setup_experiment(dataset, None)
        print(f'Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        
        # Train model
        print('Starting training...')
        results = experiment_manager.train_model()
    
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
    
    # Write Python script to temp file
    echo "$script_content" > "/tmp/exp_${session_name}.py"
    
    # Run the experiment
    tmux send-keys -t "$session_name" "python /tmp/exp_${session_name}.py 2>&1 | tee experiment_logs/${exp_name}.log" Enter
    
    echo "  → Experiment $exp_name started in tmux session: $session_name"
}

# Define experimental configurations
echo "Defining experimental configurations..."

# Base configurations
declare -a EXPERIMENTS=(
    # Format: session_name|exp_name|character_mode|holdout_killer|holdout_pct|num_epochs|use_projection|projection_dim|seed|batch_size|learning_rate
    
    # Episode-isolated mode experiments
    "exp01|ep_iso_1ep_proj_s42|episode-isolated|False|0.1|1|True|512|42|16|1e-4"
    "exp02|ep_iso_1ep_proj_s123|episode-isolated|False|0.1|1|True|512|123|16|1e-4"
    "exp03|ep_iso_1ep_noproj_s42|episode-isolated|False|0.1|1|False|512|42|16|1e-4"
    "exp04|ep_iso_1ep_noproj_s123|episode-isolated|False|0.1|1|False|512|123|16|1e-4"
    "exp05|ep_iso_5ep_proj_s42|episode-isolated|True|0.1|5|True|512|42|16|1e-4"
    "exp06|ep_iso_5ep_proj_s123|episode-isolated|True|0.1|5|True|512|123|16|1e-4"
    "exp07|ep_iso_5ep_noproj_s42|episode-isolated|True|0.1|5|False|512|42|16|1e-4"
    "exp08|ep_iso_5ep_noproj_s123|episode-isolated|True|0.1|5|False|512|123|16|1e-4"
    
    # Cross-episode mode experiments  
    "exp09|cross_ep_1ep_proj_s42|cross-episode|False|0.1|1|True|512|42|16|1e-4"
    "exp10|cross_ep_1ep_proj_s123|cross-episode|False|0.1|1|True|512|123|16|1e-4"
    "exp11|cross_ep_1ep_noproj_s42|cross-episode|False|0.1|1|False|512|42|16|1e-4"
    "exp12|cross_ep_1ep_noproj_s123|cross-episode|False|0.1|1|False|512|123|16|1e-4"
    "exp13|cross_ep_5ep_proj_s42|cross-episode|True|0.1|5|True|512|42|16|1e-4"
    "exp14|cross_ep_5ep_proj_s123|cross-episode|True|0.1|5|True|512|123|16|1e-4"
    "exp15|cross_ep_5ep_noproj_s42|cross-episode|True|0.1|5|False|512|42|16|1e-4"
    "exp16|cross_ep_5ep_noproj_s123|cross-episode|True|0.1|5|False|512|123|16|1e-4"
)

echo "Total experiments planned: ${#EXPERIMENTS[@]}"
echo

# Kill any existing experimental tmux sessions
echo "Cleaning up any existing experimental tmux sessions..."
for i in {1..16}; do
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
    holdout_killer=${PARAMS[3]}
    holdout_pct=${PARAMS[4]}
    num_epochs=${PARAMS[5]}
    use_projection=${PARAMS[6]}
    projection_dim=${PARAMS[7]}
    seed=${PARAMS[8]}
    batch_size=${PARAMS[9]}
    learning_rate=${PARAMS[10]}
    
    run_experiment "$session_name" "$exp_name" "$character_mode" "$holdout_killer" \
                  "$holdout_pct" "$num_epochs" "$use_projection" "$projection_dim" \
                  "$seed" "$batch_size" "$learning_rate"
    
    # Small delay between starting experiments to avoid resource conflicts
    sleep 2
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
echo "Experimental Design Summary:"
echo "  • 16 total experiments (2 character modes × 2 holdout settings × 2 projection settings × 2 seeds)"
echo "  • Episode-isolated vs Cross-episode character modes"
echo "  • Single epoch (no holdout) vs 5-epoch (with 10% killer reveal holdout)"
echo "  • With projection (512-dim) vs Without projection (direct 1536-dim)"
echo "  • Two random seeds per configuration for statistical robustness"
echo "  • All experiments use batch_size=16, learning_rate=1e-4 for CPU efficiency"
echo
echo "Expected runtime: ~2-4 hours per experiment on CPU (varies by dataset size)"
echo "Monitor with: watch 'tmux list-sessions | grep -c exp'"