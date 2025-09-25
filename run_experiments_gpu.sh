#!/bin/bash

# CSI Character Embedding Experiments - GPU Optimized
# For use on AWS g4dn.xlarge or similar GPU instances
# 
# Usage: bash run_experiments_gpu.sh
# 
# This script will run experiments in tmux sessions optimized for GPU training

set -e  # Exit on error

echo "=== CSI Character Embedding Experiment Suite (GPU Optimized) ==="
echo "Starting comprehensive experimental comparison..."
echo

# Check if we're in the right directory
if [ ! -f "src/data/dataset.py" ]; then
    echo "Error: Please run this script from the csi-diss-25 root directory"
    exit 1
fi

# Check if virtual environment exists or use conda
if [ -d "venv" ]; then
    PYTHON_ENV="source venv/bin/activate"
elif command -v conda &> /dev/null; then
    PYTHON_ENV="conda activate pytorch"
else
    echo "Error: No virtual environment found. Please create venv or use conda."
    exit 1
fi

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ GPU detected"
else
    echo "⚠️  Warning: nvidia-smi not found. Make sure you're on a GPU instance."
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
    echo "  GPU Batch size: $batch_size (optimized for GPU)"
    
    # Create tmux session
    tmux new-session -d -s "$session_name" "bash"
    
    # Send commands to the tmux session
    tmux send-keys -t "$session_name" "cd $(pwd)" Enter
    tmux send-keys -t "$session_name" "$PYTHON_ENV" Enter
    tmux send-keys -t "$session_name" "export PYTHONPATH=$(pwd)/src:$PYTHONPATH" Enter
    tmux send-keys -t "$session_name" "export CUDA_VISIBLE_DEVICES=0" Enter
    
    # Small delay to ensure commands execute
    sleep 1
    
    # Verify environment is working
    tmux send-keys -t "$session_name" "which python" Enter
    tmux send-keys -t "$session_name" "python --version" Enter
    tmux send-keys -t "$session_name" "python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')\"" Enter
    
    # Wait a moment for environment verification
    sleep 3
    
    # Create the Python script for this experiment
    local script_content="
import sys
import os
print(f'Python executable: {sys.executable}')
print(f'Python version: {sys.version}')
print(f'Current working directory: {os.getcwd()}')

# Set up Python path for absolute imports
project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f'Python path: {sys.path[:3]}...')  # Show first few entries

# Change to project root directory 
os.chdir(project_root)

try:
    print('Importing modules...')
    from pathlib import Path
    print('✓ pathlib imported')
    
    from data.dataset import DidISayThisDataset
    print('✓ data.dataset imported')
    
    from model.architecture import DidISayThisModel, DidISayThisConfig
    print('✓ model.architecture imported')
    
    from model.trainer import TrainingConfig
    print('✓ model.trainer imported')
    
    from model.experiment import ExperimentManager
    print('✓ model.experiment imported')
    
    import torch
    print('✓ torch imported')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    import time
    print('✓ All imports successful!')
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)

print('=== Experiment: $exp_name (GPU Optimized) ===')
print('Configuration:')
print(f'  Character mode: $character_mode')
print(f'  Epochs: $num_epochs')
print(f'  Sequential CV: $sequential_cv')
print(f'  Seed: $seed')
print(f'  Batch size: $batch_size (GPU optimized)')
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
    
    # Calculate expected training steps for sequential CV (GPU optimized)
    if '$sequential_cv' == 'true':
        # Sequential CV trains 5 separate models per fold (5 folds = 10 models total)
        steps_per_fold_model = len(dataset) // 5 // $batch_size  # Approximate steps per fold
        total_models = 10  # 5 folds × 2 models per fold (train + test)
        total_steps = steps_per_fold_model * total_models * $num_epochs
        print(f'Sequential CV plan: ~{steps_per_fold_model} steps × {total_models} models × {$num_epochs} epochs = ~{total_steps} total steps')
        print(f'Estimated time (GPU): {total_steps * 0.05 / 60:.1f}-{total_steps * 0.1 / 60:.1f} minutes (much faster!)')
    else:
        # Standard training
        steps_per_epoch = len(dataset) // $batch_size
        total_steps = steps_per_epoch * $num_epochs
        print(f'Standard training: {steps_per_epoch} steps/epoch × {$num_epochs} epochs = {total_steps} total steps')
        print(f'Estimated time (GPU): {total_steps * 0.02 / 60:.1f}-{total_steps * 0.05 / 60:.1f} minutes')
    
    # Configure model and training (GPU optimized)
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
        batch_size=$batch_size,  # GPU optimized batch size
        learning_rate=$learning_rate,
        checkpoint_every_n_steps=200,  # More frequent checkpoints for GPU
        log_every_n_steps=10,  # More frequent logging
        
        # Killer prediction evaluation (key feature)
        killer_prediction_frequency=100,  # More frequent evaluation with GPU
        killer_cv_folds=5,
        killer_cv_seed=42,
        
        # Sequential vs parallel CV training
        sequential_cv_training=('$sequential_cv' == 'true')
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
    
    # Create model and move to GPU
    model = DidISayThisModel(model_config)
    if torch.cuda.is_available():
        model = model.cuda()
        print(f'Model moved to GPU: {torch.cuda.get_device_name(0)}')
    
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
    local script_file="experiment_logs/exp_${session_name}_gpu.py"
    echo "$script_content" > "$script_file"
    
    # Run the experiment from project root
    tmux send-keys -t "$session_name" "python $script_file 2>&1 | tee experiment_logs/${exp_name}_gpu.log" Enter
    
    echo "  → GPU experiment $exp_name started in tmux session: $session_name"
}

# Define experimental configurations (GPU Optimized)
echo "Defining GPU-optimized experimental configurations..."

# Phase 1: Sequential CV Validation (4 experiments) - GPU Optimized
declare -a EXPERIMENTS=(
    # Format: session_name|exp_name|character_mode|num_epochs|seed|batch_size|learning_rate|sequential_cv
    
    # Test sequential CV training paradigm (larger batch sizes for GPU)
    "exp01|seq_cv_ep_iso_s42_gpu|episode-isolated|2|42|64|2e-5|true"
    "exp02|seq_cv_cross_ep_s42_gpu|cross-episode|2|42|64|2e-5|true"
    
    # Compare with parallel CV training (larger batch sizes for GPU)
    "exp03|parallel_cv_ep_iso_s42_gpu|episode-isolated|2|42|64|2e-5|false"
    "exp04|parallel_cv_cross_ep_s42_gpu|cross-episode|2|42|64|2e-5|false"
)

echo "Total GPU-optimized experiments planned: ${#EXPERIMENTS[@]}"
echo

# Kill any existing experimental tmux sessions
echo "Cleaning up any existing experimental tmux sessions..."
for i in {1..4}; do
    session_name=$(printf "exp%02d" $i)
    tmux kill-session -t "$session_name" 2>/dev/null || true
done

echo "Starting GPU-optimized experiments..."
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
    sleep 3
done

echo
echo "All GPU-optimized experiments started!"
echo
echo "GPU Performance Notes:"
echo "  • Batch size increased to 64 (4x larger than CPU)"
echo "  • Learning rate reduced to 2e-5 (more stable with larger batches)" 
echo "  • 2 epochs instead of 1 (faster training allows more epochs)"
echo "  • More frequent checkpointing and logging"
echo
echo "Expected GPU Performance:"
echo "  • Sequential CV: ~15-30 minutes per experiment (vs 3-6 hours CPU)"
echo "  • Parallel CV: ~5-10 minutes per experiment (vs 30-60 minutes CPU)"
echo "  • Total runtime: ~1-2 hours for all experiments (vs 6-12 hours CPU)"
echo
echo "To monitor experiments:"
echo "  tmux list-sessions                    # List all sessions"
echo "  tmux attach-session -t exp01         # Attach to experiment 1"
echo "  tmux detach-session                  # Detach (Ctrl+B, then D)"
echo "  tail -f experiment_logs/[exp_name]_gpu.log  # Monitor log files"
echo "  nvidia-smi                           # Monitor GPU usage"
echo
echo "To check progress:"
echo "  ls experiments/                       # List completed experiments"
echo "  du -sh experiments/*/                # Check experiment sizes"
echo "  watch nvidia-smi                     # Live GPU monitoring"