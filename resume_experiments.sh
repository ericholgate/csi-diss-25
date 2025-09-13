#!/bin/bash

# Resume CSI Character Embedding Experiments after VM restart
# This script automatically detects incomplete experiments and resumes them
#
# Usage: bash resume_experiments.sh

set -e

echo "=== CSI Experiment Resume Manager ==="
echo "Detecting incomplete experiments and resuming training..."
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

# Kill any existing experimental tmux sessions first
echo "Cleaning up any existing tmux sessions..."
for i in {1..4}; do
    session_name=$(printf "exp%02d" $i)
    tmux kill-session -t "$session_name" 2>/dev/null || true
done

# Check experiment status and collect resumable experiments
declare -a RESUMABLE_EXPERIMENTS=()
declare -a COMPLETED_EXPERIMENTS=()
declare -a NOT_STARTED_EXPERIMENTS=()

# Define experiment configurations (same as run_experiments.sh)
# Phase 1: Sequential CV Validation (4 experiments)
declare -a ALL_EXPERIMENTS=(
    # Format: session_name|exp_name|character_mode|num_epochs|seed|batch_size|learning_rate|sequential_cv
    "exp01|seq_cv_ep_iso_s42|episode-isolated|1|42|16|1e-4|true"
    "exp02|seq_cv_cross_ep_s42|cross-episode|1|42|16|1e-4|true"
    "exp03|parallel_cv_ep_iso_s42|episode-isolated|1|42|16|1e-4|false"
    "exp04|parallel_cv_cross_ep_s42|cross-episode|1|42|16|1e-4|false"
)

echo "🔍 Analyzing experiment status..."

# Analyze each experiment
for exp in "${ALL_EXPERIMENTS[@]}"; do
    IFS='|' read -ra PARAMS <<< "$exp"
    
    session_name=${PARAMS[0]}
    exp_name=${PARAMS[1]}
    
    exp_dir="experiments/$exp_name"
    
    if [ -d "$exp_dir" ]; then
        if [ -f "$exp_dir/final_results.json" ]; then
            # Experiment completed
            COMPLETED_EXPERIMENTS+=("$exp")
            echo "  ✅ $exp_name: COMPLETED"
        elif [ -f "$exp_dir/experiment_config.json" ]; then
            # Check for checkpoint files
            checkpoint_count=$(find "$exp_dir" -name "*.pt" | wc -l)
            if [ "$checkpoint_count" -gt 0 ]; then
                # Has checkpoints, can resume
                RESUMABLE_EXPERIMENTS+=("$exp")
                echo "  🔄 $exp_name: RESUMABLE (${checkpoint_count} checkpoints)"
            else
                # No checkpoints, treat as not started
                echo "  ⚠️  $exp_name: INCOMPLETE (no checkpoints, will restart)"
                rm -rf "$exp_dir"  # Clean up
                NOT_STARTED_EXPERIMENTS+=("$exp")
            fi
        else
            # Incomplete directory, clean up
            echo "  🗑️  $exp_name: CLEANING UP incomplete directory"
            rm -rf "$exp_dir"
            NOT_STARTED_EXPERIMENTS+=("$exp")
        fi
    else
        # Not started
        NOT_STARTED_EXPERIMENTS+=("$exp")
        echo "  ⏸️  $exp_name: NOT STARTED"
    fi
done

echo
echo "📊 EXPERIMENT STATUS SUMMARY"
echo "============================="
echo "Completed: ${#COMPLETED_EXPERIMENTS[@]}/4"
echo "Resumable: ${#RESUMABLE_EXPERIMENTS[@]}/4"
echo "Not Started: ${#NOT_STARTED_EXPERIMENTS[@]}/4"

# Combine resumable and not-started for execution
EXPERIMENTS_TO_RUN=("${RESUMABLE_EXPERIMENTS[@]}" "${NOT_STARTED_EXPERIMENTS[@]}")

if [ ${#EXPERIMENTS_TO_RUN[@]} -eq 0 ]; then
    echo
    echo "🎉 All experiments completed! Nothing to resume."
    echo "Use 'bash monitor_experiments.sh' to view results."
    exit 0
fi

echo
echo "🚀 RESUMING EXPERIMENTS"
echo "======================="
echo "Will start/resume ${#EXPERIMENTS_TO_RUN[@]} experiments:"

for exp in "${EXPERIMENTS_TO_RUN[@]}"; do
    IFS='|' read -ra PARAMS <<< "$exp"
    exp_name=${PARAMS[1]}
    echo "  - $exp_name"
done

echo
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Source the experiment running function from run_experiments.sh
source <(grep -A 50 "run_experiment()" run_experiments.sh | head -n 51)

echo
echo "Starting experiments..."

# Run experiments with staggered delays
delay=0
for exp in "${EXPERIMENTS_TO_RUN[@]}"; do
    IFS='|' read -ra PARAMS <<< "$exp"
    
    session_name=${PARAMS[0]}
    exp_name=${PARAMS[1]}
    character_mode=${PARAMS[2]}
    num_epochs=${PARAMS[3]}
    seed=${PARAMS[4]}
    batch_size=${PARAMS[5]}
    learning_rate=${PARAMS[6]}
    sequential_cv=${PARAMS[7]}
    
    # Add delay for sequential CV experiments to spread CPU load
    if [ "$sequential_cv" = "true" ]; then
        delay=$((delay + 5))  # Extra delay for sequential CV (more intensive)
    fi
    
    if [ "$delay" -gt 0 ]; then
        echo "  Waiting ${delay}s before starting $exp_name (load balancing)..."
        sleep $delay
    fi
    
    run_experiment "$session_name" "$exp_name" "$character_mode" "$num_epochs" \
                  "$seed" "$batch_size" "$learning_rate" "$sequential_cv"
    
    sleep 2  # Base delay between experiments
done

echo
echo "✅ All resumable/pending experiments started!"
echo
echo "Monitor progress:"
echo "  bash monitor_experiments.sh           # Detailed status"
echo "  watch -n 30 'bash quick_progress.sh'  # Auto-refresh progress"
echo "  tmux list-sessions                     # List tmux sessions"
echo
echo "💡 TIP: Set up a cron job to auto-resume after VM restarts:"
echo "  echo '@reboot cd $(pwd) && bash resume_experiments.sh' | crontab -"
echo
echo "Estimated completion time for remaining work:"
if [ ${#RESUMABLE_EXPERIMENTS[@]} -gt 0 ]; then
    echo "  - Resumable experiments: ~1-3 hours (depends on checkpoint progress)"
fi
if [ ${#NOT_STARTED_EXPERIMENTS[@]} -gt 0 ]; then
    sequential_count=$(printf '%s\n' "${NOT_STARTED_EXPERIMENTS[@]}" | grep -c "|true" || echo 0)
    parallel_count=$(printf '%s\n' "${NOT_STARTED_EXPERIMENTS[@]}" | grep -c "|false" || echo 0)
    
    echo "  - New sequential CV experiments ($sequential_count): ~3-6 hours each (5 fold training)"
    echo "  - New parallel CV experiments ($parallel_count): ~30-60 minutes each"
fi