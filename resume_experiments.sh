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
for i in {1..16}; do
    session_name=$(printf "exp%02d" $i)
    tmux kill-session -t "$session_name" 2>/dev/null || true
done

# Check experiment status and collect resumable experiments
declare -a RESUMABLE_EXPERIMENTS=()
declare -a COMPLETED_EXPERIMENTS=()
declare -a NOT_STARTED_EXPERIMENTS=()

# Define experiment configurations (same as run_experiments.sh)
declare -a ALL_EXPERIMENTS=(
    "exp01|ep_iso_1ep_proj_s42|episode-isolated|False|0.1|1|True|512|42|16|1e-4"
    "exp02|ep_iso_1ep_proj_s123|episode-isolated|False|0.1|1|True|512|123|16|1e-4"
    "exp03|ep_iso_1ep_noproj_s42|episode-isolated|False|0.1|1|False|512|42|16|1e-4"
    "exp04|ep_iso_1ep_noproj_s123|episode-isolated|False|0.1|1|False|512|123|16|1e-4"
    "exp05|ep_iso_5ep_proj_s42|episode-isolated|True|0.1|5|True|512|42|16|1e-4"
    "exp06|ep_iso_5ep_proj_s123|episode-isolated|True|0.1|5|True|512|123|16|1e-4"
    "exp07|ep_iso_5ep_noproj_s42|episode-isolated|True|0.1|5|False|512|42|16|1e-4"
    "exp08|ep_iso_5ep_noproj_s123|episode-isolated|True|0.1|5|False|512|123|16|1e-4"
    "exp09|cross_ep_1ep_proj_s42|cross-episode|False|0.1|1|True|512|42|16|1e-4"
    "exp10|cross_ep_1ep_proj_s123|cross-episode|False|0.1|1|True|512|123|16|1e-4"
    "exp11|cross_ep_1ep_noproj_s42|cross-episode|False|0.1|1|False|512|42|16|1e-4"
    "exp12|cross_ep_1ep_noproj_s123|cross-episode|False|0.1|1|False|512|123|16|1e-4"
    "exp13|cross_ep_5ep_proj_s42|cross-episode|True|0.1|5|True|512|42|16|1e-4"
    "exp14|cross_ep_5ep_proj_s123|cross-episode|True|0.1|5|True|512|123|16|1e-4"
    "exp15|cross_ep_5ep_noproj_s42|cross-episode|True|0.1|5|False|512|42|16|1e-4"
    "exp16|cross_ep_5ep_noproj_s123|cross-episode|True|0.1|5|False|512|123|16|1e-4"
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
echo "Completed: ${#COMPLETED_EXPERIMENTS[@]}/16"
echo "Resumable: ${#RESUMABLE_EXPERIMENTS[@]}/16"
echo "Not Started: ${#NOT_STARTED_EXPERIMENTS[@]}/16"

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
    holdout_killer=${PARAMS[3]}
    holdout_pct=${PARAMS[4]}
    num_epochs=${PARAMS[5]}
    use_projection=${PARAMS[6]}
    projection_dim=${PARAMS[7]}
    seed=${PARAMS[8]}
    batch_size=${PARAMS[9]}
    learning_rate=${PARAMS[10]}
    
    # Add delay for 5-epoch experiments to spread CPU load
    if [ "$num_epochs" -eq 5 ]; then
        delay=$((delay + 10))  # 10 second additional delay for multi-epoch
    fi
    
    if [ "$delay" -gt 0 ]; then
        echo "  Waiting ${delay}s before starting $exp_name (load balancing)..."
        sleep $delay
    fi
    
    run_experiment "$session_name" "$exp_name" "$character_mode" "$holdout_killer" \
                  "$holdout_pct" "$num_epochs" "$use_projection" "$projection_dim" \
                  "$seed" "$batch_size" "$learning_rate"
    
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
    single_epoch_count=$(printf '%s\n' "${NOT_STARTED_EXPERIMENTS[@]}" | grep -c "|1|" || echo 0)
    multi_epoch_count=$(printf '%s\n' "${NOT_STARTED_EXPERIMENTS[@]}" | grep -c "|5|" || echo 0)
    
    echo "  - New single-epoch experiments ($single_epoch_count): ~1-2 hours each"
    echo "  - New multi-epoch experiments ($multi_epoch_count): ~4-6 hours each"
fi