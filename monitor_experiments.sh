#!/bin/bash

# Monitor CSI Character Embedding Experiments  
# Companion script to check experiment progress and results
#
# Usage: bash monitor_experiments.sh [quick]

set -e

# Check if quick mode requested
QUICK_MODE=false
if [ "$1" = "quick" ]; then
    QUICK_MODE=true
fi

if [ "$QUICK_MODE" = false ]; then
    echo "=== CSI Experiment Monitor ==="
    echo
fi

# Function to check if tmux session exists
session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

# Function to get session status
get_session_status() {
    local session=$1
    if session_exists "$session"; then
        # Check if there are any processes running
        local pane_count=$(tmux list-panes -t "$session" -F "#{pane_pid}" | wc -l)
        if [ "$pane_count" -gt 0 ]; then
            echo "RUNNING"
        else
            echo "IDLE"
        fi
    else
        echo "STOPPED"
    fi
}

# Function to get experiment completion status
get_experiment_status() {
    local exp_name=$1
    local exp_dir="experiments/$exp_name"
    local log_file="experiment_logs/${exp_name}.log"
    
    if [ -d "$exp_dir" ]; then
        if [ -f "$exp_dir/final_results.json" ]; then
            echo "COMPLETE"
        elif [ -f "$exp_dir/experiment_config.json" ]; then
            echo "IN_PROGRESS"
        else
            echo "STARTED"
        fi
    elif [ -f "$log_file" ]; then
        if grep -q "EXPERIMENT FAILED" "$log_file" 2>/dev/null; then
            echo "FAILED"
        else
            echo "STARTED"
        fi
    else
        echo "NOT_STARTED"
    fi
}

# Function to extract training progress from log
get_training_progress() {
    local exp_name=$1
    local log_file="experiment_logs/${exp_name}.log"
    
    if [ ! -f "$log_file" ]; then
        echo "N/A|N/A|N/A"
        return
    fi
    
    # Extract total steps and current progress
    local total_steps=$(grep "total steps" "$log_file" 2>/dev/null | tail -1 | sed 's/.*= //' | sed 's/ total.*//' || echo "")
    local current_step=$(grep -E "Step [0-9]+/" "$log_file" 2>/dev/null | tail -1 | sed 's/.*Step //' | sed 's/\/.*//' || echo "")
    local current_epoch=$(grep -E "Epoch [0-9]+/" "$log_file" 2>/dev/null | tail -1 | sed 's/.*Epoch //' | sed 's/\/.*//' || echo "")
    
    if [ -z "$total_steps" ] || [ -z "$current_step" ]; then
        echo "N/A|N/A|N/A"
        return
    fi
    
    # Calculate progress percentage
    local progress_pct=$((current_step * 100 / total_steps))
    
    # Calculate estimated time remaining
    local log_start_time=$(stat -f "%m" "$log_file" 2>/dev/null || echo "0")
    local current_time=$(date +%s)
    local elapsed=$((current_time - log_start_time))
    
    if [ "$elapsed" -gt 0 ] && [ "$current_step" -gt 0 ]; then
        local time_per_step=$((elapsed / current_step))
        local remaining_steps=$((total_steps - current_step))
        local eta_seconds=$((remaining_steps * time_per_step))
        local eta_min=$((eta_seconds / 60))
        echo "${progress_pct}|${current_epoch}|${eta_min}m"
    else
        echo "${progress_pct}|${current_epoch}|N/A"
    fi
}

# Function to get last log lines with progress info
get_last_log_lines() {
    local exp_name=$1
    local log_file="experiment_logs/${exp_name}.log"
    if [ -f "$log_file" ]; then
        # Get last few lines but prioritize progress lines
        local recent_progress=$(grep -E "(Step [0-9]+/|Epoch [0-9]+/|Loss:|Accuracy:)" "$log_file" 2>/dev/null | tail -2 | sed 's/^/    /')
        if [ ! -z "$recent_progress" ]; then
            echo "$recent_progress"
        else
            tail -2 "$log_file" 2>/dev/null | sed 's/^/    /' || echo "    (no recent activity)"
        fi
    else
        echo "    (no log file)"
    fi
}

# Show overall status
if [ "$QUICK_MODE" = false ]; then
    echo "📊 EXPERIMENT OVERVIEW"
    echo "======================"
    echo
fi

# Count sessions and experiments  
running_sessions=0
completed_experiments=0
failed_experiments=0
total_experiments=4

for i in {1..4}; do
    session_name=$(printf "exp%02d" $i)
    session_status=$(get_session_status "$session_name")
    if [ "$session_status" = "RUNNING" ]; then
        ((running_sessions++))
    fi
done

# Count completed/failed experiments
if [ -d "experiments" ]; then
    completed_experiments=$(find experiments/ -name "final_results.json" | wc -l)
fi

if [ -d "experiment_logs" ]; then
    failed_experiments=$(grep -l "EXPERIMENT FAILED" experiment_logs/*.log 2>/dev/null | wc -l || echo 0)
fi

if [ "$QUICK_MODE" = true ]; then
    echo "CSI Phase 1 Status: $completed_experiments/4 complete, $running_sessions active, $failed_experiments failed"
else
    echo "Active tmux sessions: $running_sessions"
    echo "Completed experiments: $completed_experiments/$total_experiments (Phase 1)"
    echo "Failed experiments: $failed_experiments"
fi

# Calculate overall progress
if [ -d "experiment_logs" ]; then
    total_progress=0
    progress_count=0
    for log_file in experiment_logs/*.log; do
        if [ -f "$log_file" ]; then
            exp_name=$(basename "$log_file" .log)
            progress_info=$(get_training_progress "$exp_name")
            IFS='|' read -ra PROGRESS <<< "$progress_info"
            progress_pct=${PROGRESS[0]}
            if [ "$progress_pct" != "N/A" ] && [ "$progress_pct" != "" ]; then
                total_progress=$((total_progress + progress_pct))
                progress_count=$((progress_count + 1))
            fi
        fi
    done
    
    if [ "$progress_count" -gt 0 ] && [ "$QUICK_MODE" = false ]; then
        avg_progress=$((total_progress / progress_count))
        echo "Average training progress: ${avg_progress}%"
    fi
fi

# Exit early in quick mode
if [ "$QUICK_MODE" = true ]; then
    exit 0
fi

echo

# Detailed status for each experiment
echo "🔍 DETAILED STATUS"
echo "=================="
printf "%-23s %-12s %-12s %-8s %-8s %-8s %-8s %s\n" "EXPERIMENT" "TMUX" "STATUS" "PROGRESS" "EPOCH" "ETA" "ELAPSED" "RECENT_ACTIVITY"
echo "$(printf '%*s' 130 '' | tr ' ' '-')"

# Define experiment names (Phase 1 - same as in run_experiments.sh)
declare -a EXP_NAMES=(
    "seq_cv_ep_iso_s42"
    "seq_cv_cross_ep_s42"
    "parallel_cv_ep_iso_s42" 
    "parallel_cv_cross_ep_s42"
)

for i in {0..3}; do
    session_name=$(printf "exp%02d" $((i+1)))
    exp_name=${EXP_NAMES[i]}
    session_status=$(get_session_status "$session_name")
    exp_status=$(get_experiment_status "$exp_name")
    
    # Get training progress
    progress_info=$(get_training_progress "$exp_name")
    IFS='|' read -ra PROGRESS <<< "$progress_info"
    progress_pct=${PROGRESS[0]}
    current_epoch=${PROGRESS[1]}
    eta=${PROGRESS[2]}
    
    # Calculate elapsed time
    elapsed="N/A"
    log_file="experiment_logs/${exp_name}.log"
    if [ -f "$log_file" ]; then
        if [ "$exp_status" = "COMPLETE" ]; then
            # Try to extract duration from log
            duration_line=$(grep "Duration:" "$log_file" 2>/dev/null | tail -1)
            if [ ! -z "$duration_line" ]; then
                elapsed=$(echo "$duration_line" | sed 's/.*Duration: //' | sed 's/ minutes/m/')
            fi
        elif [ "$exp_status" != "NOT_STARTED" ]; then
            # Calculate elapsed time
            start_time=$(stat -f "%m" "$log_file" 2>/dev/null || echo "0")
            current_time=$(date +%s)
            if [ "$start_time" != "0" ]; then
                elapsed_sec=$((current_time - start_time))
                elapsed_min=$((elapsed_sec / 60))
                elapsed="${elapsed_min}m"
            fi
        fi
    fi
    
    # Format status with emojis
    case $session_status in
        "RUNNING") session_display="🟢RUN" ;;
        "IDLE") session_display="🟡IDLE" ;;
        "STOPPED") session_display="🔴STOP" ;;
    esac
    
    case $exp_status in
        "COMPLETE") exp_display="✅DONE" ;;
        "IN_PROGRESS") exp_display="🔄TRAIN" ;;
        "STARTED") exp_display="🟡START" ;;
        "FAILED") exp_display="❌FAIL" ;;
        "NOT_STARTED") exp_display="⏸️WAIT" ;;
    esac
    
    # Format progress percentage
    if [ "$progress_pct" = "N/A" ]; then
        progress_display="N/A"
    else
        progress_display="${progress_pct}%"
    fi
    
    printf "%-23s %-12s %-12s %-8s %-8s %-8s %-8s " "$exp_name" "$session_display" "$exp_display" "$progress_display" "$current_epoch" "$eta" "$elapsed"
    
    # Show recent activity inline for active experiments
    if [[ "$session_status" = "RUNNING" && "$exp_status" = "IN_PROGRESS" ]]; then
        recent_line=$(grep -E "(Step [0-9]+/|Loss:|Accuracy:)" "$log_file" 2>/dev/null | tail -1 | cut -c1-40)
        echo "$recent_line"
    else
        echo
    fi
done

echo
echo "🛠️  MONITORING & MANAGEMENT COMMANDS"
echo "===================================="
echo "Monitor experiments:"
echo "  bash monitor_experiments.sh          # Detailed status (this script)"
echo "  bash quick_progress.sh               # Quick status check"  
echo "  watch -n 30 'bash quick_progress.sh' # Auto-refresh every 30s"
echo
echo "Resume after VM restart:"
echo "  bash resume_experiments.sh           # Auto-detect and resume incomplete experiments"
echo
echo "View specific experiments:"
echo "  tmux attach-session -t exp01         # Attach to experiment session"
echo "  tail -f experiment_logs/[exp_name].log  # Follow log file"
echo
echo "Check results:"
echo "  ls -la experiments/                   # List experiment directories"
echo "  cat experiments/[exp_name]/final_results.json  # View results"
echo
echo "System monitoring:"
echo "  htop                                  # System resources"
echo "  df -h                                 # Disk usage"
echo
echo "Emergency cleanup (if needed):"
echo "  tmux kill-session -t exp01            # Kill specific session"
echo "  pkill -f python                       # Kill all Python processes (⚠️ CAREFUL!)"
echo
echo "💡 VM LEASE MANAGEMENT (Phase 1):"
echo "  - Sequential CV experiments: ~3-6 hours (may need resuming)"
echo "  - Parallel CV experiments: ~30-60 minutes (safe for 8h lease)"
echo "  - Use 'bash resume_experiments.sh' after VM restart"

# Show disk usage and persistence info
if [ -d "experiments" ]; then
    echo
    echo "💾 DISK USAGE & PERSISTENCE"
    echo "============================"
    echo "Experiment directories (with saved models/datasets):"
    for exp_dir in experiments/*/; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            size=$(du -sh "$exp_dir" 2>/dev/null | cut -f1)
            
            # Check what's saved
            saved_items=""
            [ -f "$exp_dir/final.pt" ] && saved_items="${saved_items}model "
            [ -f "$exp_dir/train_dataset.pkl.gz" ] && saved_items="${saved_items}dataset "
            [ -f "$exp_dir/final_character_embeddings.pt" ] && saved_items="${saved_items}embeddings "
            [ -f "$exp_dir/final_results.json" ] && saved_items="${saved_items}results "
            
            printf "  %-25s %8s  [%s]\n" "$exp_name" "$size" "${saved_items:-incomplete}"
        fi
    done 2>/dev/null | head -10
    
    echo
    echo "Log files:"
    du -sh experiment_logs/ 2>/dev/null || echo "No log files yet"
    
    # Total experiment storage
    total_exp_size=$(du -sh experiments/ 2>/dev/null | cut -f1)
    echo "Total experiments storage: $total_exp_size"
fi

echo
echo "Last updated: $(date)"