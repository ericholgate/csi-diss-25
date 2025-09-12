# VM Lease Management Workflow

This guide helps you manage experiments across multiple 8-hour VM lease sessions.

## Quick Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `run_experiments.sh` | Start all 16 experiments fresh | First VM session only |
| `resume_experiments.sh` | Resume incomplete experiments | After each VM restart |
| `quick_progress.sh` | Check progress | Monitor experiments |
| `monitor_experiments.sh` | Detailed status | Deep analysis |

## First VM Session (Fresh Start)

```bash
# 1. Set up environment
cd csi-diss-25
source venv/bin/activate

# 2. Start all experiments  
bash run_experiments.sh

# 3. Monitor progress
watch -n 30 'bash quick_progress.sh'

# Expected completion in first 8h session:
# ✅ All 8 single-epoch experiments (1-2h each)
# 🔄 Multi-epoch experiments will be partially complete
```

## Subsequent VM Sessions (Resume)

```bash
# 1. Set up environment
cd csi-diss-25  
source venv/bin/activate

# 2. Resume incomplete experiments
bash resume_experiments.sh

# 3. Monitor progress
watch -n 30 'bash quick_progress.sh'

# Expected: 
# ✅ Previously completed experiments skipped automatically
# 🔄 Incomplete experiments resume from last checkpoint
```

## Experiment Timeline (CPU Estimates)

### Single-Epoch Experiments (8 total)
- **Runtime**: 45-90 minutes each
- **VM Sessions**: All complete in first session
- **Status**: ✅ Safe for 8h lease

### Multi-Epoch Experiments (8 total) 
- **Runtime**: 3-6 hours each
- **VM Sessions**: 2-3 sessions needed per experiment
- **Checkpoints**: Every 500 steps (automatic)
- **Resume**: Seamless from last checkpoint

## Monitoring Commands

```bash
# Quick status (recommended)
bash quick_progress.sh

# Detailed analysis
bash monitor_experiments.sh

# Auto-refresh monitoring
watch -n 30 'bash quick_progress.sh'

# Check specific experiment
tmux attach-session -t exp05    # Detach with Ctrl+B, then D
```

## Resume Logic

The resume system automatically:

1. **✅ Skips completed experiments** (has `final_results.json`)
2. **🔄 Resumes from checkpoints** (has `*.pt` files)  
3. **🆕 Restarts failed experiments** (no valid checkpoints)
4. **🗑️ Cleans up corrupted directories**

## Expected VM Session Breakdown

### Session 1 (8 hours)
- ✅ Complete: All 8 single-epoch experiments
- 🔄 Partial: All 8 multi-epoch experiments (~60-80% done)

### Session 2 (8 hours)  
- ✅ Complete: 6-8 multi-epoch experiments
- 🔄 Partial: 0-2 remaining multi-epoch experiments

### Session 3 (4-6 hours)
- ✅ Complete: All remaining experiments
- 🎉 **Total: 16/16 experiments finished**

## Storage Requirements

Each experiment generates ~1.2GB of data:
- **Model checkpoints**: ~300MB
- **Saved datasets**: ~700MB (compressed)
- **Character embeddings**: ~50MB
- **Logs and metadata**: ~50MB

**Total expected storage**: ~20GB for all 16 experiments

## Troubleshooting

### If experiment fails to resume:
```bash
# Check experiment status
ls -la experiments/[exp_name]/

# Look for these files:
# - experiment_config.json (required for resume)
# - *.pt files (checkpoints)
# - final_results.json (completed marker)

# If corrupted, clean up and restart:
rm -rf experiments/[exp_name]
bash resume_experiments.sh
```

### If VM runs out of space:
```bash
# Check disk usage
df -h
du -sh experiments/*/

# Clean up logs (optional)
rm -rf experiment_logs/
```

### If too many experiments running:
```bash
# Kill specific session
tmux kill-session -t exp01

# Kill all experiments (emergency)
for i in {1..16}; do tmux kill-session -t $(printf "exp%02d" $i) 2>/dev/null || true; done
```

## Success Indicators

### Experiment Complete:
- ✅ `final_results.json` exists
- ✅ Log shows "Training completed successfully!"
- ✅ Directory contains: model, dataset, embeddings, results

### Ready for Analysis:
- ✅ All 16 experiments show "COMPLETE" status
- ✅ Total storage ~20GB
- ✅ No active tmux sessions (`tmux list-sessions`)

## Next Steps After All Experiments Complete

1. **Download results** from VM to local machine
2. **Run analysis notebooks** (to be created)
3. **Compare character embeddings** across configurations
4. **Generate visualizations** and performance comparisons

---

**💡 Pro Tips:**
- Use `screen -S monitor` to keep monitoring session persistent
- Set up alerts: `bash quick_progress.sh | grep "16/16 done" && echo "ALL DONE!" | mail you@domain.com`
- Keep the VM workflow simple: start → monitor → resume → repeat