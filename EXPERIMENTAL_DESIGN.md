# Experimental Design: CSI Character Embedding Learning

## Overview

This document describes the experimental configurations for training character embeddings using the "Did I Say This" proxy task on CSI television episode transcripts.

## Research Questions

1. **Character Mode Impact**: How do episode-isolated vs cross-episode character representations affect embedding quality?
2. **Killer Reveal Contamination**: Does holdout of episode endings prevent learning killer identity patterns?
3. **Architecture Impact**: Do projection layers improve character embedding learning vs direct concatenation?
4. **Training Duration**: How does single-epoch vs multi-epoch training affect performance?

## Experimental Design Matrix

### Configuration Space
- **Character Modes**: 2 (episode-isolated, cross-episode)
- **Training Epochs**: 2 (1 epoch, 5 epochs)
- **Architecture Variants**: 2 (with projection, without projection)  
- **Random Seeds**: 2 (42, 123)
- **Total Experiments**: 2 × 2 × 2 × 2 = **16 experiments**

### Experiment Configurations

| ID | Name | Character Mode | Epochs | Killer Holdout | Projection | Seed | Description |
|----|------|---------------|---------|----------------|------------|------|-------------|
| 01 | `ep_iso_1ep_proj_s42` | episode-isolated | 1 | No | Yes (512-dim) | 42 | Baseline: single epoch, episode-specific chars |
| 02 | `ep_iso_1ep_proj_s123` | episode-isolated | 1 | No | Yes (512-dim) | 123 | Replication seed for baseline |
| 03 | `ep_iso_1ep_noproj_s42` | episode-isolated | 1 | No | No | 42 | Test architecture: no projection layer |
| 04 | `ep_iso_1ep_noproj_s123` | episode-isolated | 1 | No | No | 123 | Replication seed for no-projection |
| 05 | `ep_iso_5ep_proj_s42` | episode-isolated | 5 | Yes (10%) | Yes (512-dim) | 42 | Multi-epoch with killer reveal holdout |
| 06 | `ep_iso_5ep_proj_s123` | episode-isolated | 5 | Yes (10%) | Yes (512-dim) | 123 | Replication seed for multi-epoch |
| 07 | `ep_iso_5ep_noproj_s42` | episode-isolated | 5 | Yes (10%) | No | 42 | Multi-epoch without projection |
| 08 | `ep_iso_5ep_noproj_s123` | episode-isolated | 5 | Yes (10%) | No | 123 | Replication seed |
| 09 | `cross_ep_1ep_proj_s42` | cross-episode | 1 | No | Yes (512-dim) | 42 | Cross-episode character consolidation |
| 10 | `cross_ep_1ep_proj_s123` | cross-episode | 1 | No | Yes (512-dim) | 123 | Replication seed |
| 11 | `cross_ep_1ep_noproj_s42` | cross-episode | 1 | No | No | 42 | Cross-episode without projection |
| 12 | `cross_ep_1ep_noproj_s123` | cross-episode | 1 | No | No | 123 | Replication seed |
| 13 | `cross_ep_5ep_proj_s42` | cross-episode | 5 | Yes (10%) | Yes (512-dim) | 42 | Cross-episode multi-epoch training |
| 14 | `cross_ep_5ep_proj_s123` | cross-episode | 5 | Yes (10%) | Yes (512-dim) | 123 | Replication seed |
| 15 | `cross_ep_5ep_noproj_s42` | cross-episode | 5 | Yes (10%) | No | 42 | Cross-episode multi-epoch no-projection |
| 16 | `cross_ep_5ep_noproj_s123` | cross-episode | 5 | Yes (10%) | No | 123 | Replication seed |

## Model Architecture Details

### Base Configuration
- **Sentence Encoder**: BERT-base-uncased (frozen, 768-dim output)
- **Character Embedding**: Learnable embeddings (768-dim)
- **Classifier**: Binary classification (character spoke sentence vs didn't)
- **Loss Function**: Binary cross-entropy with logits
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)

### Architecture Variants

#### With Projection (use_projection=True)
```
BERT(sentence) → 768-dim
Character Embedding → 768-dim
Concat → 1536-dim → Linear(512) → ReLU → Dropout(0.1) → Linear(1) → BCE
```
**Trainable Parameters**: ~864K

#### Without Projection (use_projection=False)  
```
BERT(sentence) → 768-dim
Character Embedding → 768-dim  
Concat → 1536-dim → Linear(1) → BCE
```
**Trainable Parameters**: ~77K (much smaller)

## Training Configuration

### Hyperparameters
- **Batch Size**: 16 (optimized for CPU training)
- **Learning Rate**: 1e-4 (conservative for stable training)
- **Weight Decay**: 1e-5
- **Gradient Clipping**: 1.0
- **Sequence Length**: 512 tokens (BERT limit)

### Training Strategy
- **No shuffling**: Maintain temporal/chronological order
- **Balanced sampling**: 50% positive, 50% negative examples per batch
- **Killer Reveal Holdout**: Remove last 10% of each episode (multi-epoch only)
- **Checkpointing**: Every 500 steps + best validation model

### Data Characteristics
- **Episodes**: 39 CSI episodes (seasons 1-5)
- **Total Sentences**: 26,188
- **Character Modes**:
  - Episode-isolated: 1,102 unique characters
  - Cross-episode: 646 unique characters (41% reduction)
- **Examples**: ~52K total (26K positive, 26K negative)

## Expected Outcomes

### Hypotheses
1. **Cross-episode mode** should learn better character representations by consolidating speaking patterns
2. **Killer reveal holdout** should prevent contamination from episode endings revealing killer identity  
3. **Projection layers** should improve feature learning vs direct concatenation
4. **Multi-epoch training** should improve performance but requires holdout to prevent contamination

### Success Metrics
- **Training Accuracy**: Proxy for character embedding quality
- **Convergence Speed**: Fewer epochs to reach target performance  
- **Character Separation**: t-SNE visualization of learned embeddings
- **Generalization**: Performance consistency across random seeds

### Failure Modes
- **Overfitting to episode structure** instead of character patterns
- **Killer identity leakage** in multi-epoch training without holdout
- **Poor character separation** in learned embedding space
- **High variance across random seeds** indicating unstable training

## Resource Requirements

- **Runtime**: ~2-4 hours per experiment (CPU)
- **Storage**: ~50-100MB per completed experiment
- **Memory**: ~4-8GB RAM (depends on dataset size)
- **Total Expected Time**: 32-64 hours for all 16 experiments

## Analysis Plan

Post-training analysis will compare:
1. Training curves across configurations
2. Final accuracies and convergence rates  
3. Embedding visualizations (t-SNE/PCA)
4. Character clustering quality
5. Statistical significance across seed pairs