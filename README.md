# CSI Character Embedding Learning System

A complete PyTorch implementation for learning character embeddings from CSI television episode transcripts using the "Did I Say This" proxy task.

## Overview

This system learns character embeddings by training a binary classifier to predict whether a character said a given sentence. The learned embeddings capture speaking patterns and character distinctions within and across episodes.

**Key Features:**
- ✅ Safe multi-epoch training with killer reveal holdout
- ✅ Complete experiment reproducibility and tracking  
- ✅ Comprehensive configuration options
- ✅ Automatic checkpointing and resumable training
- ✅ Character embedding extraction for analysis

## Quick Start

```python
# Basic single-epoch training (safest)
python scratch/example_training.py

# Or import and use directly
from pathlib import Path
from data.dataset import DidISayThisDataset
from model.experiment import ExperimentManager, ExperimentConfig
from model.architecture import DidISayThisConfig  
from model.trainer import TrainingConfig

# Create dataset
dataset = DidISayThisDataset.from_data_directory(
    Path("data/original"),
    character_mode='episode-isolated',
    seed=42
)

# Set up experiment
experiment = ExperimentManager(
    experiment_config=ExperimentConfig(experiment_name="my_experiment"),
    model_config=DidISayThisConfig(character_vocab_size=dataset.get_character_vocabulary_size()),
    training_config=TrainingConfig(num_epochs=1)
)

# Train model
model = experiment.setup_experiment(dataset, None)
results = experiment.train_model()
```

## Architecture

```
Input: (Sentence Text, Character ID) → Binary Classification (Did character say this?)

BERT-base (frozen) → [CLS] Token (768-dim)
                           ↓
Character Embedding Lookup → (768-dim)
                           ↓
                    Concatenate → (1536-dim)
                           ↓
              Optional Projection → (configurable-dim)
                           ↓ 
                    ReLU + Dropout
                           ↓
              Binary Classifier → (1-dim logit)
```

## Safe Training Configurations

### Single-Epoch Training (Recommended)
```python
# Safe default - no killer reveal contamination
TrainingConfig(
    num_epochs=1,
    holdout_killer_reveal=False  # Not needed for single epoch
)
```

### Multi-Epoch Training (Advanced)
```python
# Safe multi-epoch with killer reveal protection
TrainingConfig(
    num_epochs=3,
    holdout_killer_reveal=True,
    killer_reveal_holdout_percentage=0.1  # Hold out last 10%
)

# Dataset must match
DidISayThisDataset.from_data_directory(
    data_directory,
    holdout_killer_reveal=True,
    killer_reveal_holdout_percentage=0.1
)
```

⚠️ **WARNING**: Multi-epoch training without killer reveal holdout risks contaminating character embeddings with killer information!

## Installation

```bash
# Clone repository
git clone <repo-url>
cd csi-diss-25

# Create virtual environment  
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers numpy pandas tqdm

# Verify installation
python -c "from src.data.dataset import DidISayThisDataset; print('✅ Installation successful!')"
```

## Configuration Reference

### 📊 Dataset Configuration (`DidISayThisDataset`)

| Parameter | Type | Default | Description | Example Values |
|-----------|------|---------|-------------|----------------|
| `data_directory` | Path | **Required** | Directory containing episode TSV files | `Path("data/original")` |
| `character_mode` | str | `'episode-isolated'` | Character identity scope | `'episode-isolated'`, `'cross-episode'` |
| `holdout_killer_reveal` | bool | `False` | Enable killer reveal holdout | `True`, `False` |
| `killer_reveal_holdout_percentage` | float | `0.1` | Fraction of episode endings to remove | `0.05`, `0.1`, `0.15`, `0.2` |
| `negative_ratio` | float | `1.0` | Ratio of negative to positive examples | `0.5`, `1.0`, `2.0` |
| `tokenizer_name` | str | `'bert-base-uncased'` | HuggingFace tokenizer | `'bert-base-cased'`, `'distilbert-base-uncased'` |
| `max_length` | int | `512` | Maximum sequence length | `128`, `256`, `512` |
| `seed` | int | `None` | Random seed for reproducibility | `42`, `123`, `999` |

**Character Mode Comparison:**
```python
# Episode-isolated: Characters are episode-specific
# Example: 's01e07:grissom', 's02e01:grissom' (separate embeddings)
dataset = DidISayThisDataset.from_data_directory(
    "data/original", character_mode='episode-isolated'  # ~1,102 characters
)

# Cross-episode: Characters consolidated across episodes  
# Example: 'grissom' (single embedding across all episodes)
dataset = DidISayThisDataset.from_data_directory(
    "data/original", character_mode='cross-episode'  # ~646 characters (41% reduction)
)
```

### 🏗️ Model Architecture Configuration (`DidISayThisConfig`)

| Parameter | Type | Default | Description | Example Values |
|-----------|------|---------|-------------|----------------|
| `bert_model_name` | str | `'bert-base-uncased'` | BERT model variant | `'bert-base-cased'`, `'distilbert-base-uncased'` |
| `character_vocab_size` | int | **Required** | Number of unique characters | `646`, `1102` (from dataset) |
| `character_embedding_dim` | int | `768` | Character embedding dimension | `256`, `512`, `768` |
| `freeze_bert` | bool | `True` | Whether to freeze BERT weights | `True` (recommended), `False` |
| `use_projection_layer` | bool | `True` | Add projection after concatenation | `True`, `False` |
| `projection_dim` | int | `512` | Projection layer output dimension | `256`, `512`, `1024` |
| `dropout_rate` | float | `0.1` | Dropout rate for regularization | `0.0`, `0.1`, `0.2`, `0.5` |

**Architecture Variants:**
```python
# With Projection Layer (Default - Better Performance)
config = DidISayThisConfig(
    character_vocab_size=dataset.get_character_vocabulary_size(),
    use_projection_layer=True,    # BERT + Char → 1536-dim → 512-dim → ReLU → 1-dim
    projection_dim=512           # ~864K trainable parameters
)

# Without Projection Layer (Simpler - Faster Training)  
config = DidISayThisConfig(
    character_vocab_size=dataset.get_character_vocabulary_size(),
    use_projection_layer=False   # BERT + Char → 1536-dim → 1-dim  
)                               # ~77K trainable parameters (much smaller)
```

### 🎯 Training Configuration (`TrainingConfig`)

| Parameter | Type | Default | Description | Example Values |
|-----------|------|---------|-------------|----------------|
| `num_epochs` | int | `1` | Number of training epochs | `1` (safe), `3`, `5`, `10` |
| `batch_size` | int | `32` | Training batch size | `8`, `16`, `32`, `64` |
| `learning_rate` | float | `1e-4` | AdamW learning rate | `1e-5`, `1e-4`, `3e-4`, `1e-3` |
| `weight_decay` | float | `1e-5` | L2 regularization strength | `0.0`, `1e-5`, `1e-4` |
| `gradient_clip_norm` | float | `1.0` | Gradient clipping threshold | `0.5`, `1.0`, `2.0` |
| `warmup_steps` | int | `0` | Learning rate warmup steps | `0`, `100`, `500` |
| `use_scheduler` | bool | `False` | Enable cosine annealing LR | `True`, `False` |
| `holdout_killer_reveal` | bool | `False` | Enable killer reveal holdout | `True` (multi-epoch), `False` |
| `killer_reveal_holdout_percentage` | float | `0.1` | Percentage to hold out | `0.05`, `0.1`, `0.15`, `0.2` |

**Training Mode Examples:**
```python
# Single-Epoch Training (Safest - Recommended)
training_config = TrainingConfig(
    num_epochs=1,
    batch_size=32,
    learning_rate=1e-4,
    holdout_killer_reveal=False  # Not needed for single epoch
)

# Multi-Epoch Training with Killer Reveal Protection
training_config = TrainingConfig(
    num_epochs=5,
    batch_size=16,              # Smaller batch for longer training
    learning_rate=1e-4, 
    holdout_killer_reveal=True,  # CRITICAL: Prevent contamination
    killer_reveal_holdout_percentage=0.1  # Remove last 10% of episodes
)

# CPU-Optimized Training  
training_config = TrainingConfig(
    num_epochs=1,
    batch_size=8,               # Smaller batch for CPU
    learning_rate=3e-4,         # Slightly higher LR
    gradient_clip_norm=0.5      # Tighter clipping
)
```

### 🧪 Experiment Configuration (`ExperimentConfig`)

| Parameter | Type | Default | Description | Example Values |
|-----------|------|---------|-------------|----------------|
| `experiment_name` | str | **Required** | Unique experiment identifier | `'baseline_s42'`, `'cross_ep_5epoch'` |
| `description` | str | `''` | Human-readable description | `'Baseline episode-isolated training'` |
| `tags` | List[str] | `[]` | Experiment tags for organization | `['baseline']`, `['cross-episode', 'multi-epoch']` |
| `save_datasets` | bool | `True` | Save datasets with experiment | `True` (recommended), `False` |
| `save_model_checkpoints` | bool | `True` | Save model checkpoints | `True`, `False` |

### 🔧 Complete Configuration Examples

#### Minimal Configuration (Fastest)
```python
# Smallest, fastest configuration
dataset = DidISayThisDataset.from_data_directory(
    "data/original",
    character_mode='cross-episode',    # Fewer characters (646 vs 1102)
    max_length=128                     # Shorter sequences
)

config = DidISayThisConfig(
    character_vocab_size=dataset.get_character_vocabulary_size(),
    use_projection_layer=False,       # Simpler architecture (~77K params)
    character_embedding_dim=256       # Smaller embeddings
)

training = TrainingConfig(
    num_epochs=1,
    batch_size=64,                    # Larger batches
    learning_rate=3e-4               # Higher learning rate
)
```

#### Production Configuration (Best Performance)
```python
# High-quality configuration for best results
dataset = DidISayThisDataset.from_data_directory(
    "data/original", 
    character_mode='episode-isolated', # More detailed character representations
    holdout_killer_reveal=True,        # Safe multi-epoch training
    killer_reveal_holdout_percentage=0.1,
    negative_ratio=1.0,               # Balanced examples
    seed=42                           # Reproducible
)

config = DidISayThisConfig(
    character_vocab_size=dataset.get_character_vocabulary_size(),
    use_projection_layer=True,        # Better feature learning (~864K params)
    projection_dim=512,
    dropout_rate=0.1                  # Regularization
)

training = TrainingConfig(
    num_epochs=3,                     # More training
    batch_size=16,                    # Good balance
    learning_rate=1e-4,               # Conservative
    holdout_killer_reveal=True,       # Match dataset
    killer_reveal_holdout_percentage=0.1,
    use_scheduler=True,               # Learning rate scheduling
    warmup_steps=100
)
```

#### Experimental Comparison Configuration
```python
# Configuration for systematic comparison (see EXPERIMENTAL_DESIGN.md)
configs = {
    'character_modes': ['episode-isolated', 'cross-episode'],
    'epochs': [1, 5],
    'projection': [True, False], 
    'seeds': [42, 123]
}

# This creates 2×2×2×2 = 16 experiments
# See run_experiments.sh for automated execution
```

### ⚠️ Important Notes

**Killer Reveal Contamination:**
- Multi-epoch training WITHOUT holdout risks learning killer identity patterns
- ALWAYS use `holdout_killer_reveal=True` for `num_epochs > 1`
- Dataset and training config holdout settings must match

**Performance Considerations:**
- **CPU Training**: Use smaller batch sizes (8-16), simpler architectures
- **GPU Training**: Can handle larger batches (32-64), complex architectures  
- **Memory Usage**: Scales with vocabulary size and sequence length
- **Training Time**: Episode-isolated mode takes longer than cross-episode

**Character Mode Trade-offs:**
- **Episode-isolated**: More detailed, episode-specific character patterns (~1,102 characters)
- **Cross-episode**: Consolidated character representations, faster training (~646 characters)