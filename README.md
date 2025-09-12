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