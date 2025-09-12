"""
Neural network architecture for 'Did I Say This' character embedding learning.

Architecture: BERT (frozen) + Character Embeddings + Binary Classifier
- Input: Sentence text + Character ID
- BERT processes sentence → 768-dim vector
- Character embedding lookup → 768-dim vector  
- Fusion: Concatenate + Optional projection
- Output: Binary classification (did character say this?)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
import logging

logger = logging.getLogger(__name__)


@dataclass
class DidISayThisConfig:
    """Configuration for the DidISayThis model architecture."""
    
    # Model architecture
    bert_model_name: str = 'bert-base-uncased'
    character_vocab_size: int = 647  # Will be set based on dataset
    character_embedding_dim: int = 768  # Match BERT dimensionality
    freeze_bert: bool = True
    
    # Fusion and classification
    use_projection_layer: bool = True
    projection_dim: Optional[int] = 512  # None to skip projection
    dropout_rate: float = 0.1
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    
    # Character embedding initialization
    character_embedding_init_std: float = 0.02
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'bert_model_name': self.bert_model_name,
            'character_vocab_size': self.character_vocab_size,
            'character_embedding_dim': self.character_embedding_dim,
            'freeze_bert': self.freeze_bert,
            'use_projection_layer': self.use_projection_layer,
            'projection_dim': self.projection_dim,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'character_embedding_init_std': self.character_embedding_init_std
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DidISayThisConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class DidISayThisModel(nn.Module):
    """
    'Did I Say This' model for character embedding learning.
    
    Architecture:
    1. BERT encodes input sentence → [batch, 768]
    2. Character embedding lookup → [batch, 768]  
    3. Concatenate → [batch, 1536]
    4. Optional projection → [batch, projection_dim]
    5. ReLU + Dropout
    6. Binary classifier → [batch, 1]
    """
    
    def __init__(self, config: DidISayThisConfig):
        super().__init__()
        self.config = config
        
        # Load BERT model
        logger.info(f"Loading BERT model: {config.bert_model_name}")
        self.bert = AutoModel.from_pretrained(config.bert_model_name)
        self.bert_config = AutoConfig.from_pretrained(config.bert_model_name)
        
        # Freeze BERT if specified
        if config.freeze_bert:
            logger.info("Freezing BERT parameters")
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Character embeddings
        logger.info(f"Creating character embeddings: {config.character_vocab_size} × {config.character_embedding_dim}")
        self.character_embeddings = nn.Embedding(
            config.character_vocab_size,
            config.character_embedding_dim,
            padding_idx=None  # No padding for character vocab
        )
        
        # Initialize character embeddings with small random values
        nn.init.normal_(self.character_embeddings.weight, 
                       mean=0.0, std=config.character_embedding_init_std)
        
        # Fusion layers
        bert_dim = self.bert_config.hidden_size  # Should be 768 for bert-base
        fusion_input_dim = bert_dim + config.character_embedding_dim
        
        if config.use_projection_layer and config.projection_dim:
            logger.info(f"Using projection layer: {fusion_input_dim} → {config.projection_dim}")
            self.projection = nn.Linear(fusion_input_dim, config.projection_dim)
            classifier_input_dim = config.projection_dim
        else:
            logger.info("No projection layer, using direct concatenation")
            self.projection = None
            classifier_input_dim = fusion_input_dim
        
        # Classification layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(classifier_input_dim, 1)
        
        logger.info(f"Model architecture complete:")
        logger.info(f"  BERT dim: {bert_dim}")
        logger.info(f"  Character embedding dim: {config.character_embedding_dim}")
        logger.info(f"  Fusion dim: {fusion_input_dim}")
        logger.info(f"  Classifier input dim: {classifier_input_dim}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                character_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: BERT input token IDs [batch, seq_len]
            attention_mask: BERT attention mask [batch, seq_len]
            character_ids: Character IDs [batch]
            
        Returns:
            Logits for binary classification [batch, 1]
        """
        batch_size = input_ids.size(0)
        
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        sentence_embeddings = bert_outputs.pooler_output  # [batch, 768]
        
        # Character embeddings
        character_embeddings = self.character_embeddings(character_ids)  # [batch, 768]
        
        # Fusion: concatenate sentence and character embeddings
        fused_embeddings = torch.cat([sentence_embeddings, character_embeddings], dim=1)  # [batch, 1536]
        
        # Optional projection
        if self.projection is not None:
            fused_embeddings = self.projection(fused_embeddings)  # [batch, projection_dim]
        
        # ReLU activation and dropout
        hidden = torch.relu(fused_embeddings)
        hidden = self.dropout(hidden)
        
        # Binary classification
        logits = self.classifier(hidden)  # [batch, 1]
        
        return logits
    
    def get_character_embeddings(self) -> torch.Tensor:
        """
        Get the learned character embedding matrix.
        
        Returns:
            Character embedding matrix [vocab_size, embedding_dim]
        """
        return self.character_embeddings.weight.data.clone()
    
    def get_character_embedding(self, character_id: int) -> torch.Tensor:
        """
        Get embedding for a specific character.
        
        Args:
            character_id: Character ID to get embedding for
            
        Returns:
            Character embedding vector [embedding_dim]
        """
        return self.character_embeddings.weight[character_id].data.clone()
    
    def save_model(self, save_path: str, dataset_info: Optional[Dict] = None) -> None:
        """
        Save model state with configuration and optional dataset info.
        
        Args:
            save_path: Path to save model to
            dataset_info: Optional dataset information for reproducibility
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'bert_config': self.bert_config.to_dict(),
            'dataset_info': dataset_info,
            'model_class': self.__class__.__name__
        }
        
        torch.save(save_dict, save_path)
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, strict: bool = True) -> 'DidISayThisModel':
        """
        Load model from saved state.
        
        Args:
            load_path: Path to load model from
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Loaded DidISayThisModel instance
        """
        save_dict = torch.load(load_path, map_location='cpu')
        
        # Create config
        config = DidISayThisConfig.from_dict(save_dict['config'])
        
        # Create model
        model = cls(config)
        
        # Load state dict
        model.load_state_dict(save_dict['model_state_dict'], strict=strict)
        
        logger.info(f"Model loaded from {load_path}")
        logger.info(f"Dataset info: {save_dict.get('dataset_info', 'None')}")
        
        return model
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters by component.
        
        Returns:
            Dictionary with parameter counts
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'bert_encoder': count_params(self.bert) if not self.config.freeze_bert else 0,
            'character_embeddings': count_params(self.character_embeddings),
            'projection': count_params(self.projection) if self.projection else 0,
            'classifier': count_params(self.classifier),
            'total_trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'total_parameters': sum(p.numel() for p in self.parameters())
        }