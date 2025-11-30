"""
Embedding augmentation using vector addition for data enrichment
"""

import numpy as np
from typing import List
import config


def add_gaussian_noise(embedding: np.ndarray, scale: float = None) -> np.ndarray:
    """
    Add Gaussian noise to embedding for augmentation
    
    Args:
        embedding: Original embedding
        scale: Noise standard deviation
        
    Returns:
        Augmented embedding (L2 normalized)
    """
    if scale is None:
        scale = config.AUGMENTATION_NOISE_SCALE
    
    noise = np.random.normal(0, scale, embedding.shape)
    augmented = embedding + noise
    
    # L2 normalize
    augmented = augmented / (np.linalg.norm(augmented) + 1e-8)
    
    return augmented


def generate_synthetic_embeddings(embedding: np.ndarray, 
                                  num_samples: int = None) -> List[np.ndarray]:
    """
    Generate synthetic embeddings by adding noise
    
    Args:
        embedding: Original embedding
        num_samples: Number of synthetic samples
        
    Returns:
        List of augmented embeddings
    """
    if num_samples is None:
        num_samples = config.AUGMENTATION_SAMPLES_PER_IMAGE
    
    synthetic = []
    for _ in range(num_samples):
        aug_emb = add_gaussian_noise(embedding)
        synthetic.append(aug_emb)
    
    return synthetic


def fuse_embeddings(embeddings: List[np.ndarray], method: str = None) -> np.ndarray:
    """
    Fuse multiple embeddings into enriched representation
    
    Args:
        embeddings: List of embeddings to fuse
        method: 'average', 'weighted_average', or 'max'
        
    Returns:
        Fused embedding (L2 normalized)
    """
    if method is None:
        method = config.EMBEDDING_FUSION_METHOD
    
    embeddings = np.array(embeddings)
    
    if method == 'average':
        fused = np.mean(embeddings, axis=0)
    elif method == 'weighted_average':
        # Weight recent embeddings more
        weights = np.linspace(0.5, 1.0, len(embeddings))
        weights = weights / weights.sum()
        fused = np.average(embeddings, axis=0, weights=weights)
    elif method == 'max':
        fused = np.max(embeddings, axis=0)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    # L2 normalize
    fused = fused / (np.linalg.norm(fused) + 1e-8)
    
    return fused


def create_enriched_embedding(embeddings: List[np.ndarray], 
                              use_augmentation: bool = True) -> np.ndarray:
    """
    Create enriched embedding using vector addition and fusion
    
    Args:
        embeddings: List of embeddings from same person
        use_augmentation: Whether to add synthetic samples
        
    Returns:
        Enriched embedding
    """
    all_embeddings = list(embeddings)
    
    # Add synthetic variations
    if use_augmentation:
        for emb in embeddings:
            synthetic = generate_synthetic_embeddings(emb, num_samples=2)
            all_embeddings.extend(synthetic)
    
    # Fuse all embeddings
    enriched = fuse_embeddings(all_embeddings)
    
    return enriched
