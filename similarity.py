"""
Similarity computation and verification
"""

import numpy as np
from typing import Optional, Tuple
import config


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0-1, higher = more similar)
    """
    # Ensure L2 normalized
    emb1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
    emb2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
    
    similarity = np.dot(emb1, emb2)
    return float(similarity)


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute Euclidean distance between embeddings"""
    return float(np.linalg.norm(embedding1 - embedding2))


def verify_faces(embedding1: np.ndarray, embedding2: np.ndarray, 
                 threshold: float = None) -> Tuple[bool, float]:
    """
    Verify if two face embeddings belong to same person
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
        threshold: Similarity threshold (default from config)
        
    Returns:
        (is_same_person, similarity_score)
    """
    if threshold is None:
        threshold = config.VERIFICATION_THRESHOLD
    
    similarity = cosine_similarity(embedding1, embedding2)
    is_match = similarity >= threshold
    
    return is_match, similarity


def identify_face(query_embedding: np.ndarray, 
                  database_embeddings: dict,
                  threshold: float = None) -> Tuple[Optional[str], float]:
    """
    Identify a face from database (1:N matching)
    
    Args:
        query_embedding: Query face embedding
        database_embeddings: Dict of {user_id: embedding}
        threshold: Similarity threshold
        
    Returns:
        (best_match_user_id, similarity_score) or (None, 0.0)
    """
    if threshold is None:
        threshold = config.IDENTIFICATION_THRESHOLD
    
    best_match = None
    best_score = 0.0
    
    for user_id, db_embedding in database_embeddings.items():
        similarity = cosine_similarity(query_embedding, db_embedding)
        
        if similarity > best_score:
            best_score = similarity
            best_match = user_id
    
    if best_score >= threshold:
        return best_match, best_score
    
    return None, best_score
