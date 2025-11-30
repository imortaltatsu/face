"""
User profile management with face embeddings using DuckDB
"""

import os
import duckdb
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
import config
from embedding_augmentation import create_enriched_embedding


class UserProfile:
    """User profile with face embeddings"""
    
    def __init__(self, user_id: str, name: str = None):
        """
        Initialize user profile
        
        Args:
            user_id: Unique user identifier
            name: User's name (optional)
        """
        self.user_id = user_id
        self.name = name or user_id
        self.embeddings: List[np.ndarray] = []
        self.enriched_embedding: Optional[np.ndarray] = None
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def add_embedding(self, embedding: np.ndarray):
        """Add a face embedding to profile"""
        self.embeddings.append(embedding)
        
        # Keep only recent embeddings
        if len(self.embeddings) > config.EMBEDDINGS_PER_USER:
            self.embeddings = self.embeddings[-config.EMBEDDINGS_PER_USER:]
        
        # Update enriched embedding
        if config.USE_ENRICHED_EMBEDDINGS and len(self.embeddings) > 1:
            self.enriched_embedding = create_enriched_embedding(self.embeddings)
        else:
            self.enriched_embedding = embedding
        
        self.updated_at = datetime.now().isoformat()
    
    def get_embedding(self) -> np.ndarray:
        """Get the best embedding (enriched or latest)"""
        if self.enriched_embedding is not None:
            return self.enriched_embedding
        elif self.embeddings:
            return self.embeddings[-1]
        else:
            raise ValueError(f"No embeddings for user {self.user_id}")


class ProfileDatabase:
    """DuckDB-based database for managing user profiles"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize DuckDB profile database
        
        Args:
            db_path: Path to DuckDB database file
        """
        if db_path is None:
            os.makedirs(config.PROFILE_STORAGE_PATH, exist_ok=True)
            db_path = os.path.join(config.PROFILE_STORAGE_PATH, 'profiles.db')
        
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_tables()
        print(f"✓ DuckDB profile database initialized at {db_path}")
    
    def _init_tables(self):
        """Create database tables if they don't exist"""
        # Users table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
        
        # Embeddings table (stores individual embeddings)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                user_id VARCHAR,
                embedding DOUBLE[512],
                created_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Enriched embeddings table (stores fused embeddings)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS enriched_embeddings (
                user_id VARCHAR PRIMARY KEY,
                embedding DOUBLE[512],
                updated_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Create index for faster lookups
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS embedding_id_seq START 1")
    
    def create_profile(self, user_id: str, name: str = None, 
                      initial_embedding: np.ndarray = None) -> UserProfile:
        """
        Create new user profile
        
        Args:
            user_id: Unique user ID
            name: User's name
            initial_embedding: First face embedding
            
        Returns:
            Created UserProfile
        """
        # Check if user exists
        result = self.conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?", [user_id]
        ).fetchone()
        
        if result:
            raise ValueError(f"User {user_id} already exists")
        
        # Create user
        now = datetime.now()
        self.conn.execute("""
            INSERT INTO users (user_id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, [user_id, name or user_id, now, now])
        
        profile = UserProfile(user_id, name)
        
        # Add initial embedding if provided
        if initial_embedding is not None:
            self._add_embedding_to_db(user_id, initial_embedding)
            profile.add_embedding(initial_embedding)
        
        return profile
    
    def _add_embedding_to_db(self, user_id: str, embedding: np.ndarray):
        """Add embedding to database"""
        # Insert embedding
        self.conn.execute("""
            INSERT INTO embeddings (id, user_id, embedding, created_at)
            VALUES (nextval('embedding_id_seq'), ?, ?, ?)
        """, [user_id, embedding.tolist(), datetime.now()])
        
        # Update user's updated_at
        self.conn.execute("""
            UPDATE users SET updated_at = ? WHERE user_id = ?
        """, [datetime.now(), user_id])
        
        # Get all embeddings for this user
        embeddings = self._get_user_embeddings(user_id)
        
        # Keep only recent embeddings
        if len(embeddings) > config.EMBEDDINGS_PER_USER:
            # Delete oldest embeddings
            self.conn.execute("""
                DELETE FROM embeddings 
                WHERE id IN (
                    SELECT id FROM embeddings 
                    WHERE user_id = ? 
                    ORDER BY created_at ASC 
                    LIMIT ?
                )
            """, [user_id, len(embeddings) - config.EMBEDDINGS_PER_USER])
            
            embeddings = embeddings[-config.EMBEDDINGS_PER_USER:]
        
        # Create enriched embedding if enabled
        if config.USE_ENRICHED_EMBEDDINGS and len(embeddings) > 1:
            enriched = create_enriched_embedding(embeddings)
            self._save_enriched_embedding(user_id, enriched)
        else:
            self._save_enriched_embedding(user_id, embedding)
    
    def _save_enriched_embedding(self, user_id: str, embedding: np.ndarray):
        """Save enriched embedding"""
        self.conn.execute("""
            INSERT OR REPLACE INTO enriched_embeddings (user_id, embedding, updated_at)
            VALUES (?, ?, ?)
        """, [user_id, embedding.tolist(), datetime.now()])
    
    def _get_user_embeddings(self, user_id: str) -> List[np.ndarray]:
        """Get all embeddings for a user"""
        result = self.conn.execute("""
            SELECT embedding FROM embeddings 
            WHERE user_id = ? 
            ORDER BY created_at ASC
        """, [user_id]).fetchall()
        
        return [np.array(row[0]) for row in result]
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        # Get user info
        result = self.conn.execute("""
            SELECT user_id, name, created_at, updated_at 
            FROM users WHERE user_id = ?
        """, [user_id]).fetchone()
        
        if not result:
            return None
        
        profile = UserProfile(result[0], result[1])
        profile.created_at = result[2].isoformat() if result[2] else datetime.now().isoformat()
        profile.updated_at = result[3].isoformat() if result[3] else datetime.now().isoformat()
        
        # Get embeddings
        profile.embeddings = self._get_user_embeddings(user_id)
        
        # Get enriched embedding
        enriched_result = self.conn.execute("""
            SELECT embedding FROM enriched_embeddings WHERE user_id = ?
        """, [user_id]).fetchone()
        
        if enriched_result:
            profile.enriched_embedding = np.array(enriched_result[0])
        
        return profile
    
    def update_profile(self, user_id: str, embedding: np.ndarray):
        """Add new embedding to existing profile"""
        # Check if user exists
        result = self.conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?", [user_id]
        ).fetchone()
        
        if not result:
            raise ValueError(f"User {user_id} not found")
        
        self._add_embedding_to_db(user_id, embedding)
    
    def delete_profile(self, user_id: str):
        """Delete user profile (cascades to embeddings)"""
        self.conn.execute("DELETE FROM users WHERE user_id = ?", [user_id])
    
    def list_profiles(self) -> List[dict]:
        """List all profiles (without embeddings)"""
        result = self.conn.execute("""
            SELECT u.user_id, u.name, u.created_at, u.updated_at,
                   COUNT(e.id) as num_embeddings
            FROM users u
            LEFT JOIN embeddings e ON u.user_id = e.user_id
            GROUP BY u.user_id, u.name, u.created_at, u.updated_at
        """).fetchall()
        
        return [
            {
                'user_id': row[0],
                'name': row[1],
                'created_at': row[2].isoformat() if row[2] else None,
                'updated_at': row[3].isoformat() if row[3] else None,
                'num_embeddings': row[4]
            }
            for row in result
        ]
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all user embeddings for identification (uses enriched embeddings)"""
        result = self.conn.execute("""
            SELECT user_id, embedding FROM enriched_embeddings
        """).fetchall()
        
        return {row[0]: np.array(row[1]) for row in result}
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Global database instance
_db_instance = None

def get_database() -> ProfileDatabase:
    """Get or create global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = ProfileDatabase()
    return _db_instance
