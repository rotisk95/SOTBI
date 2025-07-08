import time
import logging
import msgpack
import numpy as np
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class TrieNode:
    """
    Unified TrieNode class with activation-based learning and lazy embedding loading.
    
    Features:
    - Lazy embedding loading from database
    - Activation-based learning with reward history
    - Comprehensive relevance calculation with bridging analysis
    - Completeness detection and temporal tracking
    - FIXED: Uses Python native types for serialization compatibility
    """
    
    def __init__(self, token: str = None, db_env=None):
        # Core trie structure
        self.token = token
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_sequence = False
        
        # Database environment for lazy loading
        self._db_env = db_env
        
        # FIXED: Use Python native types instead of numpy types
        self.relevance_score: float = 0.0          # Python float
        self.activation_level: float = 0.0         # Python float
        self.is_complete: bool = False             # Python bool
        self.reward_history: List[float] = []      # List of Python floats
        self.access_count: int = 0                 # Python int
        self.last_accessed: float = time.time()   # Python float
        
        # FIXED: Metadata uses Python native types
        self.metadata = {
            'frequency': 0,                        # Python int
            'total_reward': 0.0,                   # Python float
            'avg_reward': 0.0,                     # Python float
            'decay_rate': 0.01,                    # Python float
            'creation_time': time.time()           # Python float
        }
        
        # Lazy embedding loading (numpy array is fine - handled as bytes)
        self._embedding_key: Optional[str] = None
        self._embedding_loaded: bool = False
        self._embedding: Optional[np.ndarray] = None

    @property
    def token_embedding(self) -> Optional[np.ndarray]:
        """Lazy load embedding on first access"""
        if not self._embedding_loaded and self._embedding_key and self._db_env:
            self._load_embedding()
        return self._embedding
    
    @token_embedding.setter
    def token_embedding(self, value):
        """Set embedding directly"""
        self._embedding = value
        self._embedding_loaded = True
    
    def set_embedding_info(self, embedding_key: str):
        """Set up lazy loading for embedding"""
        self._embedding_key = embedding_key
        self._embedding_loaded = False
        self._embedding = None
    
    def _load_embedding(self):
        """Load embedding from database with improved error handling"""
        try:
            # Check if database environment is still valid
            if not self._db_env or self._db_env.info()['map_size'] == 0:
                logger.warning(f"Database environment invalid for {self._embedding_key}")
                self._embedding_loaded = True
                return
                
            with self._db_env.begin() as txn:
                value = txn.get(self._embedding_key.encode())
                if value:
                    data = msgpack.unpackb(value, raw=False, strict_map_key=False)
                    embedding_bytes = data.get('embedding')
                    if embedding_bytes:
                        self._embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                        logger.debug(f"Loaded embedding for {self._embedding_key}")
                self._embedding_loaded = True
                
        except Exception as e:
            # More specific error handling
            if "closed" in str(e).lower() or "deleted" in str(e).lower():
                logger.warning(f"Database connection closed for {self._embedding_key}, skipping lazy load")
            else:
                logger.warning(f"Failed to load embedding for {self._embedding_key}: {e}")
            self._embedding_loaded = True
    

    def calculate_relevance(self, context_embedding: np.ndarray = None, query_embedding: np.ndarray = None) -> float:
        """
        Comprehensive relevance calculation with dynamic weighting and bridging analysis.
        FIXED: All numpy scalar results converted to Python floats.
        
        Features:
        - Token-context and token-query similarities
        - Context-query relationship analysis
        - Dynamic weighting based on alignment
        - Bridging rewards for connecting dissimilar concepts
        - Historical performance integration
        """
        if self.token_embedding is None:
            return 0.0

        relevance_components = []

        # Calculate base similarities - FIXED: Convert numpy scalars to Python floats
        context_sim = 0.0
        query_sim = 0.0
        context_query_sim = 0.0

        # Token-Context similarity
        if context_embedding is not None:
            context_sim = float(np.dot(self.token_embedding, context_embedding))  # Convert numpy scalar

        # Token-Query similarity  
        if query_embedding is not None:
            query_sim = float(np.dot(self.token_embedding, query_embedding))  # Convert numpy scalar

        # Context-Query relationship analysis
        if context_embedding is not None and query_embedding is not None:
            context_query_sim = float(np.dot(context_embedding, query_embedding))  # Convert numpy scalar

            # Direct context-query similarity component
            relevance_components.append(context_query_sim * 0.15)

            # Dynamic weighting based on alignment
            if context_query_sim > 0.7:  # High alignment - context consistency matters more
                context_weight = 0.35
                query_weight = 0.25
                bridge_bonus = 0.0
            elif context_query_sim < 0.3:  # Low alignment - bridging is crucial
                context_weight = 0.25
                query_weight = 0.25  
                bridge_bonus = 0.15
            else:  # Medium alignment - balanced approach
                context_weight = 0.30
                query_weight = 0.30
                bridge_bonus = 0.05

            # Apply dynamic weights to base similarities
            relevance_components.append(context_sim * context_weight)
            relevance_components.append(query_sim * query_weight)

            # Bridging analysis with adaptive bonus
            if bridge_bonus > 0:
                # Calculate how well token bridges the context-query gap
                bridge_effectiveness = (context_sim + query_sim) / 2.0

                # Geometric mean for better bridging detection
                if context_sim > 0 and query_sim > 0:
                    bridge_geometric = float(np.sqrt(context_sim * query_sim))  # Convert numpy scalar
                    bridge_effectiveness = max(bridge_effectiveness, bridge_geometric)

                # Inverse relationship: more bridging bonus when context-query similarity is lower
                bridge_multiplier = 1.0 - context_query_sim  
                final_bridge_score = bridge_effectiveness * bridge_multiplier * bridge_bonus
                relevance_components.append(final_bridge_score)

            # Interaction effects between similarities
            synergy_score = context_sim * query_sim  # Multiplicative synergy
            relevance_components.append(synergy_score * 0.1)

            # Consistency reward: penalize tokens that are great at one but terrible at the other
            consistency_penalty = abs(context_sim - query_sim) * -0.05
            relevance_components.append(consistency_penalty)

        else:
            # Fallback when only one embedding is available
            if context_embedding is not None:
                relevance_components.append(context_sim * 0.5)
            if query_embedding is not None:
                relevance_components.append(query_sim * 0.5)

        # Historical performance components
        freq_relevance = min(1.0, self.metadata['frequency'] / 100.0)
        relevance_components.append(freq_relevance * 0.03)

        reward_relevance = max(0.0, min(1.0, self.metadata['avg_reward']))
        relevance_components.append(reward_relevance * 0.07)

        # Recency and activation state bonus
        activation_bonus = self.activation_level * 0.05
        relevance_components.append(activation_bonus)

        # Calculate final score with sigmoid normalization - FIXED: Convert numpy scalar
        raw_score = sum(relevance_components) if relevance_components else 0.0
        self.relevance_score = float(1.0 / (1.0 + np.exp(-raw_score * 2.0)))  # Convert numpy scalar to Python float

        return self.relevance_score
    
    def update_activation(self, reward: float = 0.0, context_relevance: float = 0.0):
        """
        Update activation level based on reward and context relevance.
        FIXED: Ensure all calculations use Python floats.
        
        Args:
            reward: Reward signal from user feedback or learning
            context_relevance: Relevance to current context
        """
        # FIXED: Ensure inputs are Python floats
        reward = float(reward)
        context_relevance = float(context_relevance)
        
        # Maintain reward history with Python floats
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
        
        # Calculate activation updates - all Python floats
        reward_contribution = reward * 0.1
        context_contribution = context_relevance * 0.05
        decay = self.activation_level * self.metadata['decay_rate']
        
        # Update activation with bounds - ensure Python float
        self.activation_level = max(0.0, min(1.0, 
            self.activation_level + reward_contribution + context_contribution - decay))
        
        # Update metadata with Python native types
        self.metadata['frequency'] += 1
        self.metadata['total_reward'] += reward
        self.metadata['avg_reward'] = self.metadata['total_reward'] / self.metadata['frequency']
        self.access_count += 1
        self.last_accessed = time.time()
        
        logger.debug(f"Updated activation for token '{self.token}': "
                    f"activation={self.activation_level:.3f}, reward={reward:.3f}")
    
    def update_completeness(self, heuristic_rules: Dict[str, Any] = None):
        """
        Update completeness based on token characteristics and rules.
        
        Args:
            heuristic_rules: Custom rules for determining completeness
        """
        if self.token is None:
            return
        
        default_rules = {
            'end_punctuation': ['.', '!', '?'],
            'min_sequence_length': 2,
            'complete_pos_tags': ['NOUN', 'VERB']
        }
        
        rules = heuristic_rules or default_rules
        
        # Check for end punctuation
        if self.token in rules.get('end_punctuation', []):
            self.is_complete = True
            logger.debug(f"Marked token '{self.token}' as complete due to end punctuation")
    
    def get_activation_info(self) -> Dict[str, Any]:
        """Get comprehensive activation and learning information.
        FIXED: Convert numpy results to Python floats."""
        return {
            'token': self.token,
            'activation_level': self.activation_level,
            'relevance_score': self.relevance_score,
            'is_complete': self.is_complete,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'metadata': self.metadata.copy(),
            'reward_history_stats': {
                'count': len(self.reward_history),
                'average': float(np.mean(self.reward_history)) if self.reward_history else 0.0,  # Convert numpy scalar
                'recent_average': float(np.mean(self.reward_history[-10:])) if len(self.reward_history) >= 10 else float(np.mean(self.reward_history)) if self.reward_history else 0.0,  # Convert numpy scalar
                'max': max(self.reward_history) if self.reward_history else 0.0,
                'min': min(self.reward_history) if self.reward_history else 0.0
            },
            'has_embedding': self.token_embedding is not None,
            'children_count': len(self.children),
            'is_end_of_sequence': self.is_end_of_sequence
        }
    
    def decay_activation(self):
        """Apply natural decay to activation level."""
        self.activation_level *= (1.0 - self.metadata['decay_rate'])
        self.activation_level = max(0.0, self.activation_level)
    
    def boost_activation(self, boost_factor: float = 0.1):
        """Boost activation level (used for spreading activation)."""
        boost_factor = float(boost_factor)  # Ensure Python float
        self.activation_level = min(1.0, self.activation_level + boost_factor)
    
    def __repr__(self):
        """String representation for debugging."""
        return (f"TrieNode(token='{self.token}', activation={self.activation_level:.3f}, "
                f"children={len(self.children)}, complete={self.is_complete})")
    
    def __str__(self):
        """Human-readable string representation."""
        return f"'{self.token}' (activation: {self.activation_level:.3f})"