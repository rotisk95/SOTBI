import time
import logging
import types
import msgpack
import numpy as np
import hashlib
import gc
import os
from queue import Queue
import lmdb
from typing import Dict, List, Optional, Any, Tuple
import uuid

from embedding_keys import generate_with_level

logger = logging.getLogger(__name__)

REWARD_HISTORY_LIMIT = 10000

class TrieNode:
    """
    Enhanced TrieNode with simple node mapping at root level.
    MINIMAL CHANGES: Just added node_id and path tracking to your existing class.
    """
    
    def __init__(self, token: str = None, db_env=None, context_window=None):
        # Core trie structure (UNCHANGED)
        self.token = token
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_sequence = False
        
        # NEW: Simple node identification
        self.node_id: str = self._generate_node_id()
        self.path_tokens: List[str] = []  # Will be set when added to trie
        self.hierarchy_level: int = 0     # Will be set when added to trie
        
        # Database environment for lazy loading (UNCHANGED)
        self._db_env = db_env
        self.context_window = context_window
        
        # UNCHANGED: All existing properties with Python native types
        self.relevance_score: float = 0.0
        self.activation_level: float = 0.0
        self.is_complete: bool = False
        self.reward_history: List[float] = []
        self.access_count: int = 0
        self.last_accessed: float = time.time()
        
        # UNCHANGED: Metadata uses Python native types
        self.metadata = {
            'frequency': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'decay_rate': 0.01,
            'creation_time': time.time()
        }
        
        # UNCHANGED: Event-driven activation properties
        self.is_activated: bool = False
        self.activation_timestamp: float = 0.0
        
        # UNCHANGED: Lazy embedding loading
        self._embedding_key: Optional[str] = None
        self._embedding_loaded: bool = False
        self._embedding: Optional[np.ndarray] = None
        
        # NEW: Node registry (only exists on root node)
        self.node_registry: Optional[Dict] = None
        if token is None:  # This is the root node
            self._init_node_registry()
    
    def _generate_node_id(self) -> str:
        """Generate unique ID for this node."""
        return f"node_{uuid.uuid4().hex[:12]}_{int(time.time() * 1000) % 100000}"
    
    def _init_node_registry(self):
        """Initialize the node registry (only called on root)."""
        self.node_registry = {
            'by_id': {},           # node_id -> node
            'by_path': {},         # path_string -> node  
            'by_level': {},        # level -> list of nodes
            'by_token_path': {},   # tuple(tokens) -> node
            'embedding_keys': {}   # embedding_key -> node
        }
        logger.info("Initialized node registry on root node")
    
    def register_node(self, node: 'TrieNode', path_tokens: List[str]):
        """Register a node in the root's registry."""
        if self.node_registry is None:
            logger.error("register_node called on non-root node")
            return
        
        # Set node path information
        node.path_tokens = path_tokens.copy()
        node.hierarchy_level = len(path_tokens)
        
        # Register in all lookup structures
        self.node_registry['by_id'][node.node_id] = node
        
        path_string = '/'.join(path_tokens) if path_tokens else 'ROOT'
        self.node_registry['by_path'][path_string] = node
        
        level = len(path_tokens)
        if level not in self.node_registry['by_level']:
            self.node_registry['by_level'][level] = []
        self.node_registry['by_level'][level].append(node)
        
        path_tuple = tuple(path_tokens)
        self.node_registry['by_token_path'][path_tuple] = node
        
        # Generate and register embedding key
        embedding_key = self._generate_embedding_key(path_tokens, node.token)
        self.node_registry['embedding_keys'][embedding_key] = node
        node._embedding_key = embedding_key
        
        logger.debug(f"Registered node {node.node_id} at path: {path_tokens} (level {level})")
    
    def _generate_embedding_key(self, path_tokens: List[str], token: str) -> str:
        """Generate embedding key that includes hierarchy info."""
        level = len(path_tokens)
        if not path_tokens:
            return "emb_root"
        
        emb_key = generate_with_level(path_tokens, token)
        return emb_key

    def get_node_by_path(self, path_tokens: List[str]) -> Optional['TrieNode']:
        """Get node directly by path tokens (call on root)."""
        if self.node_registry is None:
            logger.error("get_node_by_path called on non-root node")
            return None
        path_tuple = tuple(path_tokens)
        return self.node_registry['by_token_path'].get(path_tuple)
    
    def get_node_by_id(self, node_id: str) -> Optional['TrieNode']:
        """Get node directly by ID (call on root)."""
        if self.node_registry is None:
            logger.error("get_node_by_id called on non-root node")
            return None
        return self.node_registry['by_id'].get(node_id)
    
    def get_nodes_at_level(self, level: int) -> List['TrieNode']:
        """Get all nodes at specific hierarchy level (call on root)."""
        if self.node_registry is None:
            logger.error("get_nodes_at_level called on non-root node")
            return []
        return self.node_registry['by_level'].get(level, [])
    
    def get_node_by_embedding_key(self, embedding_key: str) -> Optional['TrieNode']:
        """Get node by embedding key (call on root)."""
        if self.node_registry is None:
            logger.error("get_node_by_embedding_key called on non-root node")
            return None
        return self.node_registry['embedding_keys'].get(embedding_key)
    
    # Add the lookup method
    def get_nodes_by_token(self, token: str) -> List['TrieNode']:
        """Get all nodes with a specific token."""
        return self.node_registry['by_token'].get(token, [])
    
    def get_all_registered_nodes(self) -> List['TrieNode']:
        """Get all nodes in the registry (call on root)."""
        if self.node_registry is None:
            return []
        return list(self.node_registry['by_id'].values())
    
    def enhance_trie_node_registry(self):
        """
        FIXED: Token-based registry enhancement with safer error handling.
        PREVENTS: Issues during token lookup setup.
        """
        if not hasattr(self, 'node_registry') or self.node_registry is None:
            logger.error("enhance_trie_node_registry called on non-root node")
            return

        logger.info("Enhancing registry with token-based lookup...")

        try:
            # Add token-based lookup table
            self.node_registry['by_token'] = {}

            # Populate with progress tracking
            token_count = 0
            for node in self.node_registry['by_id'].values():
                if node.token:
                    if node.token not in self.node_registry['by_token']:
                        self.node_registry['by_token'][node.token] = []
                    self.node_registry['by_token'][node.token].append(node)
                    token_count += 1

            unique_tokens = len(self.node_registry['by_token'])
            logger.info(f"Enhanced registry with {unique_tokens} unique tokens ({token_count} total token instances)")

        except Exception as e:
            logger.error(f"Error enhancing registry with token lookup: {e}")
            # Create empty token registry as fallback
            self.node_registry['by_token'] = {}
                
    def update_registry_on_node_creation(self, node, path_tokens: List[str]):
        """
        ENHANCEMENT: Update the token registry when new nodes are created.
        CALL THIS: In your add_sequence method after creating new nodes.
        """
        # Call the existing register_node method
        self.register_node(node, path_tokens)

        # Also update token registry if it exists
        if (hasattr(self, 'node_registry') and self.node_registry and 
            'by_token' in self.node_registry and node.token):

            if node.token not in self.node_registry['by_token']:
                self.node_registry['by_token'][node.token] = []
            self.node_registry['by_token'][node.token].append(node)
    
    def debug_registry(self):
        """Debug the node registry (call on root)."""
        if self.node_registry is None:
            print("❌ No registry (not called on root)")
            return
        
        total_nodes = len(self.node_registry['by_id'])
        max_level = max(self.node_registry['by_level'].keys()) if self.node_registry['by_level'] else 0
        
        print(f"\n📊 NODE REGISTRY DEBUG:")
        print(f"Total registered nodes: {total_nodes}")
        print(f"Maximum hierarchy level: {max_level}")
        print(f"Registry sizes:")
        print(f"  by_id: {len(self.node_registry['by_id'])}")
        print(f"  by_path: {len(self.node_registry['by_path'])}")
        print(f"  by_token_path: {len(self.node_registry['by_token_path'])}")
        print(f"  embedding_keys: {len(self.node_registry['embedding_keys'])}")
        
        print(f"\nNodes per level:")
        for level in sorted(self.node_registry['by_level'].keys()):
            count = len(self.node_registry['by_level'][level])
            print(f"  Level {level}: {count} nodes")
        
        print(f"\nSample registrations:")
        for i, (path_tuple, node) in enumerate(list(self.node_registry['by_token_path'].items())[:5]):
            path_str = '/'.join(path_tuple) if path_tuple else 'ROOT'
            print(f"  {node.node_id} → '{path_str}' (token: '{node.token}')")

    # UNCHANGED: All your existing methods from the original class
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
            if "closed" in str(e).lower() or "deleted" in str(e).lower():
                logger.warning(f"Database connection closed for {self._embedding_key}, skipping lazy load")
            else:
                logger.warning(f"Failed to load embedding for {self._embedding_key}: {e}")
            self._embedding_loaded = True
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:        
        """Calculate similarity between two arbitrary embeddings using ensemble method."""
        if embedding1 is None or embedding2 is None:
            logger.debug("One or both embeddings are None, returning 0.0 similarity")
            return 0.0
        
        if np.linalg.norm(embedding1) == 0 or np.linalg.norm(embedding2) == 0:
            logger.debug("One or both embeddings have zero norm, returning 0.0 similarity")
            return 0.0
        
        try:
            similarity = self.context_window._calculate_ensemble_similarity(embedding1, embedding2)
            result = float(similarity)
            logger.debug(f"Calculated ensemble similarity: {result:.4f}")
            return result
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def calculate_token_similarity(self, other_embedding: np.ndarray) -> float:
        """Compare self.token_embedding with another embedding."""
        if self.token_embedding is None or other_embedding is None:
            logger.debug(f"Token embedding or other embedding is None for token '{self.token}'")
            return 0.0
        
        return self.calculate_similarity(self.token_embedding, other_embedding)
    
    def calculate_relevance(self, context_embedding: np.ndarray = None, query_embedding: np.ndarray = None) -> float:
        """Comprehensive relevance calculation with fixed similarity method calls."""
        if self.token_embedding is None:
            logger.debug(f"No token embedding available for token '{self.token}', returning 0.0 relevance")
            return 0.0
    
        relevance_components = []
        context_sim = 0.0
        query_sim = 0.0
        context_query_sim = 0.0
    
        if context_embedding is not None:
            context_sim = self.calculate_token_similarity(context_embedding)
            logger.debug(f"Token-context similarity for '{self.token}': {context_sim:.4f}")
    
        if query_embedding is not None:
            query_sim = self.calculate_token_similarity(query_embedding)
            logger.debug(f"Token-query similarity for '{self.token}': {query_sim:.4f}")
    
        if context_embedding is not None and query_embedding is not None:
            context_query_sim = self.calculate_similarity(context_embedding, query_embedding)
            logger.debug(f"Context-query similarity: {context_query_sim:.4f}")
    
            relevance_components.append(context_query_sim * 0.15)
    
            if context_query_sim > 0.7:
                context_weight = 0.35
                query_weight = 0.25
                bridge_bonus = 0.0
            elif context_query_sim < 0.3:
                context_weight = 0.25
                query_weight = 0.25  
                bridge_bonus = 0.15
            else:
                context_weight = 0.30
                query_weight = 0.30
                bridge_bonus = 0.05
    
            relevance_components.append(context_sim * context_weight)
            relevance_components.append(query_sim * query_weight)
    
            if bridge_bonus > 0:
                bridge_effectiveness = (context_sim + query_sim) / 2.0
    
                if context_sim > 0 and query_sim > 0:
                    bridge_geometric = float(np.sqrt(context_sim * query_sim))
                    bridge_effectiveness = max(bridge_effectiveness, bridge_geometric)
    
                bridge_multiplier = 1.0 - context_query_sim  
                final_bridge_score = bridge_effectiveness * bridge_multiplier * bridge_bonus
                relevance_components.append(final_bridge_score)
                logger.debug(f"Bridge score for '{self.token}': {final_bridge_score:.4f}")
    
            synergy_score = context_sim * query_sim
            relevance_components.append(synergy_score * 0.1)
    
            consistency_penalty = abs(context_sim - query_sim) * -0.05
            relevance_components.append(consistency_penalty)
    
        else:
            if context_embedding is not None:
                relevance_components.append(context_sim * 0.5)
            if query_embedding is not None:
                relevance_components.append(query_sim * 0.5)
    
        freq_relevance = min(1.0, self.metadata['frequency'] / 100.0)
        relevance_components.append(freq_relevance * 0.03)
    
        reward_relevance = max(0.0, min(1.0, self.metadata['avg_reward']))
        relevance_components.append(reward_relevance * 0.07)
    
        activation_bonus = self.activation_level * 0.05
        relevance_components.append(activation_bonus)
    
        raw_score = sum(relevance_components) if relevance_components else 0.0
        self.relevance_score = float(1.0 / (1.0 + np.exp(-raw_score * 2.0)))
    
        logger.info(f"Calculated relevance for token '{self.token}': {self.relevance_score:.4f} "
                   f"(components: {len(relevance_components)})")
        
        return self.relevance_score
    
    def update_activation(self, reward: float = 0.0, context_relevance: float = 0.0):
        """Update activation with negative feedback support."""
        reward = float(reward)
        context_relevance = float(context_relevance)

        self.reward_history.append(reward)
        if len(self.reward_history) > REWARD_HISTORY_LIMIT:
            self.reward_history.pop(0)

        if len(self.reward_history) >= 3:
            recent_rewards = self.reward_history[-3:]
            reward_variance = sum((r - sum(recent_rewards)/3)**2 for r in recent_rewards) / 3
            confidence = max(0.1, 1.0 - reward_variance)
        else:
            confidence = 0.5

        if reward < 0:
            punishment_strength = abs(reward) * 0.15 * confidence
            reward_contribution = -punishment_strength
            logger.debug(f"Applying negative feedback: {reward} -> {reward_contribution}")
        else:
            reward_contribution = reward * 0.1 * confidence
            logger.debug(f"Applying positive feedback: {reward} -> {reward_contribution}")

        context_contribution = context_relevance * 0.05
        decay = self.activation_level * self.metadata['decay_rate']

        old_activation = self.activation_level
        self.activation_level = max(0.0, min(1.0, 
            self.activation_level + reward_contribution + context_contribution - decay))

        activation_change = self.activation_level - old_activation
        logger.debug(f"Activation change: {old_activation:.3f} -> {self.activation_level:.3f} "
                    f"(delta: {activation_change:+.3f})")

        self.metadata['frequency'] += 1
        self.metadata['total_reward'] += reward
        self.metadata['avg_reward'] = self.metadata['total_reward'] / self.metadata['frequency']

        if reward < 0:
            self.metadata['negative_feedback_count'] = self.metadata.get('negative_feedback_count', 0) + 1

        negative_ratio = self.metadata.get('negative_feedback_count', 0) / self.metadata['frequency']
        if negative_ratio > 0.3:
            logger.warning(f"High negative feedback ratio for token '{self.token}': {negative_ratio:.2f}")

        self.access_count += 1
        self.last_accessed = time.time()

        logger.debug(f"Updated activation for token '{self.token}': "
                    f"activation={self.activation_level:.3f}, reward={reward:.3f}, "
                    f"confidence={confidence:.3f}, neg_ratio={negative_ratio:.3f}")
    
    def update_completeness(self, heuristic_rules: Dict[str, Any] = None):
        """Update completeness based on token characteristics and rules."""
        if self.token is None:
            return
        
        default_rules = {
            'end_punctuation': ['.', '!', '?'],
            'min_sequence_length': 2,
            'complete_pos_tags': ['NOUN', 'VERB']
        }
        
        rules = heuristic_rules or default_rules
        
        if self.token in rules.get('end_punctuation', []):
            self.is_complete = True
            logger.debug(f"Marked token '{self.token}' as complete due to end punctuation")
    
    def get_activation_info(self) -> Dict[str, Any]:
        """Get comprehensive activation and learning information."""
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
                'average': float(np.mean(self.reward_history)) if self.reward_history else 0.0,
                'recent_average': float(np.mean(self.reward_history[-10:])) if len(self.reward_history) >= 10 else float(np.mean(self.reward_history)) if self.reward_history else 0.0,
                'max': max(self.reward_history) if self.reward_history else 0.0,
                'min': min(self.reward_history) if self.reward_history else 0.0
            },
            'has_embedding': self.token_embedding is not None,
            'children_count': len(self.children),
            'is_end_of_sequence': self.is_end_of_sequence,
            # NEW: Added hierarchy info
            'node_id': self.node_id,
            'path_tokens': self.path_tokens,
            'hierarchy_level': self.hierarchy_level
        }
    
    def decay_activation(self):
        """Apply natural decay to activation level."""
        self.activation_level *= (1.0 - self.metadata['decay_rate'])
        self.activation_level = max(0.0, self.activation_level)
    
    def boost_activation(self, boost_factor: float = 0.1):
        """Boost activation level (used for spreading activation)."""
        boost_factor = float(boost_factor)
        self.activation_level = min(1.0, self.activation_level + boost_factor)
    
    def add_child(self, token, child):
        self.children[token] = child
    
    def get_child(self, token):
        return self.children.get(token)
    
    def __repr__(self):
        """String representation for debugging."""
        return (f"TrieNode(id={self.node_id[:8]}..., token='{self.token}', "
                f"level={self.hierarchy_level}, children={len(self.children)})")
    
    def __str__(self):
        """Human-readable string representation."""
        return f"'{self.token}' (activation: {self.activation_level:.3f})"
    
