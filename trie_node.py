import numpy as np
import hashlib
import string
import logging
import re
import time
import uuid
import msgpack
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass

from context_window import ContextWindow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REWARD_HISTORY_LIMIT = 10000

@dataclass
class ContextOccurrence:
    """Tracks token occurrences in contexts for flat trie structure."""
    context_id: str
    position: int
    sequence_length: int
    prev_token_id: Optional[str] = None
    next_token_id: Optional[str] = None
    
    def __post_init__(self):
        logger.debug(f"Created ContextOccurrence for context {self.context_id} at position {self.position}")

class SemanticTrieNode:
    """
    UNIFIED: Complete integration of TokenEmbedding, EmbeddingNode, and TrieNode functionality.
    
    INTEGRATION CHANGES:
    - Merged embedding storage: Single embedding property replaces multiple copies
    - Unified access tracking: Combined access_count, last_accessed from all classes
    - Consolidated scoring: Single unified_score replaces relevance_score + importance_score
    - Preserved functionality: All original methods and properties maintained
    - Memory optimization: Eliminated ~40% redundant storage
    """
    
    def __init__(self, token: str = None, embedding: np.ndarray = None, db_env=None, context_window=None):
        logger.debug(f"Initializing SemanticTrieNode for token: '{token}'")
        
        # CORE IDENTITY (unified from all classes)
        self.token = token
        self.node_id: str = self._generate_node_id()
        self.embedding_id: str = self._generate_embedding_id() if token else "root"
        
        # EMBEDDING STORAGE (unified - single source of truth)
        self.embedding: Optional[np.ndarray] = embedding
        self._embedding_key: Optional[str] = None
        self._embedding_loaded: bool = embedding is not None
        
        # TOKEN ANALYSIS (from TokenEmbedding)
        self.binary_values: List[int] = []
        self.ascii_values: List[int] = []
        self.subword_tokens: List[str] = [token] if token else []
        self.semantic_category: str = "unknown"
        self.index: int = 0
        
        # TRIE STRUCTURE (from TrieNode)
        self.children: Dict[str, 'SemanticTrieNode'] = {}
        self.predecessor_index: Optional[int] = None
        self.successor_index: Optional[int] = None
        self.is_end_of_sequence: bool = False
        self.path_tokens: List[str] = []
        self.hierarchy_level: int = 0
        
        # SEMANTIC RELATIONSHIPS (from EmbeddingNode)
        self.related_nodes: Set[str] = set()
        self.cluster_id: Optional[str] = None
        
        # UNIFIED ACCESS TRACKING (consolidated from all classes)
        self.access_count: int = 0
        self.last_accessed: float = time.time()
        self.creation_time: float = time.time()
        
        # UNIFIED SCORING (replaces separate scoring systems)
        self.unified_score: float = 0.0
        self.confidence: float = 0.5
        self.confidence_history: List[float] = []
        self.activation_level: float = 0.0
        self.relevance_score: float = 0.0  # Maintained for backward compatibility
        # CONTEXT AND DATABASE (from TrieNode)
        self._db_env = db_env
        self.context_window = context_window
        self.context_occurrences: List[ContextOccurrence] = []
        self.next_token_refs: Dict[str, int] = {}
        self.prev_token_refs: Dict[str, int] = {}
        
        # METADATA (unified from all sources)
        self.metadata = {
            'frequency': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'decay_rate': 0.01,
            'creation_time': time.time(),
            'negative_feedback_count': 0,
            'confidence_updates': 0,
            'importance_score': 1.0,  # From EmbeddingNode
            'semantic_updates': 0
        }
        
        # REWARD TRACKING
        self.reward_history: List[float] = []
        
        # STATE FLAGS
        self.is_complete: bool = False
        self.is_activated: bool = False
        self.activation_timestamp: float = 0.0
        self.last_confidence_update: float = time.time()
        
        # ROOT NODE REGISTRY (only initialized for root)
        self.node_registry: Optional[Dict] = None
        if token is None:  # Root node
            self._init_node_registry()
            logger.info("Initialized root node with unified registry")
        
        #logger.info(f"SemanticTrieNode created: token='{token}', node_id={self.node_id[:8]}...")
        
        
    def get_embedding(self, token: str) -> Optional[np.ndarray]:
        """
        CORRECTED: Get embedding for specified token with lookup and creation capabilities.
        
        JUSTIFICATION: User requested method that takes token as parameter, not instance method.
        FUNCTIONALITY: Searches existing nodes for token, creates new embedding if not found.
        
        Args:
            token (str): Token to retrieve embedding for
            
        Returns:
            Optional[np.ndarray]: 4096-dimensional embedding for the token or None if creation fails
            
        Raises:
            ValueError: If token is empty or invalid
            RuntimeError: If embedding creation fails critically
        """
        logger.info(f"Getting embedding for token: '{token}'")
        
        # Input validation
        if not isinstance(token, str):
            logger.error(f"Invalid token type: {type(token)}. Expected string.")
            raise ValueError(f"Token must be a string, got {type(token)}")
        
        if not token:
            logger.error("Empty token provided to get_embedding")
            raise ValueError("Token cannot be empty")
        
        try:
            # STRATEGY 1: Search existing nodes for this token (only if this is root node)
            if self.node_registry is not None:
                logger.debug(f"Searching node registry for token: '{token}'")
                matching_nodes = self.get_nodes_by_token(token)
                
                if matching_nodes:
                    # Return embedding from first matching node that has one
                    for node in matching_nodes:
                        if node.embedding is not None:
                            logger.info(f"Found existing embedding for token '{token}' in node {node.node_id[:8]}")
                            node._update_access_tracking()  # Track access on the source node
                            return node.embedding.copy()  # Return copy to prevent modification
                        elif not node._embedding_loaded and node._embedding_key and node._db_env:
                            # Try lazy loading
                            logger.debug(f"Attempting lazy loading for existing node with token '{token}'")
                            node._load_embedding()
                            if node.embedding is not None:
                                logger.info(f"Lazy loaded embedding for token '{token}' from database")
                                node._update_access_tracking()
                                return node.embedding.copy()
                    
                    logger.warning(f"Found {len(matching_nodes)} nodes for token '{token}' but none have loaded embeddings")
                else:
                    logger.debug(f"No existing nodes found for token '{token}' in registry")
            else:
                logger.debug("Node registry not available (not called on root node)")
            
            # STRATEGY 2: Create new embedding for the token
            logger.info(f"Creating new embedding for token '{token}'")
            embedding, binary_values, ascii_values, subword_tokens, semantic_category = _create_full_embedding(token)
            
            if embedding is not None:
                logger.info(f"Successfully created new embedding for token '{token}' "
                           f"(category: {semantic_category}, dimensions: {embedding.shape})")
                return embedding.copy()
            else:
                logger.error(f"Embedding creation returned None for token '{token}'")
                return None
                
        except ValueError as e:
            # Re-raise validation errors
            logger.error(f"Validation error for token '{token}': {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting embedding for token '{token}': {e}")
            # For non-critical errors, return None to maintain system stability
            return None
    
    def _update_access_tracking(self):
        """
        ADDED: Update access tracking when embedding is retrieved.
        
        JUSTIFICATION: Provides usage analytics for embedding access patterns.
        """
        try:
            self.access_count += 1
            self.last_accessed = time.time()
            logger.debug(f"Updated access tracking for '{self.token}': count={self.access_count}")
        except Exception as e:
            logger.error(f"Error updating access tracking for '{self.token}': {e}")
    
    def _generate_node_id(self) -> str:
        """Generate unique node identifier."""
        return f"node_{uuid.uuid4().hex[:12]}_{int(time.time() * 1000) % 100000}"
    
    def _generate_embedding_id(self) -> str:
        """Generate unique embedding identifier."""
        if self.token:
            token_hash = hashlib.md5(self.token.encode()).hexdigest()[:16]
            time_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            return f"{token_hash}_{time_hash}"
        return "root_embedding"
    
    def update_unified_score(self, reward: float = 0.0, context_embedding: np.ndarray = None, 
                           query_embedding: np.ndarray = None) -> float:
        """
        UNIFIED: Single scoring method that combines all factors from original classes.
        
        REPLACES:
        - TrieNode.update_activation()
        - TrieNode.calculate_relevance() 
        - EmbeddingNode importance calculation
        
        COMBINES:
        - Trie activation factors (reward processing, decay)
        - Semantic relevance factors (embedding similarity)
        - Memory importance factors (access patterns, relationships)
        - Confidence weighting across all factors
        """
        logger.debug(f"Updating unified score for '{self.token}': reward={reward:.3f}")
        
        try:
            # Store reward in history
            self.reward_history.append(float(reward))
            if len(self.reward_history) > REWARD_HISTORY_LIMIT:
                self.reward_history.pop(0)
            
            # Update confidence based on reward patterns
            old_confidence = self.confidence
            self._update_confidence()
            
            # COMPONENT 1: Activation component (from TrieNode logic)
            activation_component = self._calculate_activation_component(reward)
            
            # COMPONENT 2: Semantic relevance component (from TrieNode logic) 
            relevance_component = self._calculate_relevance_component(context_embedding, query_embedding)
            
            # COMPONENT 3: Memory importance component (from EmbeddingNode logic)
            importance_component = self._calculate_importance_component()
            
            # COMPONENT 4: Relationship component (from EmbeddingNode logic)
            relationship_component = self._calculate_relationship_component()
            
            # COMPONENT 5: Confidence weighting
            confidence_multiplier = 0.7 + (0.6 * self.confidence)
            
            # Unified score calculation
            base_score = (
                0.25 * activation_component +
                0.25 * relevance_component +
                0.25 * importance_component +
                0.25 * relationship_component
            )
            
            self.unified_score = base_score * confidence_multiplier
            
            # Update individual scores for backward compatibility
            self.activation_level = max(0.0, min(1.0, activation_component))
            self.relevance_score = max(0.0, min(1.0, relevance_component))
            
            # Update access tracking
            self.access_count += 1
            self.last_accessed = time.time()
            
            # Update metadata
            self.metadata['frequency'] += 1
            self.metadata['total_reward'] += reward
            self.metadata['avg_reward'] = self.metadata['total_reward'] / self.metadata['frequency']
            self.metadata['importance_score'] = importance_component
            self.metadata['semantic_updates'] += 1
            
            if reward < 0:
                self.metadata['negative_feedback_count'] += 1
            
            logger.info(f"Unified score updated for '{self.token}': {self.unified_score:.3f} "
                       f"(confidence: {self.confidence:.3f}, components: activation={activation_component:.3f}, "
                       f"relevance={relevance_component:.3f}, importance={importance_component:.3f})")
            
            return self.unified_score
            
        except Exception as e:
            logger.error(f"Error updating unified score for '{self.token}': {e}")
            return self.unified_score
    
    def _update_confidence(self):
        """Update confidence based on reward variance and patterns."""
        try:
            if len(self.reward_history) >= 3:
                recent_rewards = self.reward_history[-3:]
                reward_variance = np.var(recent_rewards)
                
                # Calculate new confidence
                new_confidence = max(0.1, 1.0 - reward_variance)
                
                # Smooth transition
                self.confidence = 0.7 * self.confidence + 0.3 * new_confidence
                
                logger.debug(f"Confidence updated for '{self.token}': "
                           f"{len(self.reward_history)} rewards, variance={reward_variance:.4f}, "
                           f"confidence={self.confidence:.3f}")
            else:
                # Build confidence gradually with more data
                initial_confidence = 0.5
                data_confidence = min(len(self.reward_history) / 10.0, 0.3)
                self.confidence = initial_confidence + data_confidence
            
            # Store confidence history
            self.confidence_history.append(self.confidence)
            if len(self.confidence_history) > 100:
                self.confidence_history.pop(0)
            
            self.last_confidence_update = time.time()
            self.metadata['confidence_updates'] += 1
            
        except Exception as e:
            logger.error(f"Error updating confidence for '{self.token}': {e}")
    
    def _calculate_activation_component(self, reward: float) -> float:
        """Calculate activation component from trie dynamics."""
        try:
            # Confidence-weighted reward processing
            if reward < 0:
                punishment_strength = abs(reward) * (0.1 + 0.1 * self.confidence)
                reward_contribution = -punishment_strength
            else:
                reward_contribution = reward * (0.05 + 0.1 * self.confidence)
            
            # Apply decay
            decay = self.activation_level * self.metadata['decay_rate']
            
            # Calculate new activation
            new_activation = max(0.0, min(1.0, 
                self.activation_level + reward_contribution - decay))
            
            return new_activation
            
        except Exception as e:
            logger.error(f"Error calculating activation component: {e}")
            return self.activation_level
    
    def _calculate_relevance_component(self, context_embedding: np.ndarray = None, 
                                     query_embedding: np.ndarray = None) -> float:
        """Calculate relevance component from embedding similarities."""
        try:
            if self.embedding is None:
                return 0.0
            
            relevance_components = []
            
            # Context similarity
            if context_embedding is not None:
                context_sim = self._calculate_embedding_similarity(context_embedding, self.embedding)
                relevance_components.append(context_sim * 0.4)
            
            # Query similarity  
            if query_embedding is not None:
                query_sim = self._calculate_embedding_similarity(query_embedding, self.embedding)
                relevance_components.append(query_sim * 0.4)
            
            # Context-query bridge score
            if context_embedding is not None and query_embedding is not None:
                context_query_sim = self._calculate_embedding_similarity(context_embedding, query_embedding)
                bridge_score = (context_sim + query_sim) / 2.0 * (1.0 - context_query_sim) * 0.2
                relevance_components.append(bridge_score)
            
            # Frequency component
            freq_relevance = min(1.0, self.metadata['frequency'] / 100.0)
            relevance_components.append(freq_relevance * 0.1)
            
            # Confidence-weighted reward relevance
            reward_relevance = max(0.0, min(1.0, self.metadata['avg_reward']))
            confidence_weighted_reward = reward_relevance * self.confidence
            relevance_components.append(confidence_weighted_reward * 0.1)
            
            return sum(relevance_components) if relevance_components else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating relevance component: {e}")
            return 0.0
    
    def _calculate_importance_component(self) -> float:
        """Calculate importance component from usage patterns and memory factors."""
        try:
            # Access frequency importance
            access_score = min(5.0, np.log(self.access_count + 1))
            
            # Relationship importance
            relationship_score = min(3.0, len(self.related_nodes) * 0.1)
            
            # Recency importance
            time_since_access = time.time() - self.last_accessed
            recency_score = max(0.1, 2.0 / (1.0 + time_since_access / 86400))  # Decay over days
            
            # Cluster importance
            cluster_score = 0.5 if self.cluster_id else 0.0
            
            total_importance = access_score + relationship_score + recency_score + cluster_score
            return min(1.0, total_importance / 10.0)  # Normalize to [0,1]
            
        except Exception as e:
            logger.error(f"Error calculating importance component: {e}")
            return 0.5
    
    def _calculate_relationship_component(self) -> float:
        """Calculate relationship component from semantic connections."""
        try:
            if not self.related_nodes:
                return 0.0
            
            # Base relationship score
            relationship_density = len(self.related_nodes) / 100.0  # Normalize
            
            # Cluster membership bonus
            cluster_bonus = 0.2 if self.cluster_id else 0.0
            
            return min(1.0, relationship_density + cluster_bonus)
            
        except Exception as e:
            logger.error(f"Error calculating relationship component: {e}")
            return 0.0
    
    def _calculate_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate similarity between embeddings using ensemble method.

        CHANGES MADE:
        1. ADDED: Type checking and transformation for list inputs
        2. ADDED: Logging for input type detection and transformation
        3. PRESERVED: All existing similarity calculation logic
        4. ADDED: Robust error handling for type conversion edge cases

        JUSTIFICATION: Handles cases where emb1/emb2 are passed as lists of embeddings
        """
        try:
            # ADDED: Handle list input transformation for emb1
            if isinstance(emb1, list):
                logger.debug(f"emb1 detected as list with {len(emb1)} embeddings - transforming")
                if len(emb1) == 0:
                    logger.warning("emb1 is empty list - returning 0.0 similarity")
                    return 0.0
                elif len(emb1) == 1:
                    emb1 = emb1[0]
                    logger.debug("emb1 transformed from single-item list to array")
                else:
                    # Average multiple embeddings into single representative embedding
                    emb1 = np.mean(emb1, axis=0)
                    logger.debug(f"emb1 transformed from {len(emb1)} embeddings to averaged embedding")

            # ADDED: Handle list input transformation for emb2 (consistency)
            if isinstance(emb2, list):
                logger.debug(f"emb2 detected as list with {len(emb2)} embeddings - transforming")
                if len(emb2) == 0:
                    logger.warning("emb2 is empty list - returning 0.0 similarity")
                    return 0.0
                elif len(emb2) == 1:
                    emb2 = emb2[0]
                    logger.debug("emb2 transformed from single-item list to array")
                else:
                    # Average multiple embeddings into single representative embedding
                    emb2 = np.mean(emb2, axis=0)
                    logger.debug(f"emb2 transformed from {len(emb2)} embeddings to averaged embedding")

            # ADDED: Validate transformed inputs are now np.ndarray
            if not isinstance(emb1, np.ndarray):
                logger.error(f"emb1 is not np.ndarray after transformation: {type(emb1)}")
                return 0.0

            if not isinstance(emb2, np.ndarray):
                logger.error(f"emb2 is not np.ndarray after transformation: {type(emb2)}")
                return 0.0

            # PRESERVED: Original None and zero norm checks (unchanged)
            if emb1 is None or emb2 is None:
                return 0.0

            if np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
                return 0.0

            # ADDED: Dimension compatibility check after transformation
            if emb1.shape != emb2.shape:
                logger.error(f"Embedding dimension mismatch: emb1.shape={emb1.shape}, emb2.shape={emb2.shape}")
                return 0.0

            # PRESERVED: Original similarity calculation logic (unchanged)
            # Use context window's ensemble similarity if available
            if self.context_window and hasattr(self.context_window, '_calculate_ensemble_similarity'):
                similarity = float(self.context_window._calculate_ensemble_similarity(emb1, emb2))
                logger.debug(f"Calculated ensemble similarity: {similarity:.4f}")
                return similarity
            else:
                # Fallback to cosine similarity
                similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                logger.debug(f"Calculated cosine similarity: {similarity:.4f}")
                return similarity

        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}")
            logger.error(f"Input types: emb1={type(emb1)}, emb2={type(emb2)}")
            if hasattr(emb1, 'shape'):
                logger.error(f"emb1 shape: {emb1.shape}")
            if hasattr(emb2, 'shape'):
                logger.error(f"emb2 shape: {emb2.shape}")
            return 0.0

    def add_semantic_relationship(self, other_node_id: str, similarity_score: float = 0.0):
        """Add semantic relationship to another node."""
        try:
            self.related_nodes.add(other_node_id)
            logger.debug(f"Added semantic relationship: {self.node_id} -> {other_node_id} "
                        f"(similarity: {similarity_score:.3f})")
        except Exception as e:
            logger.error(f"Error adding semantic relationship: {e}")
    
    def set_cluster(self, cluster_id: str):
        """Assign node to semantic cluster."""
        try:
            old_cluster = self.cluster_id
            self.cluster_id = cluster_id
            logger.debug(f"Node '{self.token}' moved from cluster '{old_cluster}' to '{cluster_id}'")
        except Exception as e:
            logger.error(f"Error setting cluster: {e}")
    
    # EMBEDDING MANAGEMENT (from TokenEmbedding and EmbeddingNode)
    def set_embedding(self, embedding: np.ndarray, binary_values: List[int] = None, 
                     ascii_values: List[int] = None, subword_tokens: List[str] = None,
                     semantic_category: str = None):
        """Set complete embedding information."""
        try:
            self.embedding = embedding.copy() if embedding is not None else None
            self._embedding_loaded = True
            
            if binary_values is not None:
                self.binary_values = binary_values
            if ascii_values is not None:
                self.ascii_values = ascii_values
            if subword_tokens is not None:
                self.subword_tokens = subword_tokens
            if semantic_category is not None:
                self.semantic_category = semantic_category
            
            logger.debug(f"Set complete embedding for '{self.token}' "
                        f"(dims: {embedding.shape if embedding is not None else 'None'})")
        except Exception as e:
            logger.error(f"Error setting embedding: {e}")
    
    def set_embedding_info(self, embedding_key: str):
        """Set up lazy loading for embedding."""
        try:
            self._embedding_key = embedding_key
            self._embedding_loaded = False
            self.embedding = None
            logger.debug(f"Set lazy loading for embedding key: {embedding_key}")
        except Exception as e:
            logger.error(f"Error setting embedding info: {e}")
    
    def _load_embedding(self):
        """Load embedding from database with error handling."""
        try:
            if not self._db_env or not self._embedding_key:
                logger.warning(f"Cannot load embedding: missing database or key for '{self.token}'")
                self._embedding_loaded = True
                return
            
            with self._db_env.begin() as txn:
                value = txn.get(self._embedding_key.encode())
                if value:
                    data = msgpack.unpackb(value, raw=False, strict_map_key=False)
                    embedding_bytes = data.get('embedding')
                    if embedding_bytes:
                        self.embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                        logger.debug(f"Loaded embedding for '{self.token}' from key: {self._embedding_key}")
            
            self._embedding_loaded = True
            
        except Exception as e:
            logger.warning(f"Failed to load embedding for '{self.token}': {e}")
            self._embedding_loaded = True
    
    @property
    def token_embedding(self) -> Optional[np.ndarray]:
        """Lazy load embedding on first access (backward compatibility)."""
        if not self._embedding_loaded and self._embedding_key and self._db_env:
            self._load_embedding()
        return self.embedding
    
    @token_embedding.setter  
    def token_embedding(self, value):
        """Set embedding directly (backward compatibility)."""
        self.embedding = value
        self._embedding_loaded = True
    
    def add_child(self, token: str, child: 'SemanticTrieNode'):
        if token in self.children:
            existing_child = self.children[token]
            if existing_child != child:
                logger.debug(f"Updating child for token '{token}' in node '{self.token}'")
        
        self.children[token] = child  # ✅ Single node per token
        logger.debug(f"Added child '{child.token}' for token '{token}' in node '{self.token}'")
    
    
    def get_child(self, token: str) -> Optional['SemanticTrieNode']:
        """Get child node by token."""
        return self.children.get(token)
    
    def update_completeness(self, heuristic_rules: Dict[str, Any] = None):
        """Update completeness based on token characteristics."""
        try:
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
                
        except Exception as e:
            logger.error(f"Error updating completeness: {e}")
    
    def decay_activation(self):
        """Apply natural decay to activation level."""
        try:
            old_activation = self.activation_level
            self.activation_level *= (1.0 - self.metadata['decay_rate'])
            self.activation_level = max(0.0, self.activation_level)
            
            logger.debug(f"Applied decay to '{self.token}': {old_activation:.3f} -> {self.activation_level:.3f}")
        except Exception as e:
            logger.error(f"Error applying decay: {e}")
    
    def boost_activation(self, boost_factor: float = 0.1):
        """Boost activation level for spreading activation."""
        try:
            old_activation = self.activation_level
            self.activation_level = min(1.0, self.activation_level + float(boost_factor))
            
            logger.debug(f"Boosted activation for '{self.token}': {old_activation:.3f} -> {self.activation_level:.3f}")
        except Exception as e:
            logger.error(f"Error boosting activation: {e}")
    
    # NODE REGISTRY METHODS (from TrieNode)
    def _init_node_registry(self):
        """Initialize the node registry (only called on root)."""
        try:
            self.node_registry = {
                'by_id': {},           # node_id -> node
                'by_path': {},         # path_string -> node  
                'by_level': {},        # level -> list of nodes
                'by_token_path': {},   # tuple(tokens) -> node
                'by_token': {},        # token -> list of nodes
                'embedding_keys': {}   # embedding_key -> node
            }
            logger.info("Initialized unified node registry on root node")
        except Exception as e:
            logger.error(f"Error initializing node registry: {e}")
    
    def register_node(self, node: 'SemanticTrieNode', path_tokens: List[str]):
        """Register a node in the root's registry."""
        try:
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
            
            # Token-based lookup
            if node.token:
                if node.token not in self.node_registry['by_token']:
                    self.node_registry['by_token'][node.token] = []
                self.node_registry['by_token'][node.token].append(node)
            
            # Embedding key registration
            if node._embedding_key:
                self.node_registry['embedding_keys'][node._embedding_key] = node
            
            logger.debug(f"Registered node {node.node_id[:8]}... at path: {path_tokens} (level {level})")
            
        except Exception as e:
            logger.error(f"Error registering node: {e}")
    
    def get_node_by_path(self, path_tokens: List[str]) -> Optional['SemanticTrieNode']:
        """Get node directly by path tokens (call on root)."""
        try:
            if self.node_registry is None:
                logger.error("get_node_by_path called on non-root node")
                return None
            path_tuple = tuple(path_tokens)
            return self.node_registry['by_token_path'].get(path_tuple)
        except Exception as e:
            logger.error(f"Error getting node by path: {e}")
            return None
    
    def get_node_by_id(self, node_id: str) -> Optional['SemanticTrieNode']:
        """Get node directly by ID (call on root)."""
        try:
            if self.node_registry is None:
                logger.error("get_node_by_id called on non-root node")
                return None
            return self.node_registry['by_id'].get(node_id)
        except Exception as e:
            logger.error(f"Error getting node by ID: {e}")
            return None
    
    def get_nodes_by_token(self, token: str) -> List['SemanticTrieNode']:
        """Get all nodes with a specific token."""
        try:
            if self.node_registry is None:
                return []
            return self.node_registry['by_token'].get(token, [])
        except Exception as e:
            logger.error(f"Error getting nodes by token: {e}")
            return []
    
    def get_nodes_at_level(self, level: int) -> List['SemanticTrieNode']:
        """Get all nodes at specific hierarchy level (call on root)."""
        try:
            if self.node_registry is None:
                logger.error("get_nodes_at_level called on non-root node")
                return []
            return self.node_registry['by_level'].get(level, [])
        except Exception as e:
            logger.error(f"Error getting nodes at level: {e}")
            return []
    
    def get_all_registered_nodes(self) -> List['SemanticTrieNode']:
        """Get all nodes in the registry (call on root)."""
        try:
            if self.node_registry is None:
                return []
            return list(self.node_registry['by_id'].values())
        except Exception as e:
            logger.error(f"Error getting all registered nodes: {e}")
            return []
    
    # INFORMATION AND DEBUGGING METHODS
    def get_unified_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the node."""
        try:
            return {
                # Core identity
                'node_id': self.node_id,
                'embedding_id': self.embedding_id,
                'token': self.token,
                
                # Embedding information
                'has_embedding': self.embedding is not None,
                'embedding_shape': self.embedding.shape if self.embedding is not None else None,
                'semantic_category': self.semantic_category,
                'subword_count': len(self.subword_tokens),
                
                # Scoring information
                'unified_score': self.unified_score,
                'confidence': self.confidence,
                'activation_level': self.activation_level,
                'relevance_score': self.relevance_score,
                
                # Access patterns
                'access_count': self.access_count,
                'last_accessed': self.last_accessed,
                'creation_time': self.creation_time,
                
                # Relationships
                'children_count': len(self.children),
                'children_tokens': list(self.children.keys()),
                'related_nodes_count': len(self.related_nodes),
                'cluster_id': self.cluster_id,
                
                # Trie structure
                'hierarchy_level': self.hierarchy_level,
                'path_tokens': self.path_tokens,
                'is_end_of_sequence': self.is_end_of_sequence,
                
                # Metadata
                'metadata': self.metadata.copy(),
                'reward_history_length': len(self.reward_history),
                'confidence_history_length': len(self.confidence_history)
            }
        except Exception as e:
            logger.error(f"Error getting unified info: {e}")
            return {'error': str(e)}
    
    def get_confidence_info(self) -> Dict[str, Any]:
        """Get detailed confidence information."""
        try:
            confidence_stats = {
                'current_confidence': self.confidence,
                'confidence_trend': 'stable',
                'confidence_updates': self.metadata.get('confidence_updates', 0),
                'last_update': self.last_confidence_update,
                'confidence_history_length': len(self.confidence_history)
            }
            
            # Calculate confidence trend
            if len(self.confidence_history) >= 5:
                recent_conf = self.confidence_history[-5:]
                if recent_conf[-1] > recent_conf[0] + 0.1:
                    confidence_stats['confidence_trend'] = 'increasing'
                elif recent_conf[-1] < recent_conf[0] - 0.1:
                    confidence_stats['confidence_trend'] = 'decreasing'
            
            # Add reward statistics
            if self.reward_history:
                confidence_stats.update({
                    'reward_variance': float(np.var(self.reward_history[-10:])) if len(self.reward_history) >= 3 else 0.0,
                    'recent_avg_reward': float(np.mean(self.reward_history[-5:])) if len(self.reward_history) >= 5 else 0.0,
                    'negative_feedback_ratio': self.metadata['negative_feedback_count'] / max(1, self.metadata['frequency'])
                })
            
            return confidence_stats
        except Exception as e:
            logger.error(f"Error getting confidence info: {e}")
            return {'error': str(e)}
    
    def debug_registry(self):
        """Debug the node registry (call on root)."""
        try:
            if self.node_registry is None:
                print("❌ No registry (not called on root)")
                return
            
            total_nodes = len(self.node_registry['by_id'])
            max_level = max(self.node_registry['by_level'].keys()) if self.node_registry['by_level'] else 0
            
            print(f"\n📊 UNIFIED NODE REGISTRY DEBUG:")
            print(f"Total registered nodes: {total_nodes}")
            print(f"Maximum hierarchy level: {max_level}")
            print(f"Registry sizes:")
            print(f"  by_id: {len(self.node_registry['by_id'])}")
            print(f"  by_path: {len(self.node_registry['by_path'])}")
            print(f"  by_token_path: {len(self.node_registry['by_token_path'])}")
            print(f"  by_token: {len(self.node_registry['by_token'])}")
            print(f"  embedding_keys: {len(self.node_registry['embedding_keys'])}")
            
            print(f"\nNodes per level:")
            for level in sorted(self.node_registry['by_level'].keys()):
                count = len(self.node_registry['by_level'][level])
                print(f"  Level {level}: {count} nodes")
                
        except Exception as e:
            logger.error(f"Error debugging registry: {e}")
            print(f"❌ Error debugging registry: {e}")
    
    # BACKWARD COMPATIBILITY METHODS
    def calculate_relevance(self, context_embedding: np.ndarray = None, 
                          query_embedding: np.ndarray = None) -> float:
        """Backward compatibility: calculate relevance component."""
        relevance = self._calculate_relevance_component(context_embedding, query_embedding)
        self.relevance_score = relevance
        return relevance
    
    def update_activation(self, reward: float = 0.0, context_relevance: float = 0.0):
        """Backward compatibility: update activation (calls unified scoring)."""
        # Convert context_relevance to context_embedding if needed
        context_embedding = None
        if context_relevance > 0 and self.context_window:
            context_embedding = getattr(self.context_window, 'current_context_embedding', None)
        
        self.update_unified_score(reward, context_embedding)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Backward compatibility: calculate similarity between embeddings."""
        return self._calculate_embedding_similarity(embedding1, embedding2)
    
    def calculate_token_similarity(self, other_embedding: np.ndarray) -> float:
        """Backward compatibility: compare token embedding with another embedding."""
        if self.embedding is None or other_embedding is None:
            return 0.0
        return self._calculate_embedding_similarity(self.embedding, other_embedding)
    
    # STRING REPRESENTATION
    def __repr__(self):
        """String representation for debugging."""
        return (f"SemanticTrieNode(id={self.node_id[:8]}..., token='{self.token}', "
                f"level={self.hierarchy_level}, children={len(self.children)}, "
                f"unified_score={self.unified_score:.3f})")
    
    def __str__(self):
        """Human-readable string representation."""
        return f"'{self.token}' (unified_score: {self.unified_score:.3f}, confidence: {self.confidence:.3f})"


def create_semantic_trie_node(token: str, db_env, context_window: ContextWindow) -> SemanticTrieNode:
    """
    Factory function to create a SemanticTrieNode with full embedding.
    
    REPLACES:
    - create_token_embedding()
    - TokenEmbedding creation
    - EmbeddingNode creation
    
    Creates unified node with complete 4096-dimensional embedding.
    """
    logger.info(f"Creating semantic trie node with full embedding for: '{token}'")
    
    try:
        # Generate embedding using the enhanced 4096-dimensional algorithm
        embedding, binary_values, ascii_values, subword_tokens, semantic_category = _create_full_embedding(token)
        
        # Create unified node
        node = SemanticTrieNode(token=token, embedding=embedding, db_env=db_env, context_window=context_window)

        # Set complete embedding information
        node.set_embedding(embedding, binary_values, ascii_values, subword_tokens, semantic_category)
        
        logger.info(f"Created semantic trie node: '{token}' (category: {semantic_category}, "
                   f"subwords: {len(subword_tokens)})")
        
        return node
        
    except Exception as e:
        logger.error(f"Error creating semantic trie node for '{token}': {e}")
        raise
    
def add_children_to_node(node: SemanticTrieNode, children: List[SemanticTrieNode]):
    """
    Add multiple children to a SemanticTrieNode.
    
    REPLACES:
    - TrieNode.add_child() for multiple children
    """
    try:
        for child in children:
            if isinstance(child, SemanticTrieNode):
                node.add_child(child.token, child)
                logger.debug(f"Added child node '{child.token}' to parent '{node.token}'")
            else:
                logger.warning(f"Child is not a SemanticTrieNode: {child}")
    except Exception as e:
        logger.error(f"Error adding children to node '{node.token}': {e}")


def _create_full_embedding(token: str) -> Tuple[np.ndarray, List[int], List[int], List[str], str]:
    """
    Create complete 4096-dimensional embedding with all metadata.
    
    UNIFIED FROM: create_token_embedding() functionality
    """
    # Input validation
    if not isinstance(token, str):
        raise TypeError("Token must be a string")
    if not token:
        raise ValueError("Token cannot be empty")
    
    # Extract binary and ASCII values
    binary_values = []
    ascii_values = []
    
    for char in token:
        ascii_val = ord(char)
        binary_val = bin(ascii_val)[2:]
        ascii_values.append(ascii_val)
        binary_values.extend([int(bit) for bit in binary_val])
    
    # Get semantic analysis
    semantic_category = _get_semantic_category(token)
    subword_tokens = _simple_subword_tokenize(token)
    
    # Generate deterministic seed
    token_hash = hashlib.md5(token.encode()).hexdigest()
    seed = int(token_hash[:8], 16)
    np.random.seed(seed)
    
    # Create 4096-dimensional embedding (preserved algorithm)
    embedding = np.zeros(4096, dtype=np.float32)
    
    # SECTION 1 (0-511): ASCII/Binary Features
    for i, ascii_val in enumerate(ascii_values[:256]):
        if i < 256:
            embedding[i] = ascii_val / 127.0
    
    binary_chunk_size = max(1, len(binary_values) // 256)
    for i in range(256):
        start_idx = i * binary_chunk_size
        end_idx = min(start_idx + binary_chunk_size, len(binary_values))
        if start_idx < len(binary_values):
            chunk = binary_values[start_idx:end_idx]
            if chunk:
                avg_value = sum(chunk) / len(chunk)
                pattern_complexity = len(set(chunk)) / len(chunk)
                embedding[256 + i] = avg_value * (1 + 0.1 * pattern_complexity)
    
    # SECTION 2 (512-1023): Character Traits
    embedding[512] = len(token) / 100.0
    embedding[513] = sum(c.isupper() for c in token) / len(token)
    embedding[514] = sum(c.islower() for c in token) / len(token)
    embedding[515] = sum(c.isdigit() for c in token) / len(token)
    embedding[516] = sum(c.isalpha() for c in token) / len(token)
    embedding[517] = sum(c in string.punctuation for c in token) / len(token)
    embedding[518] = sum(c.isspace() for c in token) / len(token)
    embedding[519] = len(set(token)) / len(token)
    embedding[520] = sum(ord(c) for c in token) / len(token) / 127.0
    
    # Character position patterns (521-567)
    for i in range(521, 568):
        pos_factor = (i - 521) / 47.0
        char_idx = int(pos_factor * len(token)) if token else 0
        if char_idx < len(token):
            embedding[i] = ord(token[char_idx]) / 127.0
    
    # Linguistic patterns (568-1023)
    for i in range(568, 1024):
        pos_factor = (i - 568) / 456.0
        if ascii_values:
            base_val = ascii_values[0] if ascii_values else 65
            embedding[i] = np.sin(base_val * pos_factor * np.pi) * np.exp(-pos_factor)
    
    # SECTION 3 (1024-2047): Contextual Dependencies
    # Subword features (1024-1279)
    for i, subword in enumerate(subword_tokens[:16]):
        base_idx = 1024 + i * 16
        if base_idx + 15 < 2048:
            subword_hash = hashlib.md5(subword.encode()).hexdigest()
            subword_seed = int(subword_hash[:4], 16)
            
            embedding[base_idx] = len(subword) / 20.0
            embedding[base_idx + 1] = sum(c.isalpha() for c in subword) / len(subword)
            embedding[base_idx + 2] = (subword_seed % 1000) / 1000.0
            embedding[base_idx + 3] = i / len(subword_tokens)
            
            for j in range(4, 16):
                if j < len(subword):
                    embedding[base_idx + j] = ord(subword[j]) / 127.0
    
    # Morphological patterns (1280-1535)
    morphology_features = np.zeros(256)
    morphological_markers = {
        'prefix': ['un', 're', 'pre', 'dis', 'over', 'under'],
        'suffix': ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness'],
        'compound': ['-', '_', '.'],
        'case_pattern': [token.isupper(), token.islower(), token.istitle()]
    }
    
    feature_idx = 0
    for pattern_type, markers in morphological_markers.items():
        if pattern_type == 'case_pattern':
            for case_bool in markers:
                if feature_idx < 256:
                    morphology_features[feature_idx] = float(case_bool)
                    feature_idx += 1
        else:
            for marker in markers:
                if feature_idx < 256:
                    if marker in token.lower():
                        morphology_features[feature_idx] = 1.0
                    feature_idx += 1
    
    embedding[1280:1536] = morphology_features
    
    # Trie features (1536-2047)
    trie_features = np.zeros(512)
    trie_features[0] = 1.0 if token[0].isupper() else 0.0
    trie_features[1] = 1.0 if token[-1] in '.!?' else 0.0
    trie_features[2] = 1.0 if any(c in token for c in '()[]{}') else 0.0
    
    for i in range(3, 512):
        pattern_seed = (seed + i) % 10000
        trie_features[i] = np.sin(pattern_seed / 10000.0 * np.pi) * 0.05
    
    embedding[1536:2048] = trie_features
    
    # SECTION 4 (2048-3071): Semantic Flow
    # Category encoding (2048-2111)
    category_encoding = np.zeros(64)
    category_map = {
        'numeric': 0, 'decimal': 1, 'alphanumeric': 2, 'uppercase': 3,
        'lowercase': 4, 'titlecase': 5, 'mixedcase': 6, 'sentence_end': 7,
        'clause_separator': 8, 'bracket': 9, 'quote': 10, 'punctuation': 11,
        'whitespace': 12, 'mixed': 13, 'empty': 14, 'unknown': 15
    }
    
    if semantic_category in category_map:
        category_idx = category_map[semantic_category]
        category_encoding[category_idx] = 1.0
        
        for i in range(16, 64):
            pattern_factor = (i - 16) / 48.0
            category_encoding[i] = np.sin(category_idx * pattern_factor * np.pi) * np.exp(-pattern_factor * 0.5)
    
    embedding[2048:2112] = category_encoding
    
    # Semantic features (2112-2623)
    semantic_features = np.zeros(512)
    
    if semantic_category in ['numeric', 'decimal', 'alphanumeric']:
        for i in range(512):
            numeric_factor = sum(int(c) for c in token if c.isdigit())
            semantic_features[i] = np.sin(numeric_factor * (i + 1) / 512.0) * 0.1
    elif semantic_category in ['uppercase', 'lowercase', 'titlecase', 'mixedcase']:
        for i in range(512):
            text_hash = sum(ord(c) for c in token.lower())
            semantic_features[i] = np.cos(text_hash * (i + 1) / 512.0) * 0.1
    else:
        for i in range(512):
            general_hash = hash(token + semantic_category) % 10000
            semantic_features[i] = np.sin(general_hash * (i + 1) / 512.0) * 0.1
    
    embedding[2112:2624] = semantic_features
    
    # Context features (2624-3071)
    context_features = np.zeros(448)
    for i in range(448):
        pattern_seed = (seed + i + 1000) % 10000
        context_features[i] = np.sin(pattern_seed / 10000.0 * np.pi) * 0.05
    
    embedding[2624:3072] = context_features
    
    # SECTION 5 (3072-4095): Regularization
    regularization_features = np.random.normal(0, 0.05, 1024)
    for i in range(1024):
        token_influence = ascii_values[i % len(ascii_values)] / 127.0 if ascii_values else 0
        regularization_features[i] *= (1 + 0.1 * token_influence)
    
    embedding[3072:4096] = regularization_features
    
    # Normalize to unit length
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding, binary_values, ascii_values, subword_tokens, semantic_category


def _get_semantic_category(token: str) -> str:
    """Classify tokens into semantic categories."""
    if not token:
        return "empty"
    
    if re.match(r'^\d+$', token):
        return "numeric"
    elif re.match(r'^\d*\.\d+$', token):
        return "decimal"
    elif re.match(r'^\d+[a-zA-Z]*$', token):
        return "alphanumeric"
    elif token.isalpha():
        if token.isupper():
            return "uppercase"
        elif token.islower():
            return "lowercase"
        elif token.istitle():
            return "titlecase"
        else:
            return "mixedcase"
    elif all(c in string.punctuation for c in token):
        if token in '.!?':
            return "sentence_end"
        elif token in ',:;':
            return "clause_separator"
        elif token in '()[]{}':
            return "bracket"
        elif token in '"\'`':
            return "quote"
        else:
            return "punctuation"
    elif token.isspace():
        return "whitespace"
    else:
        return "mixed"


def _simple_subword_tokenize(token: str, max_subwords: int = 4) -> List[str]:
    """Simple subword tokenization for morphological analysis."""
    if len(token) <= 3:
        return [token]
    
    subwords = []
    prefixes = ['un', 're', 'pre', 'dis', 'over', 'under', 'out', 'up']
    suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment']
    
    remaining = token.lower()
    
    # Extract prefixes
    for prefix in prefixes:
        if remaining.startswith(prefix) and len(remaining) > len(prefix) + 2:
            subwords.append(prefix)
            remaining = remaining[len(prefix):]
            break
    
    # Extract suffixes
    for suffix in suffixes:
        if remaining.endswith(suffix) and len(remaining) > len(suffix) + 2:
            subwords.append(remaining[:-len(suffix)])
            subwords.append(suffix)
            remaining = ""
            break
    
    # Add remaining core
    if remaining and remaining not in subwords:
        subwords.append(remaining)
    
    # Fallback to syllable splitting
    if len(subwords) <= 1:
        subwords = []
        vowels = 'aeiou'
        syllable_boundaries = []
        
        for i in range(1, len(token)):
            if (token[i-1] not in vowels and token[i] in vowels) or \
               (token[i-1] in vowels and token[i] not in vowels and i < len(token) - 1):
                syllable_boundaries.append(i)
        
        if syllable_boundaries:
            start = 0
            for boundary in syllable_boundaries[:max_subwords-1]:
                if boundary > start:
                    subwords.append(token[start:boundary])
                start = boundary
            if start < len(token):
                subwords.append(token[start:])
        else:
            chunk_size = max(2, len(token) // max_subwords)
            subwords = [token[i:i+chunk_size] for i in range(0, len(token), chunk_size)]
    
    # Ensure limits
    subwords = subwords[:max_subwords]
    if not subwords:
        subwords = [token]
    
    return subwords