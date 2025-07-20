"""
REFACTORED: Streamlined TrieMemory with registry removal and essential methods only.

ACCOUNTABILITY CHANGES:
1. REMOVED: All registry functionality (self.root, node_registry)
2. REMOVED: Unused debug methods and complex traversal methods
3. PRESERVED: Core functionality (embeddings, learn_sequence, add_sequence, find_best_continuation)
4. SIMPLIFIED: Direct embeddings-based operations only
5. MAINTAINED: All existing logging and error handling
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import gc
import msgpack
import numpy as np
import logging
import lmdb
import time
from typing import List, Dict, Tuple, Optional, Any
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from context_window import ContextWindow
from trie_node import SemanticTrieNode, create_semantic_trie_node, _create_full_embedding
from utils import _calculate_safe_cosine_similarity, search_embeddings_chunk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REWARD_HISTORY_LIMIT = 10000

class TrieMemory:
    """
    STREAMLINED: Essential TrieMemory functionality without registry complexity.
    
    ACCOUNTABILITY CHANGES:
    - REMOVED: self.root and all registry operations
    - PRESERVED: self.embeddings as single source of truth
    - SIMPLIFIED: Direct token-based operations only
    - MAINTAINED: All core learning and prediction functionality
    """
    
    def __init__(self, db_path: str = "./trie_memory.lmdb", embed_dim: int = 4096):
        """
        SIMPLIFIED: Initialize TrieMemory with embeddings-only approach.
        
        ACCOUNTABILITY CHANGES:
        1. REMOVED: self.root initialization and registry setup
        2. PRESERVED: Database, context window, and embeddings initialization
        3. MAINTAINED: All existing database configuration and loading
        """
        logger.info("Initializing streamlined TrieMemory with embeddings-only approach")
        
        # Core configuration
        self.embed_dim = embed_dim
        self.db_path = db_path
        self.context_window = ContextWindow()
        
        # SINGLE SOURCE OF TRUTH: embeddings dictionary only
        self.embeddings = {}  # Dict[str, SemanticTrieNode] - hybrid flat/hierarchical structure
        
        # Hardware acceleration configuration
        self.acceleration_mode = 'auto'
        self.max_cpu_processes = min(16, mp.cpu_count())
        self.gpu_batch_size = 1000
        
        # Database setup
        initial_map_size = self._calculate_adaptive_map_size()
        logger.info(f"Initializing LMDB with adaptive map size: {initial_map_size / (1024*1024*1024):.1f}GB")
        
        try:
            self.env = lmdb.open(
                db_path, 
                map_size=initial_map_size,
                max_dbs=2,
                writemap=True,  
                map_async=True  
            )
            self.nodes_db = self.env.open_db(b'nodes')
            self.metadata_db = self.env.open_db(b'metadata')
            
            # Database monitoring
            self.current_map_size = initial_map_size
            self.resize_threshold = 0.8  
            self.resize_factor = 2.0
            
            logger.info("Successfully initialized LMDB database")
            
        except Exception as e:
            logger.error(f"Failed to initialize LMDB: {e}")
            raise
        
        # Load existing data
        try:
            self._load_embeddings_only()
            logger.info(f"TrieMemory initialization complete: {len(self.embeddings)} tokens loaded")
            
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            logger.info("Starting with empty embeddings dictionary")
    
    def learn_sequence(self, tokens: List[str], reward: float = 1.0) -> np.ndarray:
        """
        PRESERVED: Learn sequence with embeddings-based approach.
        
        ACCOUNTABILITY: Unchanged functionality, uses embeddings instead of registry.
        """
        logger.info(f"Learning sequence: {len(tokens)} tokens with reward {reward}")
        
        try:
            # Get token embeddings and establish children relationships
            token_embeddings = self._get_token_embeddings(tokens)
            
            if not token_embeddings:
                logger.warning("No token embeddings generated - sequence learning aborted")
                return np.zeros(self.embed_dim, dtype=np.float32)
            
            # Calculate sequence embedding
            embeddings_array = np.array([embedding for embedding in token_embeddings])
            sequence_embedding = np.mean(embeddings_array, axis=0)
            
            # Normalize and add to context
            norm = np.linalg.norm(sequence_embedding)
            if norm > 0:
                sequence_embedding = sequence_embedding / norm
            
            self.context_window.add_turn(tokens, sequence_embedding)
            
            # Store each token with context information
            current_timestamp = time.time()
            sequence_embedding_bytes = sequence_embedding.tobytes()
            
            for i, (token, token_embedding) in enumerate(zip(tokens, token_embeddings)):
                context_info = {
                    'sequence': tokens,
                    'sequence_embedding': sequence_embedding_bytes,
                    'position': i,
                    'reward': reward,
                    'timestamp': current_timestamp,
                    'source': 'trie_sequence'
                }
                
                self.add_embedding(token=token, embedding=token_embedding, context_info=context_info)
            
            logger.info(f"Successfully learned sequence: {len(tokens)} tokens")
            return sequence_embedding
            
        except Exception as e:
            logger.error(f"Error in learn_sequence: {e}")
            return np.zeros(self.embed_dim, dtype=np.float32)
    
    def add_sequence(self, tokens: List[str], reward: float = 0.0) -> str:
        """
        SIMPLIFIED: Add sequence using embeddings-only approach.
        
        ACCOUNTABILITY CHANGES:
        1. REMOVED: Registry operations and root traversal
        2. PRESERVED: Token embedding creation and children relationship building
        3. MAINTAINED: Database persistence and reward processing
        """
        logger.info(f"Adding sequence to embeddings: {tokens} with reward: {reward}")
        
        try:
            # Get token embeddings with children relationships
            token_embeddings = self._get_token_embeddings(tokens)
            
            if not token_embeddings:
                logger.error("Failed to create token embeddings - aborting sequence addition")
                return ""
            
            # Process rewards and build children relationships
            negative_feedback_applied = reward < 0
            if negative_feedback_applied:
                logger.info(f"Applying negative feedback (reward: {reward}) through sequence")
            
            # Update activations and establish children relationships
            for i, token in enumerate(tokens):
                node = self.embeddings[token]
                
                # Calculate position-weighted reward
                position_weight = (i + 1) / len(tokens)
                adjusted_reward = reward * (0.5 + 0.5 * position_weight) if negative_feedback_applied else reward * position_weight
                
                # Update node activation
                context_relevance = self.context_window.get_context_similarity(node.embedding)
                node.update_activation(adjusted_reward, context_relevance)
                
                if negative_feedback_applied:
                    logger.debug(f"Applied negative feedback to '{token}': "
                               f"original_reward={reward:.3f}, adjusted={adjusted_reward:.3f}")
            
            # Mark end of sequence
            if tokens:
                final_node = self.embeddings[tokens[-1]]
                final_node.is_end_of_sequence = True
                final_node.update_completeness()
            
            # Save to database
            sequence_id = self._generate_sequence_id(tokens)
            path_nodes = [self.embeddings[token] for token in tokens]
            sequence_embeddings = [node.embedding for node in path_nodes]
            sequence_embedding = np.mean(sequence_embeddings, axis=0)
            
            self._save_to_db(sequence_id, path_nodes, sequence_embedding)
            
            logger.info(f"Successfully added sequence with ID: {sequence_id[:8]}...")
            return sequence_id
            
        except Exception as e:
            logger.error(f"Error adding sequence: {e}")
            raise
    
    def find_best_continuation(self, query_tokens: List[str], 
                              context_embedding: np.ndarray,
                              query_sequence_embedding: np.ndarray,
                              max_candidates: int = 10, 
                              max_continuations: int = 500) -> Tuple[List[str], float]:
        """
        STREAMLINED: Find best continuation using direct embeddings lookup.
        
        ACCOUNTABILITY CHANGES:
        1. REMOVED: Registry-based node lookup
        2. SIMPLIFIED: Direct embeddings access for token matching
        3. PRESERVED: All scoring, confidence tracking, and candidate collection logic
        """
        logger.info(f"Finding best continuation for: {query_tokens} (max_candidates={max_candidates})")
        
        if not query_tokens:
            logger.warning("Empty query tokens provided")
            return [], 0.0
        
        try:
            # Direct token lookup from embeddings
            matched_nodes = []
            matched_tokens = []
            match_confidences = []
            
            for token in query_tokens:
                if token in self.embeddings:
                    node = self.embeddings[token]
                    matched_nodes.append(node)
                    matched_tokens.append(token)
                    node_confidence = getattr(node, 'confidence', 0.5)
                    match_confidences.append(node_confidence)
                    logger.debug(f"Matched token: '{token}' (confidence: {node_confidence:.3f})")
                else:
                    logger.debug(f"Token not found: '{token}'")
                    break
            
            matched_length = len(matched_nodes)
            logger.info(f"Token match result: {matched_length}/{len(query_tokens)} tokens matched")
            
            if matched_length == 0:
                logger.warning("No tokens found in embeddings")
                return [], 0.0
            
            # Calculate match confidence quality
            avg_match_confidence = sum(match_confidences) / len(match_confidences)
            min_match_confidence = min(match_confidences)
            logger.info(f"Match confidence: avg={avg_match_confidence:.3f}, min={min_match_confidence:.3f}")
            
            # Collect continuations from last matched node
            last_matched_node = matched_nodes[-1]
            logger.info(f"Collecting continuations from token '{last_matched_node.token}'")
            
            continuation_candidates = []
            self._collect_continuations(
                last_matched_node, matched_tokens, continuation_candidates, 
                query_sequence_embedding, max_continuations, 
                context_embedding=context_embedding
            )
            
            logger.info(f"Collected {len(continuation_candidates)} continuation candidates")
            
            if not continuation_candidates:
                logger.warning("No continuation candidates found")
                return [], 0.0
            
            # Score candidates with confidence consideration
            scored_candidates = []
            for path, base_score in continuation_candidates:
                # Add confidence bonuses
                path_confidence_bonus = avg_match_confidence * 0.05
                length_bonus = 0.1 * matched_length
                sequential_bonus = 0.02 * len(path)
                
                final_score = base_score + length_bonus + path_confidence_bonus + sequential_bonus
                scored_candidates.append((path, final_score))
                
                logger.debug(f"Scored candidate: {path} -> final_score={final_score:.3f}")
            
            # Filter viable candidates
            viability_threshold = max(0.01, min_match_confidence * 0.02)
            viable_candidates = [(cont, score) for cont, score in scored_candidates if score > viability_threshold]
            
            logger.info(f"Viability check: {len(viable_candidates)}/{len(scored_candidates)} candidates viable")
            
            if not viable_candidates:
                logger.warning("No viable candidates above threshold")
                return [], 0.0
            
            # Select best candidate
            best_continuation, best_score = max(viable_candidates, key=lambda x: x[1])
            
            logger.info(f"Best continuation selected: {best_continuation} (score: {best_score:.3f})")
            return best_continuation, best_score
            
        except Exception as e:
            logger.error(f"Error in find_best_continuation: {e}")
            return [], 0.0
    
    def _get_embedding_for_token(self, token: str) -> SemanticTrieNode:
        """
        PRESERVED: Get or create embedding for token.
        
        ACCOUNTABILITY: Unchanged functionality, critical for token management.
        """
        try:
            # Ensure token is string
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            elif not isinstance(token, str):
                token = str(token)
            
            # Get existing node from embeddings
            if token in self.embeddings:
                logger.debug(f"Found existing embedding for '{token}'")
                return self.embeddings[token]
            
            # Create new node
            logger.debug(f"Creating new embedding for '{token}'")
            node = create_semantic_trie_node(token, db_env=self.env, context_window=self.context_window)
            self.embeddings[token] = node
            return node
            
        except Exception as e:
            logger.error(f"Error getting embedding for '{token}': {e}")
            # Fallback creation
            try:
                node = create_semantic_trie_node(token, db_env=self.env, context_window=self.context_window)
                logger.info(f"Fallback embedding created for '{token}'")
                return node
            except Exception as fallback_error:
                logger.error(f"Fallback embedding creation failed: {fallback_error}")
                raise
    
    def _get_token_embeddings(self, tokens: List[str]) -> List[np.ndarray]:
        """
        PRESERVED: Create token embeddings and establish children relationships.
        
        ACCOUNTABILITY: Core functionality unchanged, builds trie structure in embeddings.
        """
        logger.info(f"Getting token embeddings for {len(tokens)} tokens: {tokens}")
        
        # Create/retrieve all nodes
        token_embeddings = []
        for token in tokens:
            try:
                node = self._get_embedding_for_token(token)
                token_embeddings.append(node.embedding)
                logger.debug(f"Retrieved embedding for '{token}'")
            except Exception as e:
                logger.error(f"Error getting embedding for token '{token}': {e}")
                raise
        
        # Build immediate-next children relationships
        logger.info("Building immediate-next children relationships")
        
        children_added = 0
        for i, token in enumerate(tokens):
            current_node = self.embeddings[token]
            
            # Add immediate next token as child
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                next_node = self.embeddings[next_token]
                
                if next_token not in current_node.children:
                    current_node.add_child(next_token, next_node)
                    children_added += 1
                    logger.debug(f"Added child '{next_token}' to '{token}'")
                else:
                    logger.debug(f"Child '{next_token}' already exists for '{token}'")
        
        logger.info(f"Children relationships: {children_added} new relationships added")
        return token_embeddings
    
    def add_embedding(self, token: str, embedding: np.ndarray, context_info: dict = None):
        """
        FIXED: Update existing node without destroying children relationships.
        
        ACCOUNTABILITY CHANGES:
        1. FIXED: Preserve existing nodes and their children
        2. MAINTAINED: Embedding and metadata updates
        3. PREVENTED: Accidental destruction of trie structure
        """
        try:
            # Get existing node or create if doesn't exist
            if token in self.embeddings:
                node = self.embeddings[token]
                logger.debug(f"Updating existing node for '{token}' (preserving {len(node.children)} children)")
            else:
                node = create_semantic_trie_node(token, db_env=self.env, context_window=self.context_window)
                self.embeddings[token] = node
                logger.debug(f"Created new node for '{token}'")
            
            # Update embedding and metadata without destroying node
            node.embedding = embedding
            if context_info:
                node.metadata.update(context_info)
            
            logger.debug(f"Updated embedding for '{token}' without destroying children")
            
        except Exception as e:
            logger.error(f"Error storing embedding for '{token}': {e}")
    
    def _collect_continuations(self, node: SemanticTrieNode, current_path: List[str], 
                               candidates: List[Tuple[List[str], float]], 
                               query_embedding: np.ndarray, max_continuations: int, 
                               max_depth: int = 12, context_embedding: np.ndarray = None):
        """
        PRESERVED: Collect continuation candidates from node children.
        
        ACCOUNTABILITY: Unchanged logic, works with embeddings structure.
        """
        node_confidence = getattr(node, 'confidence', 0.5)
        
        logger.debug(f"Collecting continuations from '{node.token}': path_len={len(current_path)}, "
                    f"candidates={len(candidates)}/{max_continuations}")
        
        # Termination conditions
        if len(candidates) >= max_continuations or len(current_path) >= max_depth:
            return
        
        # Process current path as candidate
        if current_path:
            try:
                relevance = node.calculate_relevance(context_embedding=context_embedding, query_embedding=query_embedding)
                activation = node.activation_level
                avg_reward = node.metadata.get('avg_reward', 0.0)
                completeness_bonus = 0.2 if node.is_complete else 0.0
                
                confidence_multiplier = 0.8 + (0.4 * node_confidence)
                base_score = (0.4 * relevance + 0.3 * activation + 0.2 * avg_reward + completeness_bonus)
                final_score = base_score * confidence_multiplier
                
                candidates.append((current_path.copy(), final_score))
                logger.debug(f"Added candidate: {current_path} (score: {final_score:.3f})")
                
            except Exception as e:
                logger.error(f"Error processing path {current_path}: {e}")
        
        # Process children
        if node.children:
            for child_token, child_node in node.children.items():
                if len(candidates) >= max_continuations:
                    break
                
                try:
                    self._collect_continuations(
                        child_node, current_path + [child_token], candidates, 
                        query_embedding, max_continuations, max_depth, context_embedding
                    )
                except Exception as e:
                    logger.error(f"Error processing child '{child_token}': {e}")
                    continue
    
    def _save_to_db(self, sequence_id: str, path_nodes: List[SemanticTrieNode], sequence_embedding: np.ndarray):
        """
        STREAMLINED: Save sequence to database without registry operations.
        
        ACCOUNTABILITY CHANGES:
        1. REMOVED: Registry-related saves
        2. PRESERVED: Node and metadata persistence
        3. MAINTAINED: All existing error handling and statistics
        """
        try:
            logger.info(f"Saving sequence {sequence_id[:8]}... with {len(path_nodes)} nodes")
            self._check_and_resize_if_needed()
            
            with self.env.begin(write=True) as txn:
                # Save sequence metadata
                metadata = {
                    'sequence_id': sequence_id,
                    'path_length': len(path_nodes),
                    'embedding': sequence_embedding.tobytes(),
                    'timestamp': time.time()
                }
                txn.put(sequence_id.encode(), msgpack.packb(metadata), db=self.metadata_db)
                
                # Save nodes
                for i, node in enumerate(path_nodes):
                    try:
                        node_key = f"{sequence_id}_{i}".encode()
                        
                        # Serialize children relationships
                        children_data = {}
                        if node.children:
                            for child_token, child_node in node.children.items():
                                children_data[child_token] = {
                                    'node_id': child_node.node_id,
                                    'token': child_node.token
                                }
                        
                        # Complete node data
                        node_data = {
                            'token': node.token,
                            'embedding': node.embedding.tobytes() if node.embedding is not None else None,
                            'relevance_score': node.relevance_score,
                            'activation_level': node.activation_level,
                            'is_complete': node.is_complete,
                            'metadata': node.metadata,
                            'reward_history': node.reward_history,
                            'access_count': node.access_count,
                            'last_accessed': node.last_accessed,
                            'is_end_of_sequence': node.is_end_of_sequence,
                            'children_relationships': children_data,
                            'node_id': node.node_id
                        }
                        
                        txn.put(node_key, msgpack.packb(node_data), db=self.nodes_db)
                        
                    except Exception as node_error:
                        logger.error(f"Error saving node {i}: {node_error}")
                        continue
            
            logger.info(f"Successfully saved sequence {sequence_id[:8]}...")
            
        except Exception as e:
            logger.error(f"Error saving sequence: {e}")
            raise
    
    def _load_embeddings_only(self):
        """
        STREAMLINED: Load embeddings without registry operations.
        
        ACCOUNTABILITY CHANGES:
        1. REMOVED: Registry rebuilding and root operations
        2. PRESERVED: Node loading and children relationship restoration
        3. SIMPLIFIED: Direct embeddings population only
        """
        logger.info("Loading embeddings without registry operations")
        
        nodes = {}
        sequence_groups = {}
        
        # Load all nodes from database
        with self.env.begin(db=self.nodes_db) as txn:
            for key, value in txn.cursor():
                try:
                    path = key.decode()
                    data = msgpack.unpackb(value, raw=False, strict_map_key=False)
                    
                    seq_id, pos = self._parse_node_key(path)
                    
                    # Create node
                    token = data.get('token')
                    if isinstance(token, bytes):
                        token = token.decode('utf-8')
                    
                    node = SemanticTrieNode(token=token, db_env=self.env, context_window=self.context_window)
                    
                    # Load node properties
                    node.relevance_score = data.get('relevance_score', 0.0)
                    node.activation_level = data.get('activation_level', 0.0)
                    node.is_complete = data.get('is_complete', False)
                    node.metadata = data.get('metadata', {})
                    node.reward_history = data.get('reward_history', [])
                    node.access_count = data.get('access_count', 0)
                    node.last_accessed = data.get('last_accessed', time.time())
                    node.is_end_of_sequence = data.get('is_end_of_sequence', False)
                    node.node_id = data.get('node_id', node._generate_node_id())
                    
                    # Load embedding
                    raw_embedding = data.get('embedding')
                    if raw_embedding:
                        node.embedding = np.frombuffer(raw_embedding, dtype=np.float32)
                    
                    # Store node
                    nodes[path] = node
                    if token:
                        self.embeddings[token] = node
                    
                    # Group by sequence
                    if seq_id not in sequence_groups:
                        sequence_groups[seq_id] = {}
                    sequence_groups[seq_id][pos] = (path, node)
                    
                except Exception as e:
                    logger.error(f"Error loading node {key}: {e}")
                    continue
        
        # Restore children relationships
        logger.info("Restoring children relationships")
        self._link_children_simple(nodes)
        
        # Restore additional children from database
        self._restore_additional_children(nodes)
        
        logger.info(f"Loaded {len(self.embeddings)} tokens into embeddings dictionary")
    
    def _link_children_simple(self, nodes):
        """PRESERVED: Link parent-child relationships within sequences."""
        logger.info("Linking children relationships within sequences")
        
        successful_links = 0
        for path, node in nodes.items():
            try:
                seq_id, pos = self._parse_node_key(path)
                child_path = f"{seq_id}_{pos + 1}"
                
                if child_path in nodes:
                    child = nodes[child_path]
                    if child.token and node.token:
                        node.children[child.token] = child
                        successful_links += 1
                        logger.debug(f"Linked: {node.token} -> {child.token}")
                        
            except Exception as e:
                logger.error(f"Error linking node {path}: {e}")
                continue
        
        logger.info(f"Successfully linked {successful_links} parent-child relationships")
    
    def _restore_additional_children(self, nodes):
        """PRESERVED: Restore additional children relationships from database."""
        logger.info("Restoring additional children relationships from database")
        
        restored_children = 0
        
        with self.env.begin(db=self.nodes_db) as txn:
            for key, value in txn.cursor():
                try:
                    path = key.decode()
                    data = msgpack.unpackb(value, raw=False, strict_map_key=False)
                    
                    if path not in nodes:
                        continue
                    
                    parent_node = nodes[path]
                    children_data = data.get('children_relationships', {})
                    
                    for child_token, child_info in children_data.items():
                        if child_token in self.embeddings:
                            child_node = self.embeddings[child_token]
                            
                            if child_token not in parent_node.children:
                                parent_node.children[child_token] = child_node
                                restored_children += 1
                                
                except Exception as e:
                    logger.debug(f"Error restoring children for {key}: {e}")
                    continue
        
        logger.info(f"Restored {restored_children} additional children relationships")
    
    def _parse_node_key(self, path: str) -> Tuple[str, int]:
        """PRESERVED: Parse sequence ID and position from node key."""
        try:
            parts = path.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0], int(parts[1])
            return path, 0
        except Exception:
            return path, 0
    
    def _calculate_adaptive_map_size(self) -> int:
        """PRESERVED: Calculate initial LMDB map size."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(os.path.dirname(os.path.abspath(self.db_path)))
            recommended_size = int(free * 0.1)
            min_size = 1024 * 1024 * 1024
            max_size = 10 * 1024 * 1024 * 1024
            adaptive_size = max(min_size, min(recommended_size, max_size))
            return adaptive_size
        except Exception:
            return 4 * 1024 * 1024 * 1024
    
    def _check_and_resize_if_needed(self):
        """PRESERVED: Check and resize database if needed."""
        try:
            stat = self.env.stat()
            used_pages = stat['psize'] * stat['leaf_pages']
            usage_ratio = used_pages / self.current_map_size
            
            if usage_ratio >= self.resize_threshold:
                new_size = int(self.current_map_size * self.resize_factor)
                self._resize_database(new_size)
                
        except Exception as e:
            logger.warning(f"Error checking database capacity: {e}")
    
    def _resize_database(self, new_size: int):
        """PRESERVED: Resize database when needed."""
        try:
            logger.info(f"Resizing database to {new_size/(1024**3):.1f}GB")
            
            self.env.sync(force=True)
            self.env.close()
            time.sleep(0.1)
            
            self.env = lmdb.open(
                self.db_path,
                map_size=new_size,
                max_dbs=2,
                writemap=True,
                map_async=True
            )
            
            self.nodes_db = self.env.open_db(b'nodes')
            self.metadata_db = self.env.open_db(b'metadata')
            self.current_map_size = new_size
            
            logger.info("Database resize completed successfully")
            
        except Exception as e:
            logger.error(f"Error during database resize: {e}")
            raise
    
    def _generate_sequence_id(self, tokens: List[str]) -> str:
        """PRESERVED: Generate deterministic sequence ID."""
        sequence_text = ' '.join(tokens)
        return hashlib.md5(sequence_text.encode()).hexdigest()
    
    def close(self):
        """PRESERVED: Close database connections."""
        if hasattr(self, 'env'):
            self.env.close()
            logger.info("Closed LMDB environment")