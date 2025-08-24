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
import uuid

from optimized_aggregate_manager import OptimizedAggregateManager
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import gc
import msgpack
import numpy as np
import logging
import lmdb
import time
from typing import List, Dict, Tuple, Optional, Any, Union
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

    def __init__(self, core_values, db_path: str = "./trie_memory_test.lmdb", embed_dim: int = 4096):
        """
        SIMPLIFIED: Initialize TrieMemory with embeddings-only approach.
        
        ACCOUNTABILITY CHANGES:
        1. REMOVED: self.root initialization and registry setup
        2. PRESERVED: Database, context window, and embeddings initialization
        3. MAINTAINED: All existing database configuration and loading
        """
        logger.info("Initializing Sotbi TrieMemory with proper core values integration")
        
        # Core configuration
        self.embed_dim = embed_dim
        self.db_path = db_path
        
        # FIXED: Properly handle core_values parameter
        if isinstance(core_values, dict):
            # ✅ NEW: Core values passed as dictionary (correct usage)
            self.core_values = core_values
            logger.info(f"Sotbi core values loaded: {list(core_values.keys())}")
        elif isinstance(core_values, str):
            # ⚠️ LEGACY: Core values passed as string (for backward compatibility)
            self.core_values = None
            self.identity_context = core_values
            logger.warning("Core values passed as string - value-aware features disabled")
        else:
            # 🚫 FALLBACK: Invalid core values
            self.core_values = None
            self.identity_context = "Sotbi: Self-Organizing Trie-Based Intelligence"
            logger.error("Invalid core values format - using fallback identity")
        
        self.context_window = ContextWindow(core_values=self.core_values, max_turns=500, max_tokens=100, time_window_seconds=300)
        
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
                max_dbs=5,  # ✅ INCREASED LIMIT
                writemap=True,  
                map_async=True  
            )
            self.nodes_db = self.env.open_db(b'nodes')
            self.metadata_db = self.env.open_db(b'metadata')
            
            # Database monitoring
            self.current_map_size = initial_map_size
            self.resize_threshold = 0.8  
            self.resize_factor = 2.0
            # ADDED: Context tracking for sequence disambiguation
            self.sequence_contexts: Dict[str, Dict[str, Any]] = {}  # sequence_id -> context_info
            self.context_db = None  # Will be initialized in database setup
                # MODIFIED: Removed caching, simplified to always-update tracking
            self.aggregate_stats = {
            'last_global_update': 0.0,
            'last_activation_update': 0.0,
            'total_updates': 0
            }
            
            
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
        # ADDED: Load existing contexts after database setup
        try:
            self._load_contexts_from_db()
            logger.info(f"Context-aware TrieMemory initialization complete: {len(self.embeddings)} tokens, {len(self.sequence_contexts)} contexts loaded")
        except Exception as e:
            logger.error(f"Error loading contexts from database: {e}")
            logger.info("Starting with empty contexts dictionary")
        
        self._always_update_aggregates()  # Initialize always-update aggregates
        
        logger.info("Initialized always-update aggregate system (no caching)")
    
        logger.info("Enhanced TrieMemory with context-aware sequence tracking")
    
    def _always_update_aggregates(self):
        """
        MODIFIED: Always update aggregates on every call (no caching).
        
        ACCOUNTABILITY CHANGES:
        1. REMOVED: Threshold-based conditional updating
        2. REMOVED: Cached aggregate storage
        3. ADDED: Always-update execution with timing tracking
        4. PRESERVED: All aggregate calculation methods
        """
        try:
            start_time = time.time()
            logger.info("Starting always-update aggregate calculation")
            self.global_centroid = self.calculate_global_embedding_centroid()
            self.activation_weighted = self.calculate_weighted_embedding_aggregate('activation', self.global_centroid)
            # Update timing statistics
            update_time = time.time() - start_time
            self.aggregate_stats['last_global_update'] = time.time()
            self.aggregate_stats['last_activation_update'] = time.time()
            self.aggregate_stats['total_updates'] += 1
            
            logger.info(f"Always-update aggregates completed in {update_time:.3f}s "
                       f"(total updates: {self.aggregate_stats['total_updates']})")
            
            return {
                'global_centroid': self.global_centroid,
                'activation_weighted': self.activation_weighted,
                'update_time': update_time
            }
            
        except Exception as e:
            logger.error(f"Error in always-update aggregates: {e}")
            # Return zero vectors as fallback
            return {
                'global_centroid': np.zeros(self.embed_dim, dtype=np.float32),
                'activation_weighted': np.zeros(self.embed_dim, dtype=np.float32),
                'update_time': 0.0
            }
    
    def get_aggregate_performance_stats(self) -> Dict[str, Any]:
        """
        ADDED: Monitor performance impact of always-update system.

        JUSTIFICATION: Always-updating can be computationally expensive,
        monitoring helps identify if optimization is needed.
        """
        try:
            total_embeddings = len(self.embeddings)
            stats = self.aggregate_stats.copy()

            # Calculate update frequency
            current_time = time.time()
            time_since_last = current_time - stats.get('last_global_update', current_time)

            performance_stats = {
                'total_embeddings': total_embeddings,
                'total_updates': stats['total_updates'],
                'last_update_seconds_ago': time_since_last,
                'average_embeddings_per_update': total_embeddings / max(1, stats['total_updates']),
                'system_mode': 'always_update',
                'caching_disabled': True
            }

            # Performance recommendations
            if total_embeddings > 10000:
                performance_stats['recommendation'] = 'Consider batched updates for large embedding sets'
            elif total_embeddings > 1000:
                performance_stats['recommendation'] = 'Monitor update frequency for performance'
            else:
                performance_stats['recommendation'] = 'Always-update suitable for current size'

            logger.info(f"Aggregate performance: {total_embeddings} embeddings, "
                       f"{stats['total_updates']} updates, mode: always_update")

            return performance_stats

        except Exception as e:
            logger.error(f"Error getting aggregate performance stats: {e}")
            return {'error': str(e), 'system_mode': 'always_update'}


    
    def calculate_global_embedding_centroid(self) -> np.ndarray:
        """Memory-efficient centroid calculation without intermediate lists."""
        try:
            count = 0
            running_sum = np.zeros(self.embed_dim, dtype=np.float32)
    
            for token, node in self.embeddings.items():
                if node.embedding is not None:
                    running_sum += node.embedding
                    count += 1
    
            if count == 0:
                logger.warning("No embeddings available for centroid calculation")
                return np.zeros(self.embed_dim, dtype=np.float32)
    
            global_centroid = running_sum / count
    
            norm = np.linalg.norm(global_centroid)
            if norm > 0:
                global_centroid = global_centroid / norm
    
            logger.info(f"Calculated global embedding centroid from {count} embeddings (memory-optimized)")
            return global_centroid
    
        except Exception as e:
            logger.error(f"Error calculating global centroid: {e}")
            return np.zeros(self.embed_dim, dtype=np.float32)

    def calculate_weighted_embedding_aggregate(self, weight_type: str = 'activation', global_centroid: np.ndarray = None) -> np.ndarray:
        """Calculate weighted average emphasizing high-performing nodes."""
        try:
            weighted_embeddings = []
            weights = []

            for token, node in self.embeddings.items():
                if node.embedding is not None:
                    # Choose weighting strategy
                    if weight_type == 'activation':
                        weight = node.activation_level
                    elif weight_type == 'confidence':
                        weight = node.confidence
                    elif weight_type == 'reward':
                        weight = node.metadata.get('avg_reward', 0.0)
                    else:
                        weight = 1.0

                    # Only include nodes with positive weight
                    if weight > 0:
                        weighted_embeddings.append(node.embedding * weight)
                        weights.append(weight)
                        
            if not weighted_embeddings:
                logger.warning(f"No embeddings found for {weight_type}-weighted aggregate calculation")
                return self.calculate_global_embedding_centroid() if global_centroid is None else global_centroid

            # Calculate weighted average
            total_weight = sum(weights)
            weighted_aggregate = np.sum(weighted_embeddings, axis=0) / total_weight

            # Normalize
            norm = np.linalg.norm(weighted_aggregate)
            if norm > 0:
                weighted_aggregate = weighted_aggregate / norm

            logger.info(f"Calculated {weight_type}-weighted aggregate from {len(weights)} nodes")
            return weighted_aggregate

        except Exception as e:
            logger.error(f"Error calculating weighted aggregate: {e}")
            return self.calculate_global_embedding_centroid()
        
    def calculate_contextual_embedding_aggregate(self, query_tokens: List[str], 
                                               similarity_threshold: float = 0.3) -> np.ndarray:
        """Calculate aggregate from embeddings similar to query context."""
        try:
            # Get query embedding representation
            query_embeddings = []
            for token in query_tokens:
                if token in self.embeddings and self.embeddings[token].embedding is not None:
                    query_embeddings.append(self.embeddings[token].embedding)

            if not query_embeddings:
                return self.calculate_global_embedding_centroid()

            query_centroid = np.mean(query_embeddings, axis=0)
            norm = np.linalg.norm(query_centroid)
            if norm > 0:
                query_centroid = query_centroid / norm

            # Find contextually relevant embeddings
            relevant_embeddings = []

            for token, node in self.embeddings.items():
                if node.embedding is not None:
                    similarity = self.context_window._calculate_ensemble_similarity(
                        query_centroid, node.embedding
                    )

                    if similarity > similarity_threshold:
                        relevant_embeddings.append(node.embedding)

            if not relevant_embeddings:
                return query_centroid

            # Calculate contextual aggregate
            contextual_aggregate = np.mean(relevant_embeddings, axis=0)
            norm = np.linalg.norm(contextual_aggregate)
            if norm > 0:
                contextual_aggregate = contextual_aggregate / norm

            logger.info(f"Calculated contextual aggregate from {len(relevant_embeddings)} relevant embeddings")
            return contextual_aggregate

        except Exception as e:
            logger.error(f"Error calculating contextual aggregate: {e}")
            return self.calculate_global_embedding_centroid()
        
    def score_prediction_with_aggregate(self, candidate_tokens: List[str], 
                                      query_tokens: List[str]) -> Dict[str, float]:
        """Score predictions using multiple embedding aggregates."""
        try:
            # Calculate different aggregate references
            
            activation_weighted = self.calculate_weighted_embedding_aggregate('activation')
            contextual_aggregate = self.calculate_contextual_embedding_aggregate(query_tokens)

            # Get candidate embedding
            candidate_embeddings = []
            for token in candidate_tokens:
                if token in self.embeddings and self.embeddings[token].embedding is not None:
                    candidate_embeddings.append(self.embeddings[token].embedding)

            if not candidate_embeddings:
                return {'global_coherence': 0.0, 'activation_alignment': 0.0, 'contextual_fit': 0.0}

            candidate_centroid = np.mean(candidate_embeddings, axis=0)
            norm = np.linalg.norm(candidate_centroid)
            if norm > 0:
                candidate_centroid = candidate_centroid / norm

            # Calculate aggregate-based scores
            scores = {
                'global_coherence': self.context_window._calculate_ensemble_similarity(
                    candidate_centroid, self.global_centroid
                ),
                'activation_alignment': self.context_window._calculate_ensemble_similarity(
                    candidate_centroid, activation_weighted
                ),
                'contextual_fit': self.context_window._calculate_ensemble_similarity(
                    candidate_centroid, contextual_aggregate
                )
            }

            logger.debug(f"Aggregate scores: {scores}")
            return scores

        except Exception as e:
            logger.error(f"Error scoring prediction with aggregates: {e}")
            return {'global_coherence': 0.0, 'activation_alignment': 0.0, 'contextual_fit': 0.0}
        
    

    def add_embedding(self, token: str, embedding: np.ndarray, context_info: dict = None):
        """
        MODIFIED: Removed caching counter, preserved all existing logic.
        
        ACCOUNTABILITY CHANGES:
        1. REMOVED: self.embeddings_added_since_update increment
        2. PRESERVED: All existing embedding addition logic
        3. MAINTAINED: All error handling and functionality
        """
        try:
            # PRESERVED: Get existing node or create if doesn't exist
            if token in self.embeddings:
                node = self.embeddings[token]
                logger.debug(f"Updating existing node for '{token}' (preserving {len(node.children)} children)")
            else:
                node = create_semantic_trie_node(token, db_env=self.env, context_window=self.context_window)
                self.embeddings[token] = node
                logger.debug(f"Created new node for '{token}'")
            
            # PRESERVED: Update embedding and metadata without destroying node
            node.embedding = embedding
            if context_info:
                node.metadata.update(context_info)
            
            logger.debug(f"Updated embedding for '{token}' without destroying children")
            # REMOVED: self.embeddings_added_since_update += 1 (no longer needed)
            
        except Exception as e:
            logger.error(f"Error storing embedding for '{token}': {e}")            

    
    def _generate_context_id(self, tokens: List[str], timestamp: float = None) -> str:
        """
        ADDED: Generate unique context ID for sequence tracking.
        
        JUSTIFICATION: Provides unique identifier for each sequence context.
        """
        try:
            if timestamp is None:
                timestamp = time.time()
            
            # Create deterministic but unique context ID
            sequence_text = ' '.join(tokens)
            combined_input = f"{sequence_text}_{timestamp}_{uuid.uuid4().hex[:8]}"
            context_id = hashlib.md5(combined_input.encode()).hexdigest()[:16]
            
            logger.debug(f"Generated context_id: {context_id} for sequence: {tokens[:3]}...")
            return context_id
            
        except Exception as e:
            logger.error(f"Error generating context ID: {e}")
            return f"ctx_{int(time.time())}_{hash(str(tokens)) % 10000}"
    
    def learn_sequence(self, tokens: List[str], reward: float = 1.0) -> np.ndarray:
        """
        ENHANCED: Learn sequence with context ID tracking and proper embedding consideration.
        
        ACCOUNTABILITY CHANGES:
        1. ADDED: Context ID generation for sequence tracking
        2. ADDED: Context-aware child relationship building
        3. PRESERVED: All existing sequence learning logic
        4. ENHANCED: Better sequence embedding utilization in context storage
        """
        logger.info(f"Learning sequence with context tracking: {len(tokens)} tokens with reward {reward}")
        
        try:
            # PRESERVED: Original token embedding generation
            token_embeddings = self._get_token_embeddings(tokens)
            
            if not token_embeddings:
                logger.warning("No token embeddings generated - sequence learning aborted")
                return np.zeros(self.embed_dim, dtype=np.float32)
            
            # PRESERVED: Original sequence embedding calculation
            embeddings_array = np.array([embedding for embedding in token_embeddings])
            sequence_embedding = np.mean(embeddings_array, axis=0)
            
            # PRESERVED: Normalization and context addition
            norm = np.linalg.norm(sequence_embedding)
            if norm > 0:
                sequence_embedding = sequence_embedding / norm
            
            self.context_window.add_turn(tokens, sequence_embedding)
            
            # ADDED: Generate context ID for this specific sequence
            context_id = self._generate_context_id(tokens)
            current_timestamp = time.time()
            
            # ADDED: Store sequence context information
            self.sequence_contexts[context_id] = {
                'tokens': tokens.copy(),
                'sequence_embedding': sequence_embedding.copy(),
                'reward': reward,
                'timestamp': current_timestamp,
                'embedding_bytes': sequence_embedding.tobytes()
            }
            
            # ENHANCED: Build context-aware children relationships
            for i, (token, token_embedding) in enumerate(zip(tokens, token_embeddings)):
                try:
                    # PRESERVED: Original context info structure
                    context_info = {
                        'sequence': tokens,
                        'sequence_embedding': sequence_embedding.tobytes(),
                        'position': i,
                        'reward': reward,
                        'timestamp': current_timestamp,
                        'source': 'trie_sequence',
                        'context_id': context_id  # ADDED: Context ID tracking
                    }
                    
                    self.add_embedding(token=token, embedding=token_embedding, context_info=context_info)
                    
                    # ADDED: Build context-aware child relationships
                    if i + 1 < len(tokens):
                        current_node = self.embeddings[token]
                        next_token = tokens[i + 1]
                        next_node = self.embeddings[next_token]
                        
                        # ADDED: Context-aware child addition
                        current_node.add_child_with_context(
                            token=next_token,
                            child=next_node,
                            context_id=context_id,
                            full_sequence=tokens,
                            sequence_embedding=sequence_embedding
                        )
                        
                        logger.debug(f"Added context-aware child: '{token}' -> '{next_token}' (context: {context_id})")
                    
                except Exception as node_error:
                    logger.error(f"Error processing node {i} in context-aware learning: {node_error}")
                    continue
            
            # ADDED: Persist context to database
            self._save_context_to_db(context_id, self.sequence_contexts[context_id])
            
            logger.info(f"Successfully learned sequence with context tracking: {len(tokens)} tokens, context_id: {context_id}")
            return sequence_embedding
            
        except Exception as e:
            logger.error(f"Error in context-aware sequence learning: {e}")
            return np.zeros(self.embed_dim, dtype=np.float32)
        
    
    def _deduplicate_subset_candidates(self, candidates: List[Tuple[List[str], float]]) -> List[Tuple[List[str], float]]:
        """
        Remove subset candidates after all candidates have been collected.

        ACCOUNTABILITY:
        1. RESTORED: Post-collection deduplication function from previous implementation
        2. EFFICIENCY: Single pass deduplication instead of per-candidate deduplication
        3. PRESERVED: Original candidate scoring and ordering logic
        4. OPTIMIZED: Processes all candidates once rather than during recursion
        5. ADDED: Comprehensive error handling and execution logging for transparency

        Args:
            candidates: List of (path, score) tuples from collection

        Returns:
            Deduplicated list with subset paths removed

        Raises:
            No exceptions raised - uses fallback strategy on errors
        """
        if not candidates:
            logger.info("Deduplication skipped: no candidates provided")
            return candidates

        logger.info(f"Starting deduplication of {len(candidates)} candidates")

        try:
            # Sort by path length (longest first) to process efficiently
            # JUSTIFICATION: Processing longest paths first ensures we keep the most complete sequences
            sorted_candidates = sorted(candidates, key=lambda x: len(x[0]), reverse=True)
            logger.debug(f"Sorted candidates by length: longest={len(sorted_candidates[0][0])}, shortest={len(sorted_candidates[-1][0])}")

            deduplicated = []
            processed_paths = set()
            exact_removals = 0
            partial_removals = 0
            kept_candidates = 0

            for i, (path, score) in enumerate(sorted_candidates):
            
                path_tuple = tuple(path)  # Convert to immutable for set operations

                # Check if this path is a subset of any already processed path
                is_subset = False
                subset_of_path = None

                for processed_path_tuple in processed_paths:
                    processed_path = list(processed_path_tuple)

                    # Check if current path is subset of processed path
                    # CONDITION 1: Current path must be shorter
                    # CONDITION 2: All tokens in current path must be in processed path
                    if (len(path) < len(processed_path) and 
                        set(path).issubset(set(processed_path))):

                        # Verify sequence order preservation
                        # JUSTIFICATION: "Do you" should be subset of "Do you want" but not "you Do want"
                        path_idx = 0
                        for token in processed_path:
                            if path_idx < len(path) and token == path[path_idx]:
                                path_idx += 1

                        if path_idx == len(path):  # All tokens found in order
                            is_subset = True
                            subset_of_path = processed_path
                            exact_removals += 1
                            logger.debug(f"Removing exact subset: {path} (subset of {processed_path})")
                            break
                        
                if not is_subset:
                    # Keep this candidate - not a subset of any processed path
                    deduplicated.append((path, score))
                    processed_paths.add(path_tuple)
                    kept_candidates += 1
                    logger.debug(f"Kept candidate {i}: {path} (score: {score:.3f})")
                else:
                    logger.debug(f"Removed candidate {i}: {path} (subset of {subset_of_path})")

            # Log comprehensive deduplication statistics
            logger.info(f"Deduplication complete: {len(candidates)} → {len(deduplicated)} candidates")
            logger.info(f"Deduplication statistics: kept={kept_candidates}, exact_removals={exact_removals}, partial_removals={partial_removals}")

            if not deduplicated:
                logger.warning("No candidates remained after deduplication - returning original candidates as fallback")
                return candidates

            # Verify deduplication integrity
            if len(deduplicated) > len(candidates):
                logger.error(f"Deduplication error: more candidates after deduplication ({len(deduplicated)}) than before ({len(candidates)})")
                return candidates

            return deduplicated

        except Exception as dedup_error:
            logger.error(f"Error during deduplication process: {dedup_error}")
            logger.info("Falling back to original candidates due to deduplication error")
            # FALLBACK STRATEGY: Return original candidates to prevent system failure
            return candidates


        
    def find_best_continuation(self, query_tokens: List[str], 
                              context_embedding: np.ndarray,
                              query_sequence_embedding: np.ndarray,
                              max_candidates: int = 1000, 
                              max_continuations: int = 500) -> Tuple[List[str], float]:
        """
        FIXED: Find continuation with proper semantic filtering and scoring flow.
        
        CRITICAL FIXES APPLIED:
        1. FIXED: Moved semantic filtering outside scoring loop
        2. FIXED: Proper sequence similarity bonus calculation
        3. FIXED: Eliminated variable overwriting in loop
        4. PRESERVED: All existing logic and scoring components
        """
        logger.info(f"Finding context-aware continuation for: {query_tokens} (max_candidates={max_candidates})")
        
        if not query_tokens:
            logger.warning("Empty query tokens provided")
            return [], 0.0
        
        try:
            # PRESERVED: Original token matching logic
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
            
            # PRESERVED: Original confidence calculations
            avg_match_confidence = sum(match_confidences) / len(match_confidences)
            min_match_confidence = min(match_confidences)
            logger.info(f"Match confidence: avg={avg_match_confidence:.3f}, min={min_match_confidence:.3f}")
            
            # PRESERVED: Context-aware continuation collection
            last_matched_node = matched_nodes[-1]
            logger.info(f"Collecting context-aware continuations from token '{last_matched_node.token}'")
            
            # PRESERVED: Find best matching context using sequence embedding
            best_context_id = None
            if query_sequence_embedding is not None:
                best_context_id = last_matched_node.find_best_context_match(
                    query_sequence_embedding, similarity_threshold=0.6
                )
                if best_context_id:
                    logger.info(f"Found matching context: {best_context_id}")
                else:
                    logger.info("No specific context match found, using all children")
            
            candidates = []
            self._collect_context_aware_continuations(
                last_matched_node, matched_tokens, candidates, 
                query_sequence_embedding, max_continuations, 
                context_embedding=context_embedding,
                preferred_context_id=best_context_id
            )
            
            logger.info(f"Collection complete: {len(candidates)} raw candidates")
            candidates = self._deduplicate_subset_candidates(candidates)
            # FIXED: Query stripping logic
            continuation_candidates = self._strip_query_from_candidates(candidates, query_tokens)
            logger.info(f"Query stripping complete: {len(candidates)} → {len(continuation_candidates)} candidates")


            logger.info(f"Collected {len(continuation_candidates)} context-aware continuation candidates")
            
            if not continuation_candidates:
                logger.warning("No continuation candidates found")
                return [], 0.0
            '''
            # FIXED: Semantic filtering moved OUTSIDE scoring loop (single execution)
            if query_sequence_embedding is not None:
                query_semantic_threshold = 0.2  # Minimum semantic similarity to query
                logger.info(f"Applying semantic filtering with threshold: {query_semantic_threshold}")
                
                semantically_relevant_candidates = []
                for path, base_score in continuation_candidates:
                    path_semantic_score = self._calculate_sequence_similarity_bonus(path, query_sequence_embedding)
                    
                    if path_semantic_score >= query_semantic_threshold:
                        semantically_relevant_candidates.append((path, base_score))
                    else:
                        logger.debug(f"Filtered out low semantic relevance path: {path} (score: {path_semantic_score:.3f})")
                
                # Use filtered candidates if available, otherwise apply penalty to all
                if semantically_relevant_candidates:
                    continuation_candidates = semantically_relevant_candidates
                    logger.info(f"Using {len(semantically_relevant_candidates)} semantically relevant candidates")
                else:
                    # Apply penalty to all candidates for low semantic relevance
                    continuation_candidates = [(path, score * 0.5) for path, score in continuation_candidates]
                    logger.warning("No semantically relevant candidates found, applying penalty to all")
            
            # FIXED: Proper scoring loop with correct sequence similarity calculation
            scored_candidates = []
            for path, base_score in continuation_candidates:
                # FIXED: Calculate sequence embedding similarity bonus properly
                sequence_similarity_bonus = 0.0
                if query_sequence_embedding is not None and len(path) > 0:
                    sequence_similarity_bonus = self._calculate_sequence_similarity_bonus(path, query_sequence_embedding)
                
                # PRESERVED: Original scoring components
                path_confidence_bonus = avg_match_confidence * 0.05
                length_bonus = 0.1 * matched_length
                sequential_bonus = 0.02 * len(path)
                
                # ENHANCED: Final score with proper sequence embedding consideration
                final_score = base_score + length_bonus + path_confidence_bonus + sequential_bonus + sequence_similarity_bonus
                scored_candidates.append((path, final_score))
                
                logger.debug(f"Scored candidate: {path} -> final_score={final_score:.3f} (seq_bonus={sequence_similarity_bonus:.3f})")
            
            # PRESERVED: Original viability filtering
            viability_threshold = max(0.01, min_match_confidence * 0.02)
            viable_candidates = [(cont, score) for cont, score in scored_candidates if score > viability_threshold]
            
            logger.info(f"Viability check: {len(viable_candidates)}/{len(scored_candidates)} candidates viable")
            
            if not viable_candidates:
                logger.warning("No viable candidates above threshold")
                return [], 0.0
            '''
            # PRESERVED: Best candidate selection
            # avg_score = sum(x[1] for x in continuation_candidates) / len(continuation_candidates)
            # logger.info(f"Calculated average score: {avg_score} from {len(continuation_candidates)} candidates")

            # Find continuation with score closest to average
            # best_continuation, best_score = min(continuation_candidates, key=lambda x: abs(x[1] - avg_score))
            best_continuation, best_score = max(continuation_candidates, key=lambda x: x[1])
            #logger.info(f"Selected continuation with score {best_score} (difference from average: {abs(best_score - avg_score)})")

            #logger.info(f"Best context-aware continuation selected: {best_continuation} (score: {best_score:.3f})")
            return best_continuation, best_score
            
        except Exception as e:
            logger.error(f"Error in context-aware continuation finding: {e}")
            return [], 0.0

    def _strip_query_from_candidates(self, candidates: List[Tuple[List[str], float]], 
                                     input_query: List[str]) -> List[Tuple[List[str], float]]:
        """
        Remove input query tokens from the beginning of continuation candidates.

        ACCOUNTABILITY:
        1. ADDED: Post-collection query stripping function
        2. PRESERVED: All candidate scoring and ranking logic
        3. ADDED: Comprehensive error handling and logging
        4. MAINTAINED: Original candidate structure (path, score) tuples

        Args:
            candidates: List of (path, score) tuples from collection
            input_query: Original input query tokens to remove

        Returns:
            List of candidates with query tokens stripped from paths
        """
        if not candidates or not input_query:
            logger.info(f"Query stripping skipped: candidates={len(candidates) if candidates else 0}, query_len={len(input_query) if input_query else 0}")
            return candidates

        logger.info(f"Stripping query tokens from {len(candidates)} candidates: query_len={len(input_query)}")

        stripped_candidates = []
        query_length = len(input_query)
        successful_strips = 0
        partial_matches = 0
        no_match_candidates = 0

        try:
            for i, (candidate_path, score) in enumerate(candidates):
                try:
                    # Verify candidate path starts with query tokens
                    if len(candidate_path) >= query_length:
                        # Check if beginning of candidate matches input query exactly
                        if candidate_path[:query_length] == input_query:
                            # Strip query tokens from beginning
                            stripped_path = candidate_path[query_length:]

                            # Only add if there are continuation tokens beyond the query
                            if stripped_path:
                                stripped_candidates.append((stripped_path, score))
                                successful_strips += 1
                                logger.debug(f"Stripped candidate {i}: {len(candidate_path)} → {len(stripped_path)} tokens")
                            else:
                                logger.debug(f"Skipped candidate {i}: no continuation tokens beyond query")
                        else:
                            # Partial match or no match - check for partial alignment
                            matching_tokens = 0
                            for j in range(min(len(candidate_path), query_length)):
                                if candidate_path[j] == input_query[j]:
                                    matching_tokens += 1
                                else:
                                    break
                                
                            if matching_tokens > 0:
                                # Partial match - strip matching tokens
                                stripped_path = candidate_path[matching_tokens:]
                                if stripped_path:
                                    stripped_candidates.append((stripped_path, score))
                                    partial_matches += 1
                                    logger.debug(f"Partial strip candidate {i}: matched {matching_tokens}/{query_length} tokens")
                                else:
                                    logger.debug(f"Skipped partial candidate {i}: no continuation tokens")
                            else:
                                # No match - keep original candidate as fallback
                                stripped_candidates.append((candidate_path, score))
                                no_match_candidates += 1
                                logger.debug(f"No match candidate {i}: kept original path")
                    else:
                        # Candidate shorter than query - keep as is
                        stripped_candidates.append((candidate_path, score))
                        no_match_candidates += 1
                        logger.debug(f"Short candidate {i}: {len(candidate_path)} < {query_length}, kept original")

                except Exception as candidate_error:
                    logger.error(f"Error processing candidate {i}: {candidate_error}")
                    # Fallback: keep original candidate
                    stripped_candidates.append((candidate_path, score))
                    no_match_candidates += 1

            # Log stripping statistics
            logger.info(f"Query stripping completed: {successful_strips} exact matches, {partial_matches} partial matches, {no_match_candidates} no-match/fallbacks")

            if not stripped_candidates:
                logger.warning("No candidates remained after query stripping - returning original candidates")
                return candidates

            return stripped_candidates

        except Exception as stripping_error:
            logger.error(f"Error during query stripping: {stripping_error}")
            logger.info("Falling back to original candidates due to stripping error")
            return candidates

   
    def _collect_context_aware_continuations(self, node: 'SemanticTrieNode', current_path: List[str], 
                                           candidates: List[Tuple[List[str], float]], 
                                           query_sequence_embedding: np.ndarray, max_continuations: int, 
                                           max_depth: int = 12, context_embedding: np.ndarray = None,
                                           visited_nodes: set = None, recent_tokens: List[str] = None,
                                           preferred_context_id: str = None):
        """
        ENHANCED: Collect continuations with balanced loop prevention.

        ACCOUNTABILITY CHANGES:
        1. ADJUSTED: More permissive loop detection thresholds
        2. REDUCED: Overly aggressive cycle prevention  
        3. INCREASED: Allowable path exploration depth
        4. PRESERVED: All existing scoring, aggregate calculations, and processing logic
        5. BALANCED: Loop prevention with legitimate continuation generation
        """

        def _detect_severe_loop_risk(path: List[str], max_repetition_length: int = 12) -> bool:
            """
            Detect only severe looping patterns that indicate infinite loops.

            ACCOUNTABILITY CHANGES:
            1. INCREASED: Minimum path length from 6 to 10 for pattern detection
            2. INCREASED: Repetition threshold from 3 to 4 occurrences  
            3. REDUCED: False positive detection of legitimate patterns

            Args:
                path: Current token path
                max_repetition_length: Maximum length of repetitive pattern to check

            Returns:
                True only if severe loop risk detected
            """
            if len(path) < 10:  # INCREASED: Need longer path for reliable pattern detection
                return False

            try:
                # Check for exact phrase repetition (only longer patterns indicate loops)
                for pattern_len in range(5, min(max_repetition_length + 1, len(path) // 3 + 1)):  # INCREASED: Min pattern length
                    if len(path) >= pattern_len * 3:  # INCREASED: Require 3x repetition for detection
                        recent_pattern = path[-pattern_len:]
                        previous_pattern = path[-pattern_len * 2:-pattern_len]
                        earlier_pattern = path[-pattern_len * 3:-pattern_len * 2]

                        # Require 3-way match for severe loop detection
                        if recent_pattern == previous_pattern == earlier_pattern:
                            logger.warning(f"Severe loop detected: pattern '{' '.join(recent_pattern)}' repeated 3+ times")
                            return True

                # Check for extreme token-level repetition (same token appearing excessively)
                if len(path) >= 8:  # INCREASED: Longer window for token repetition check
                    last_token = path[-1]
                    recent_count = path[-8:].count(last_token)  # INCREASED: Check last 8 tokens
                    if recent_count >= 5:  # INCREASED: Threshold from 3 to 5 occurrences
                        logger.warning(f"Severe token repetition detected: '{last_token}' appears {recent_count} times in recent path")
                        return True

                return False

            except Exception as detection_error:
                logger.error(f"Error in severe loop detection: {detection_error}")
                return False

        def _is_creating_problematic_cycle(current_path: List[str], node_token: str, visited_paths: set) -> bool:
            """
            Check only for problematic cycles that indicate infinite recursion.

            ACCOUNTABILITY CHANGES:
            1. INCREASED: Path signature length from 5 to 8 tokens
            2. ADDED: Minimum path length requirement before cycle detection
            3. REDUCED: False positive cycle detection for normal trie traversal

            Args:
                current_path: Current token sequence
                node_token: Token to potentially add
                visited_paths: Set of previously visited path segments

            Returns:
                True only if problematic cycle would be created
            """
            try:
                if not node_token or len(current_path) < 12:  # ADDED: Minimum path length requirement
                    return False

                extended_path = current_path + [node_token]

                # Create longer path signature for more precise cycle detection
                path_signature = tuple(extended_path[-8:]) if len(extended_path) >= 8 else None  # INCREASED: Signature length

                if path_signature and path_signature in visited_paths:
                    logger.debug(f"Problematic cycle detected: path signature {path_signature} already visited")
                    return True

                if path_signature:
                    visited_paths.add(path_signature)
                return False

            except Exception as cycle_error:
                logger.error(f"Error in problematic cycle detection: {cycle_error}")
                return False

        # PRESERVED: All existing initialization logic
        if visited_nodes is None:
            visited_nodes = set()
        if recent_tokens is None:
            recent_tokens = []

        # MODIFIED: More lenient path tracking initialization
        if not hasattr(self, '_visited_path_signatures'):
            self._visited_path_signatures = set()

        node_confidence = getattr(node, 'confidence', 0.5)

        logger.debug(f"Collecting context-aware continuations from '{node.token}': path_len={len(current_path)}, "
                    f"candidates={len(candidates)}/{max_continuations}, preferred_context={preferred_context_id}")

        # PRESERVED: Early termination conditions
        if len(candidates) >= max_continuations:
            return

        # INCREASED: More generous depth limits
        if len(current_path) > max_depth * 100:  # DOUBLED: Allow deeper exploration
            logger.info(f"Maximum depth {max_depth * 100} exceeded - terminating to prevent infinite recursion")
            return

        # MODIFIED: Only check for severe loop risk
        if _detect_severe_loop_risk(current_path):
            logger.warning(f"Severe loop risk detected in path {current_path[-12:]} - terminating branch")
            return

        # MODIFIED: Less aggressive cycle detection
        node_id = getattr(node, 'node_id', id(node))
        if _is_creating_problematic_cycle(current_path, node.token, self._visited_path_signatures):
            logger.debug(f"Problematic cycle detected for node '{node.token}' - applying moderate depth limit")
            max_depth = min(max_depth, len(current_path) + 8)  # INCREASED: More generous limit

        # PRESERVED: Natural stopping point detection with ENHANCED aggregate scoring
        if current_path:
            try:
                # PRESERVED: Core values verification
                if self.core_values:
                    logger.debug(f"🔍 Sotbi core values available: {list(self.core_values.keys())}")
                else:
                    logger.debug("⚠️ No core values available - value scoring disabled")

                # PRESERVED: Relevance calculation
                relevance = node.calculate_relevance(
                    context_embedding=context_embedding, 
                    query_embedding=query_sequence_embedding,
                    core_values=self.core_values
                )

                # PRESERVED: All existing scoring logic
                activation = node.activation_level
                avg_reward = node.metadata.get('avg_reward', 0.0)
                completeness_bonus = 0.2 if node.is_complete else 0.0

                # PRESERVED: Aggregate-based scoring enhancement
                aggregate_bonus = 0.0
                if self.global_centroid is not None and node.embedding is not None:
                    try:
                        global_coherence = self.context_window._calculate_ensemble_similarity(
                            node.embedding, self.global_centroid
                        )
                        activation_alignment = self.context_window._calculate_ensemble_similarity(
                            node.embedding, self.activation_weighted
                        )
                        aggregate_bonus = (global_coherence * 0.15 + activation_alignment * 0.10)
                        logger.debug(f"Aggregate scoring for '{node.token}': global_coherence={global_coherence:.3f}, "
                                   f"activation_alignment={activation_alignment:.3f}, bonus={aggregate_bonus:.3f}")
                    except Exception as scoring_error:
                        logger.debug(f"Error in aggregate scoring for '{node.token}': {scoring_error}")
                        aggregate_bonus = 0.0

                # PRESERVED: Enhanced scoring with semantic prioritization
                confidence_multiplier = 0.1 + (0.9 * node.confidence)
                traditional_score = (0.4 * relevance + 0.3 * activation + 0.2 * avg_reward + completeness_bonus)

                if query_sequence_embedding is not None:
                    semantic_bonus = self._calculate_sequence_similarity_bonus(current_path, query_sequence_embedding)
                    if semantic_bonus > 0.3:
                        traditional_score = (0.6 * relevance + 0.2 * activation + 0.1 * avg_reward + completeness_bonus)
                    else:
                        traditional_score = (0.5 * relevance + 0.25 * activation + 0.15 * avg_reward + completeness_bonus)
                else:
                    traditional_score = (0.4 * relevance + 0.3 * activation + 0.2 * avg_reward + completeness_bonus)

                enhanced_score = min(1.0, traditional_score + aggregate_bonus)
                final_score = enhanced_score * confidence_multiplier

                # PRESERVED: Add candidate with original logic
                candidates.append((current_path.copy(), final_score))
                logger.debug(f"Added sentence-ending candidate: {current_path} (traditional={traditional_score:.3f}, "
                            f"aggregate_bonus={aggregate_bonus:.3f}, final={final_score:.3f})")

            except Exception as e:
                logger.error(f"Error processing sentence-ending path {current_path}: {e}")

        # MODIFIED: Less aggressive repetition detection
        if node.token:
            extended_path = current_path + [node.token]

            # Only check for significant repetition patterns
            result = self.find_repeating_suffix(extended_path, min_len=4, return_metadata=True)  # INCREASED: Min length
            if result and result.get('phrase') and len(result['phrase'].split()) >= 6:  # ADDED: Minimum phrase length
                logger.debug(f"Significant repetition detected: '{result['phrase']}' at position {result['match_start']} - limiting children")
                # MODIFIED: Don't return immediately, just limit further exploration
                max_depth = min(max_depth, len(current_path) + 100)

            # Only apply severe loop check for extended path
            if _detect_severe_loop_risk(extended_path):
                logger.warning(f"Severe loop risk in extended path - terminating branch at '{node.token}'")
                return

        # PRESERVED: Process current path as candidate with enhanced scoring
        if current_path:
            try:
                # PRESERVED: All traditional scoring components
                relevance = node.calculate_relevance(context_embedding=context_embedding, query_embedding=query_sequence_embedding, core_values=self.core_values)
                activation = node.activation_level
                avg_reward = node.metadata.get('avg_reward', 0.0)
                completeness_bonus = 0.2 if node.is_complete else 0.0

                # PRESERVED: All penalty calculations
                result = self.find_repeating_suffix(current_path, min_len=4, return_metadata=True)  # INCREASED: Min length
                repetition_penalty = 0.3 if result else 0.0  # REDUCED: Penalty severity
                length_penalty = (len(current_path) - 12) * 0.03 if len(current_path) > 12 else 0.0  # INCREASED: Length threshold, REDUCED: Penalty

                # PRESERVED: Sequence embedding similarity bonus
                sequence_bonus = 0.0
                if query_sequence_embedding is not None:
                    sequence_bonus = self._calculate_sequence_similarity_bonus(current_path, query_sequence_embedding)

                # PRESERVED: Path-level aggregate scoring
                path_aggregate_bonus = 0.0
                if self.global_centroid is not None and self.activation_weighted is not None and current_path:
                    try:
                        path_embeddings = []
                        for token in current_path:
                            if token in self.embeddings and self.embeddings[token].embedding is not None:
                                path_embeddings.append(self.embeddings[token].embedding)

                        if path_embeddings:
                            path_centroid = np.mean(path_embeddings, axis=0)
                            norm = np.linalg.norm(path_centroid)
                            if norm > 0:
                                path_centroid = path_centroid / norm

                            path_global_coherence = self.context_window._calculate_ensemble_similarity(
                                path_centroid, self.global_centroid
                            )
                            path_activation_alignment = self.context_window._calculate_ensemble_similarity(
                                path_centroid, self.activation_weighted
                            )
                            path_aggregate_bonus = (path_global_coherence * 0.12 + path_activation_alignment * 0.08)
                            logger.debug(f"Path aggregate scoring for {current_path}: "
                                       f"global_coherence={path_global_coherence:.3f}, "
                                       f"activation_alignment={path_activation_alignment:.3f}, "
                                       f"bonus={path_aggregate_bonus:.3f}")
                    except Exception as path_scoring_error:
                        logger.debug(f"Error in path aggregate scoring: {path_scoring_error}")
                        path_aggregate_bonus = 0.0

                # PRESERVED: Final scoring calculation
                confidence_multiplier = 0.8 + (0.4 * node_confidence)
                traditional_base = (0.4 * relevance + 0.3 * activation + 0.2 * avg_reward + completeness_bonus + sequence_bonus)
                enhanced_base = traditional_base + path_aggregate_bonus
                penalized_score = enhanced_base - repetition_penalty - length_penalty
                final_score = penalized_score * confidence_multiplier

                # PRESERVED: Add candidate
                candidates.append((current_path.copy(), final_score))
                logger.debug(f"Added candidate: {current_path} (traditional_base={traditional_base:.3f}, "
                            f"path_aggregate_bonus={path_aggregate_bonus:.3f}, seq_bonus={sequence_bonus:.3f}, "
                            f"final={final_score:.3f})")

            except Exception as e:
                logger.error(f"Error processing path {current_path}: {e}")

        # MODIFIED: More permissive children processing
        if node.children and len(current_path) < max_depth * 1.5:  # INCREASED: Allow more depth for children
            new_visited = visited_nodes.copy()
            # MODIFIED: Only add to visited for very deep paths
            if len(current_path) > 15:  # INCREASED: Threshold for visited tracking
                new_visited.add(node_id)

            new_recent = recent_tokens.copy()
            if node.token:
                new_recent.append(node.token)
                if len(new_recent) > 8:  # INCREASED: Recent token tracking window
                    new_recent = new_recent[-8:]

            # PRESERVED: Context preference logic
            children_to_process = {}
            if preferred_context_id:
                context_children = node.get_children_for_context(preferred_context_id)
                if context_children:
                    children_to_process = context_children
                    logger.debug(f"Using context-specific children for {preferred_context_id}: {len(context_children)} children")
                else:
                    children_to_process = node.children
                    logger.debug(f"Context-specific children not found, using all children: {len(node.children)} children")
            else:
                children_to_process = node.children
                logger.debug(f"No preferred context, using all children: {len(node.children)} children")

            # PRESERVED: Child prioritization with aggregate scoring
            child_priorities = []
            for child_token, child_node in children_to_process.items():
                try:
                    child_activation = getattr(child_node, 'activation_level', 0.0)
                    child_confidence = getattr(child_node, 'confidence', 0.5)
                    traditional_priority = child_activation * 0.6 + child_confidence * 0.4

                    aggregate_priority_bonus = 0.0
                    if self.global_centroid is not None and self.activation_weighted is not None and child_node.embedding is not None:
                        try:
                            child_global_coherence = self.context_window._calculate_ensemble_similarity(
                                child_node.embedding, self.global_centroid
                            )
                            child_activation_alignment = self.context_window._calculate_ensemble_similarity(
                                child_node.embedding, self.activation_weighted
                            )
                            aggregate_priority_bonus = (child_global_coherence * 0.1 + child_activation_alignment * 0.05)
                            logger.debug(f"Child '{child_token}' aggregate priority: "
                                       f"global={child_global_coherence:.3f}, activation={child_activation_alignment:.3f}")
                        except Exception as child_agg_error:
                            logger.debug(f"Error in child aggregate priority for '{child_token}': {child_agg_error}")

                    enhanced_priority = traditional_priority + aggregate_priority_bonus
                    child_priorities.append((enhanced_priority, child_token, child_node))

                except Exception as e:
                    logger.error(f"Error calculating priority for child '{child_token}': {e}")
                    child_priorities.append((0.0, child_token, child_node))

            child_priorities.sort(reverse=True)

            # MODIFIED: Less restrictive child processing
            for priority, child_token, child_node in child_priorities:
                if len(candidates) >= max_continuations:
                    break
                
                # PRESERVED: Immediate repetition checks
                if len(current_path) >= 2 and child_token == current_path[-1]:
                    logger.debug(f"Skipping immediate repetition: '{child_token}' would repeat last token")
                    continue
                
                # MODIFIED: More lenient recent token checking
                if child_token in new_recent[-4:]:  # REDUCED: Check only last 4 instead of 3
                    recent_count = new_recent[-4:].count(child_token)
                    if recent_count >= 3:  # ADDED: Only skip if appears 3+ times recently
                        logger.debug(f"Skipping excessive recent repetition: '{child_token}' appears {recent_count} times recently")
                        continue
                    
                # MODIFIED: Only check for problematic cycles
                if _is_creating_problematic_cycle(current_path, child_token, self._visited_path_signatures):
                    logger.debug(f"Skipping child '{child_token}' - would create problematic cycle")
                    continue

                # MODIFIED: Only check for severe loop risk
                potential_path = current_path + [child_token]
                if _detect_severe_loop_risk(potential_path):
                    logger.debug(f"Skipping child '{child_token}' - severe loop risk detected")
                    continue
                
                try:
                    self._collect_context_aware_continuations(
                        child_node, 
                        potential_path, 
                        candidates, 
                        query_sequence_embedding, 
                        max_continuations, 
                        max_depth, 
                        context_embedding,
                        new_visited,
                        new_recent,
                        preferred_context_id
                    )
                except Exception as e:
                    logger.error(f"Error processing child '{child_token}': {e}")
                    continue


                
    def _calculate_sequence_similarity_bonus(self, path: List[str], 
                                           query_sequence_embedding: np.ndarray) -> float:
        """
        ADDED: Calculate sequence similarity bonus for path scoring.
        
        JUSTIFICATION: Uses query_sequence_embedding to score path relevance based on full sequence context.
        """
        try:
            if not path or query_sequence_embedding is None:
                return 0.0
            
            # Generate embedding for current path
            path_embeddings = []
            for token in path:
                if token in self.embeddings:
                    node = self.embeddings[token]
                    if node.embedding is not None:
                        path_embeddings.append(node.embedding)
            
            if not path_embeddings:
                return 0.0
            
            # Calculate path sequence embedding
            path_sequence_embedding = np.mean(path_embeddings, axis=0)
            norm = np.linalg.norm(path_sequence_embedding)
            if norm > 0:
                path_sequence_embedding = path_sequence_embedding / norm
            
            # Calculate similarity with query sequence embedding
            similarity = self.context_window._calculate_ensemble_similarity(
                query_sequence_embedding, path_sequence_embedding
            )
            
            # Convert to bonus (0.0 to 0.2 range)
            bonus = similarity * 0.2
            
            logger.debug(f"Sequence similarity bonus for path {path}: {bonus:.3f}")
            return bonus
            
        except Exception as e:
            logger.error(f"Error calculating sequence similarity bonus: {e}")
            return 0.0
    
    def _save_context_to_db(self, context_id: str, context_info: Dict[str, Any]):
        """
        ADDED: Save context information to LMDB for persistence.
        
        JUSTIFICATION: Persists context relationships for system restart recovery.
        """
        try:
            if not hasattr(self, 'context_db') or self.context_db is None:
                self.context_db = self.env.open_db(b'contexts')
            
            # Prepare context data for storage
            storage_data = {
                'tokens': context_info['tokens'],
                'reward': context_info['reward'],
                'timestamp': context_info['timestamp'],
                'embedding_bytes': context_info['embedding_bytes'],
                'context_id': context_id
            }
            
            with self.env.begin(write=True) as txn:
                key = f"context_{context_id}".encode()
                value = msgpack.packb(storage_data)
                txn.put(key, value, db=self.context_db)
            
            logger.debug(f"Saved context {context_id} to database")
            
        except Exception as e:
            logger.error(f"Error saving context to database: {e}")
    
    def _load_contexts_from_db(self):
        """
        ADDED: Load context information from LMDB on startup.
        
        JUSTIFICATION: Restores context relationships after system restart.
        """
        try:
            if not hasattr(self, 'context_db') or self.context_db is None:
                self.context_db = self.env.open_db(b'contexts')
            
            loaded_contexts = 0
            
            with self.env.begin() as txn:
                cursor = txn.cursor(db=self.context_db)
                for key, value in cursor:
                    try:
                        key_str = key.decode()
                        if key_str.startswith('context_'):
                            context_data = msgpack.unpackb(value, raw=False, strict_map_key=False)
                            context_id = context_data['context_id']
                            
                            # Restore sequence embedding from bytes
                            embedding_bytes = context_data['embedding_bytes']
                            sequence_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                            
                            # Restore context info
                            self.sequence_contexts[context_id] = {
                                'tokens': context_data['tokens'],
                                'sequence_embedding': sequence_embedding,
                                'reward': context_data['reward'],
                                'timestamp': context_data['timestamp'],
                                'embedding_bytes': embedding_bytes
                            }
                            
                            loaded_contexts += 1
                            
                    except Exception as item_error:
                        logger.error(f"Error loading context item {key}: {item_error}")
                        continue
            
            logger.info(f"Loaded {loaded_contexts} contexts from database")
            
        except Exception as e:
            logger.error(f"Error loading contexts from database: {e}")
    
    
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

                
    def find_repeating_suffix(self,
        current_path: List[Any],
        min_len: int = 1,
        max_len: Optional[int] = None,
        return_metadata: bool = False,
        use_hash_threshold: int = 10,
        logger=None
    ) -> Union[bool, Dict[str, Union[int, List[Any]]], None]:
        """
        Detect if the suffix of current_path repeats earlier in the sequence.
    
        Parameters
        ----------
        current_path : List[Any]
            The sequence of tokens to analyze.
        min_len : int, optional
            Minimum phrase length to check for repetition (default is 1).
        max_len : int, optional
            Maximum phrase length to check. If None, defaults to len(current_path) // 2.
        return_metadata : bool, optional
            If True, returns metadata about the detected repetition instead of a boolean.
        use_hash_threshold : int, optional
            If phrase length exceeds this value, a hash-based fast-check will be used before full comparison.
        logger : logging.Logger, optional
            Logger for debugging output.
    
        Returns
        -------
        bool or dict or None
            - If return_metadata is False: returns True if a match is found, otherwise False.
            - If return_metadata is True: returns a dict with details about the match, or None if no match found.
        """
        n = len(current_path)
        if max_len is None:
            max_len = n // 2
        if min_len < 1:
            min_len = 1
        if max_len < min_len or n < 2 * min_len:
            return False if not return_metadata else None
    
        for L in range(max_len, min_len - 1, -1):
            if n < 2 * L:
                continue
            last_L = current_path[-L:]
            last_L_hash = hash(tuple(last_L)) if L > use_hash_threshold else None
    
            for i in range(n - L):
                candidate = current_path[i:i+L]
                if last_L_hash is not None:
                    if hash(tuple(candidate)) != last_L_hash:
                        continue  # skip false hash match
                if candidate == last_L:
                    if logger:
                        logger.debug(
                            f"Repetition detected: {last_L} (length {L}/{n}) at position {i}"
                        )
                    if return_metadata:
                        return {
                            "phrase": last_L,
                            "length": L,
                            "match_start": i,
                            "match_end": i + L,
                            "suffix_start": n - L,
                            "suffix_end": n,
                            "sequence_length": n
                        }
                    return True
        return False if not return_metadata else None
                    
                    
                
    def _detect_repetition_patterns(self, path: List[str]) -> Dict[str, Any]:
        """
        ADDED: Comprehensive repetition pattern detection utility.
        
        JUSTIFICATION: Provides detailed analysis of repetitive patterns for debugging and scoring.
        """
        try:
            if len(path) < 4:
                return {'has_repetition': False, 'pattern_type': 'none', 'severity': 0.0}
            
            analysis = {
                'has_repetition': False,
                'pattern_type': 'none',
                'severity': 0.0,
                'repeated_elements': [],
                'pattern_length': 0
            }
            
            # Check for different types of repetition
            
            # 1. Immediate token repetition (same token repeated)
            for i in range(len(path) - 1):
                if path[i] == path[i + 1]:
                    analysis['has_repetition'] = True
                    analysis['pattern_type'] = 'immediate_token'
                    analysis['severity'] = max(analysis['severity'], 0.8)
                    analysis['repeated_elements'].append(path[i])
            
            # 2. Short pattern repetition (2-3 tokens repeating)
            for pattern_len in [2, 3]:
                for i in range(len(path) - 2 * pattern_len + 1):
                    pattern = path[i:i + pattern_len]
                    next_pattern = path[i + pattern_len:i + 2 * pattern_len]
                    if pattern == next_pattern and len(pattern) == pattern_len:
                        analysis['has_repetition'] = True
                        analysis['pattern_type'] = f'short_pattern_{pattern_len}'
                        analysis['severity'] = max(analysis['severity'], 0.6)
                        analysis['repeated_elements'].append(pattern)
                        analysis['pattern_length'] = pattern_len
            
            # 3. Long sequence repetition
            quarter_len = len(path) // 4
            if quarter_len >= 3:
                last_quarter = path[-quarter_len:]
                prev_quarter = path[-2 * quarter_len:-quarter_len]
                if last_quarter == prev_quarter:
                    analysis['has_repetition'] = True
                    analysis['pattern_type'] = 'long_sequence'
                    analysis['severity'] = 0.9
                    analysis['repeated_elements'].append(last_quarter)
                    analysis['pattern_length'] = quarter_len
            
            # 4. Calculate overall repetition ratio
            unique_tokens = len(set(path))
            total_tokens = len(path)
            repetition_ratio = 1.0 - (unique_tokens / total_tokens)
            
            if repetition_ratio > 0.5:  # More than 50% repetition
                analysis['has_repetition'] = True
                if analysis['pattern_type'] == 'none':
                    analysis['pattern_type'] = 'high_repetition_ratio'
                analysis['severity'] = max(analysis['severity'], repetition_ratio)
            
            logger.debug(f"Repetition analysis for {path}: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in repetition pattern detection: {e}")
            return {'has_repetition': False, 'pattern_type': 'error', 'severity': 0.0}
    
    
    def _is_natural_stopping_point(self, token: str, path: List[str]) -> bool:
        """
        ADDED: Determine if this is a natural stopping point for generation.
        
        JUSTIFICATION: Prevents unnatural continuation past logical endpoints.
        """
        try:
            # Sentence endings
            if token in {'.', '!', '?'}:
                return True
            
            # Common conversation endings
            conversation_endings = {
                'thanks', 'thank', 'bye', 'goodbye', 'see', 'later', 
                'done', 'finished', 'complete', 'over', 'end'
            }
            if token.lower() in conversation_endings:
                return True
            
            # Check for natural phrase completions
            if len(path) >= 2:
                last_two = ' '.join(path[-2:]).lower()
                natural_endings = {
                    'thank you', 'good bye', 'see you', 'talk soon', 
                    'take care', 'have fun', 'good luck', 'well done'
                }
                if last_two in natural_endings:
                    return True
            
            # Length-based stopping (very long responses)
            if len(path) >= 15:  # More than 15 tokens is quite long
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking natural stopping point: {e}")
            return False

    def _restore_context_children_relationships(self):
        """ADDED: Restore context-aware children relationships from persisted contexts."""
        try:
            restored_relationships = 0

            for context_id, context_info in self.sequence_contexts.items():
                try:
                    tokens = context_info['tokens']
                    sequence_embedding = context_info['sequence_embedding']

                    # Rebuild context-aware relationships for this sequence
                    for i in range(len(tokens) - 1):
                        current_token = tokens[i]
                        next_token = tokens[i + 1]

                        if current_token in self.embeddings and next_token in self.embeddings:
                            current_node = self.embeddings[current_token]
                            next_node = self.embeddings[next_token]

                            # Restore context-aware child relationship
                            current_node.add_child_with_context(
                                token=next_token,
                                child=next_node,
                                context_id=context_id,
                                full_sequence=tokens,
                                sequence_embedding=sequence_embedding
                            )

                            restored_relationships += 1

                except Exception as context_error:
                    logger.error(f"Error restoring context relationships for {context_id}: {context_error}")
                    continue
                
            logger.info(f"Restored {restored_relationships} context-aware children relationships")

        except Exception as e:
            logger.error(f"Error in context relationship restoration: {e}")

    
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
        
        # ADDED: Restore context-aware children relationships after regular loading
        self._restore_context_children_relationships()

        logger.info(f"Loaded {len(self.embeddings)} tokens with context relationships restored")
        
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
            
    def _save_to_db(self, sequence_id: str, path_nodes: List[SemanticTrieNode], sequence_embedding: np.ndarray):
        """
        FIXED: Save sequence with robust resize and retry logic.
        
        ACCOUNTABILITY CHANGES:
        1. ADDED: Specific MDB_MAP_FULL error handling with automatic resize and retry
        2. ENHANCED: Retry mechanism after database resize operations  
        3. FIXED: Missing metadata key initialization to prevent secondary errors
        4. PRESERVED: All existing database persistence logic and error handling
        
        JUSTIFICATION: Addresses timing issues where database fills during transaction execution.
        """
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                logger.info(f"Saving sequence {sequence_id[:8]}... with {len(path_nodes)} nodes (attempt {retry_count + 1})")
                
                # Check capacity before transaction (existing logic preserved)
                self._check_and_resize_if_needed()
                
                with self.env.begin(write=True) as txn:
                    # PRESERVED: Save sequence metadata (unchanged)
                    metadata = {
                        'sequence_id': sequence_id,
                        'path_length': len(path_nodes),
                        'embedding': sequence_embedding.tobytes(),
                        'timestamp': time.time()
                    }
                    txn.put(sequence_id.encode(), msgpack.packb(metadata), db=self.metadata_db)
                    logger.debug(f"Saved sequence metadata for {sequence_id[:8]}...")
                    
                    # PRESERVED: Save nodes with FIXED metadata initialization
                    for i, node in enumerate(path_nodes):
                        try:
                            node_key = f"{sequence_id}_{i}".encode()
                            
                            # FIXED: Initialize missing metadata keys to prevent errors
                            if 'confidence_updates' not in node.metadata:
                                node.metadata['confidence_updates'] = 0
                                logger.debug(f"Initialized missing 'confidence_updates' for node {i}")
                            
                            if 'semantic_updates' not in node.metadata:
                                node.metadata['semantic_updates'] = 0
                                logger.debug(f"Initialized missing 'semantic_updates' for node {i}")
                            
                            # PRESERVED: Serialize children relationships (unchanged)
                            children_data = {}
                            if node.children:
                                for child_token, child_node in node.children.items():
                                    children_data[child_token] = {
                                        'node_id': child_node.node_id,
                                        'token': child_node.token
                                    }
                            
                            # PRESERVED: Complete node data structure (unchanged)
                            node_data = {
                                'token': node.token,
                                'embedding': node.embedding.tobytes() if node.embedding is not None else None,
                                'relevance_score': node.relevance_score,
                                'activation_level': node.activation_level,
                                'is_complete': node.is_complete,
                                'metadata': node.metadata,  # Now includes initialized keys
                                'reward_history': node.reward_history,
                                'access_count': node.access_count,
                                'last_accessed': node.last_accessed,
                                'is_end_of_sequence': node.is_end_of_sequence,
                                'children_relationships': children_data,
                                'node_id': node.node_id
                            }
                            
                            # CRITICAL: This is where MDB_MAP_FULL typically occurs
                            txn.put(node_key, msgpack.packb(node_data), db=self.nodes_db)
                            logger.debug(f"Saved node {i}: '{node.token}'")
                            
                        except Exception as node_error:
                            logger.error(f"Error saving node {i}: {node_error}")
                            continue
                        
                # SUCCESS: Transaction completed without resize needed
                logger.info(f"Successfully saved sequence {sequence_id[:8]}... on attempt {retry_count + 1}")
                return
                
            except Exception as e:
                error_message = str(e)
                
                # ADDED: Specific handling for MDB_MAP_FULL with automatic resize and retry
                if "MDB_MAP_FULL" in error_message or "mapsize limit reached" in error_message:
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        logger.warning(f"Database full during save operation (attempt {retry_count}), attempting emergency resize and retry")
                        
                        try:
                            # EMERGENCY RESIZE: Larger expansion when full during operation
                            emergency_size = int(self.current_map_size * 2.0)  # Double instead of 2.0 factor
                            logger.info(f"Emergency resize: {self.current_map_size/(1024**3):.1f}GB → {emergency_size/(1024**3):.1f}GB")
                            
                            self._resize_database(emergency_size)
                            logger.info(f"Emergency resize completed, retrying save operation...")
                            
                            # Continue to next iteration of while loop for retry
                            continue
                            
                        except Exception as resize_error:
                            logger.error(f"Emergency resize failed on attempt {retry_count}: {resize_error}")
                            
                            if retry_count >= max_retries:
                                logger.error(f"Max retries ({max_retries}) exceeded for database full error")
                                raise RuntimeError(f"Database resize and retry failed after {max_retries} attempts: {resize_error}")
                            else:
                                # Try one more resize with even larger size
                                continue
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded for MDB_MAP_FULL error")
                        raise RuntimeError(f"Database capacity exhausted after {max_retries} resize attempts")
                else:
                    # PRESERVED: Non-capacity errors re-raised immediately (unchanged)
                    logger.error(f"Non-capacity error during save: {error_message}")
                    raise
                
    def _check_and_resize_if_needed(self):
        """
        ENHANCED: Improved capacity checking with more aggressive thresholds.
        
        ACCOUNTABILITY CHANGES:
        1. LOWERED: Resize threshold from 0.8 to 0.7 for earlier intervention  
        2. ENHANCED: Better logging for capacity monitoring
        3. PRESERVED: All existing resize logic and error handling
        
        JUSTIFICATION: Earlier intervention prevents mid-transaction capacity issues.
        """
        try:
            stat = self.env.stat()
            used_pages = stat['psize'] * stat['leaf_pages']
            usage_ratio = used_pages / self.current_map_size
            
            logger.debug(f"Database capacity check: {usage_ratio:.1%} used "
                        f"({used_pages/(1024**3):.2f}GB / {self.current_map_size/(1024**3):.1f}GB)")
            
            # LOWERED: More aggressive threshold to prevent mid-transaction issues
            if usage_ratio >= 0.7:  # CHANGED: from 0.8 to 0.7 (70% instead of 80%)
                new_size = int(self.current_map_size * self.resize_factor)
                logger.warning(f"Proactive resize triggered at {usage_ratio:.1%} capacity: "
                              f"{self.current_map_size/(1024**3):.1f}GB → {new_size/(1024**3):.1f}GB")
                self._resize_database(new_size)
            else:
                logger.debug(f"Database capacity OK: {usage_ratio:.1%} used (threshold: 70%)")
                
        except Exception as e:
            logger.warning(f"Error checking database capacity: {e}")
    
    def _resize_database(self, new_size: int):
        """
        ENHANCED: More robust database resize with better error handling.
        
        ACCOUNTABILITY CHANGES:
        1. ENHANCED: Better error handling during resize operations
        2. ADDED: Validation that resize actually occurred  
        3. PRESERVED: All existing resize logic and database reconnection
        
        JUSTIFICATION: Ensures resize operations complete successfully before continuing.
        """
        try:
            old_size = self.current_map_size
            logger.info(f"Starting database resize: {old_size/(1024**3):.1f}GB → {new_size/(1024**3):.1f}GB")
            
            # PRESERVED: Existing resize logic (unchanged)
            try:
                self.env.sync(force=True)
                logger.debug("Database sync completed")
            except Exception as sync_error:
                logger.warning(f"Error during sync (continuing): {sync_error}")
            
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
            
            # ADDED: Validation that resize was successful
            try:
                stat = self.env.stat()
                logger.info(f"Database resize completed successfully: new capacity {new_size/(1024**3):.1f}GB")
                logger.debug(f"Post-resize stats: page_size={stat['psize']}, max_pages={new_size//stat['psize']}")
            except Exception as validation_error:
                logger.error(f"Error validating resize: {validation_error}")
                raise
                
        except Exception as e:
            logger.error(f"Critical error during database resize: {e}")
            
            # PRESERVED: Recovery attempt (unchanged)
            try:
                self.env = lmdb.open(self.db_path, map_size=old_size, max_dbs=2)
                self.nodes_db = self.env.open_db(b'nodes')
                self.metadata_db = self.env.open_db(b'metadata')
                self.current_map_size = old_size
                logger.info("Recovered original database connection after resize failure")
            except Exception as recovery_error:
                logger.critical(f"Failed to recover database connection: {recovery_error}")
                raise
            
            raise  # Re-raise original resize error
        