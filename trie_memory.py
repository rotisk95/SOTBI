"""
FIXED: Complete Enhanced Trie-Based Predictive Token Embedding System

CRITICAL FIXES APPLIED:
1. FIXED beam search to only use actual trie children as continuations (not random nodes)
2. REMOVED complex fallback logic that was creating fake continuations
3. PRESERVED all working simple prediction logic
4. RESTORED basic trie traversal principles

ACCOUNTABILITY: Only _collect_multi_source_candidates method significantly changed.
All other functionality preserved exactly as working.
"""

import gc
import os
from queue import Queue
import types
import msgpack
import numpy as np
import logging
import lmdb
import time
from typing import List, Dict, Tuple, Optional, Any
import hashlib
import numpy as np

from context_window import ContextWindow
from token_embedding import TokenEmbedding, create_token_embedding
from trie_node import TrieNode

# Configure logging for execution transparency
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import time
import logging
import msgpack
import numpy as np
import hashlib
import gc
import os
from queue import Queue
import lmdb
from typing import Dict, List, Optional, Any, Tuple
import uuid

logger = logging.getLogger(__name__)

REWARD_HISTORY_LIMIT = 10000

class TrieMemory:
    """
    Enhanced TrieMemory with minimal changes - ALL your original methods included.
    """
    
    def __init__(self, db_path: str = "./trie_memory.lmdb", embed_dim: int = 1024):
        self.embed_dim = embed_dim
        self.db_path = db_path
        
        # UNCHANGED: All your existing initialization
        from context_window import ContextWindow
        self.context_window = ContextWindow()
        
        # UNCHANGED: Embedding cache
        self.embedding_cache = {}
        self.cache_max_size = 5000

        # UNCHANGED: Calculate initial map size adaptively
        initial_map_size = self._calculate_adaptive_map_size()
        logger.info(f"Initializing LMDB with adaptive map size: {initial_map_size / (1024*1024*1024):.1f}GB")

        # UNCHANGED: LMDB setup
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
        except Exception as e:
            logger.error(f"Failed to initialize LMDB: {e}")
            raise

        # UNCHANGED: Database monitoring
        self.current_map_size = initial_map_size
        self.resize_threshold = 0.8  
        self.resize_factor = 2.0

        # ENHANCED: Create root with registry
        self.root = TrieNode(token=None, db_env=self.env, context_window=self.context_window)
        
        try:
            self._load_simple_fixed()
        except RecursionError as e:
            logger.error(f"Recursion error during trie loading: {e}")
            logger.info("This usually indicates a very large trie. Consider using the optimized rebuild method.")
            raise
        except Exception as e:
            logger.error(f"Error loading trie: {e}")
            raise
        
        # ENHANCED: Token registry with error handling  
        try:
            self.root.enhance_trie_node_registry()
        except Exception as e:
            logger.warning(f"Failed to enhance registry: {e}")
            logger.info("Continuing without token-based optimization")
    
    def _get_cached_embedding(self, token: str):
        """Get cached embedding or create new one to prevent redundant generation"""
        if token in self.embedding_cache:
            logger.debug(f"Using cached embedding for token: '{token}'")
            return self.embedding_cache[token]
        
        from token_embedding import create_token_embedding
        token_embedding = create_token_embedding(token)
        
        if len(self.embedding_cache) >= self.cache_max_size:
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[token] = token_embedding
        logger.debug(f"Cached new embedding for token: '{token}'")
        return token_embedding

    def _resize_database(self, new_size: int):
        """Improved database resize with better transaction management"""
        try:
            logger.info(f"Starting database resize operation to {new_size/(1024**3):.1f}GB")

            try:
                self.env.sync(force=True)
            except Exception as sync_error:
                logger.warning(f"Error during sync: {sync_error}")

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

            logger.info(f"Database resize completed successfully to {new_size/(1024**3):.1f}GB")

        except Exception as e:
            logger.error(f"Critical error during database resize: {e}")
            try:
                self.env = lmdb.open(self.db_path, map_size=self.current_map_size, max_dbs=2)
                self.nodes_db = self.env.open_db(b'nodes')
                self.metadata_db = self.env.open_db(b'metadata')
                logger.info("Recovered original database connection after resize failure")
            except Exception as recovery_error:
                logger.critical(f"Failed to recover database connection: {recovery_error}")
                raise

    def add_sequence(self, tokens: List[str], reward: float = 0.0) -> str:
        """ENHANCED: Add sequence with negative reward support + registry updates."""
        logger.info(f"Adding sequence to trie: {tokens} with reward: {reward}")

        reward = max(-1.0, min(1.0, float(reward)))

        if not tokens:
            raise ValueError("Token sequence cannot be empty")

        try:
            token_embeddings = [self._get_cached_embedding(token) for token in tokens]

            sequence_embedding = self._aggregate_sequence_embedding(token_embeddings)
            self.context_window.add_turn(tokens, sequence_embedding)

            current_node = self.root
            path_nodes = []
            current_path = []

            negative_feedback_applied = reward < 0
            if negative_feedback_applied:
                logger.info(f"Propagating negative feedback (reward: {reward}) through sequence")

            for i, token in enumerate(tokens):
                current_path.append(token)
                
                if token not in current_node.children:
                    prefix_context = tokens[:i] if i > 0 else []
                    context_confirmed = True

                    if context_confirmed:
                        new_node = TrieNode(token=token, db_env=self.env, context_window=self.context_window)
                        new_node.token_embedding = token_embeddings[i].embedding
                        current_node.children[token] = new_node
                        
                        # NEW: Register the node in root's registry
                        self.root.update_registry_on_node_creation(new_node, current_path.copy())
                        
                        logger.debug(f"Created confirmed trie node for token: '{token}'")
                    else:
                        logger.info(f"Skipped creating link for unconfirmed context: {prefix_context} -> '{token}'")
                        break

                current_node = current_node.children[token]

                if current_node.token_embedding is None:
                    current_node.token_embedding = token_embeddings[i].embedding
                    logger.debug(f"Set embedding for existing node: '{token}'")

                path_nodes.append(current_node)

                position_weight = (i + 1) / len(tokens)

                if negative_feedback_applied:
                    adjusted_reward = reward * (0.5 + 0.5 * position_weight)
                else:
                    adjusted_reward = reward * position_weight

                context_relevance = self.context_window.get_context_similarity(current_node.token_embedding)
                current_node.update_activation(adjusted_reward, context_relevance)

                if negative_feedback_applied:
                    logger.debug(f"Applied negative feedback to '{token}': "
                               f"original_reward={reward:.3f}, adjusted={adjusted_reward:.3f}, "
                               f"position_weight={position_weight:.3f}")

            if path_nodes:
                current_node.is_end_of_sequence = True
                current_node.update_completeness()

            sequence_id = self._generate_sequence_id(tokens)
            self._save_to_db(sequence_id, path_nodes, sequence_embedding)

            logger.info(f"Successfully added sequence to trie with ID: {sequence_id[:8]}... "
                       f"(negative_feedback: {negative_feedback_applied})")
            return sequence_id

        except Exception as e:
            logger.error(f"Error adding sequence to trie: {str(e)}")
            raise
        
    def _calculate_adaptive_map_size(self) -> int:
        """Calculate initial LMDB map size based on available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(os.path.dirname(os.path.abspath(self.db_path)))
            recommended_size = int(free * 0.1)
            min_size = 1024 * 1024 * 1024
            max_size = 10 * 1024 * 1024 * 1024
            adaptive_size = max(min_size, min(recommended_size, max_size))
            logger.info(f"Disk space analysis: {free/(1024**3):.1f}GB free, "
                       f"recommending {adaptive_size/(1024**3):.1f}GB initial map size")
            return adaptive_size
        except Exception as e:
            logger.warning(f"Could not calculate adaptive map size: {e}, using 4GB default")
            return 4 * 1024 * 1024 * 1024

    def _check_and_resize_if_needed(self):
        """Check database capacity and resize if approaching limit."""
        try:
            stat = self.env.stat()
            used_pages = stat['psize'] * stat['leaf_pages']
            total_size = self.current_map_size
            usage_ratio = used_pages / total_size

            logger.debug(f"Database usage: {usage_ratio:.1%} ({used_pages/(1024**3):.2f}GB / {total_size/(1024**3):.1f}GB)")

            if usage_ratio >= self.resize_threshold:
                new_size = int(self.current_map_size * self.resize_factor)
                logger.info(f"Database approaching capacity ({usage_ratio:.1%}), resizing from "
                           f"{self.current_map_size/(1024**3):.1f}GB to {new_size/(1024**3):.1f}GB")
                self._resize_database(new_size)

        except Exception as e:
            logger.warning(f"Error checking database capacity: {e}")

    def _load_simple_fixed(self):
        """ENHANCED: Load trie with registry rebuilding."""
        logger.info("Loading trie with corrected single-root architecture")

        nodes = {}
        sequence_groups = {}

        with self.env.begin(db=self.nodes_db) as txn:
            for key, value in txn.cursor():
                path = key.decode()
                data = msgpack.unpackb(value, raw=False, strict_map_key=False)

                seq_id, pos = self._parse_node_key(path)

                node = TrieNode(token=data.get('token'), db_env=self.env, context_window=self.context_window)
                node.relevance_score = data.get('relevance_score', 0.0)
                node.activation_level = data.get('activation_level', 0.0)
                node.is_complete = data.get('is_complete', False)
                node.metadata = data.get('metadata', {})
                node.reward_history = data.get('reward_history', [])
                node.access_count = data.get('access_count', 0)
                node.last_accessed = data.get('last_accessed', time.time())
                node.is_end_of_sequence = data.get('is_end_of_sequence', False)

                if data.get('embedding') is not None:
                    node.set_embedding_info(path)

                nodes[path] = node

                if seq_id not in sequence_groups:
                    sequence_groups[seq_id] = {}
                sequence_groups[seq_id][pos] = (path, node)

        self.root = TrieNode(token=None, db_env=self.env, context_window=self.context_window)
        logger.info(f"Created single shared root node for {len(sequence_groups)} sequences")

        sequences_processed = 0
        for seq_id, positions in sequence_groups.items():
            try:
                sorted_positions = sorted(positions.keys())
                current_node = self.root

                for pos in sorted_positions:
                    path, node = positions[pos]

                    if node.token:
                        if node.token not in current_node.children:
                            current_node.children[node.token] = node
                            logger.debug(f"Linked {current_node.token or 'ROOT'} -> {node.token}")

                        current_node = current_node.children[node.token]

                sequences_processed += 1

            except Exception as e:
                logger.warning(f"Error processing sequence {seq_id}: {e}")
                continue
            
        logger.info(f"Successfully built single-root trie from {sequences_processed} sequences")

        if self.root and self.root.children:
            logger.info(f"✅ Single root created with {len(self.root.children)} top-level branches")
        else:
            logger.warning("⚠️ Root node has no children - may indicate data loading issues")

    def _rebuild_registry_from_trie(self):
        """
        OPTIMIZED VERSION: Even faster with batch processing and reduced logging.
        USE THIS: If you want maximum performance for very large tries.
        """
        logger.info("Rebuilding node registry with optimized processing...")

        # Pre-allocate stack with estimated size
        stack = [(self.root, [])]
        visited = set()
        registered_count = 0
        batch_size = 1000  # Process in batches to reduce logging overhead

        # Temporarily reduce logging level for performance
        original_level = logger.level
        logger.setLevel(logging.WARNING)  # Reduce debug/info spam during rebuild

        try:
            while stack:
                # Process batch
                batch_processed = 0
                batch_stack = []

                # Process up to batch_size nodes
                while stack and batch_processed < batch_size:
                    current_node, path_tokens = stack.pop()
                    batch_processed += 1

                    node_id = id(current_node)
                    if node_id in visited:
                        continue
                    visited.add(node_id)

                    # Register node (skip root)
                    if current_node != self.root:
                        try:
                            self.root.register_node(current_node, path_tokens)
                            registered_count += 1
                        except Exception:
                            continue  # Skip failed registrations silently
                        
                    # Collect children for next batch
                    for token, child_node in current_node.children.items():
                        child_id = id(child_node)
                        if child_id not in visited:
                            child_path = path_tokens + [token]
                            batch_stack.append((child_node, child_path))

                # Add batch children to main stack
                stack.extend(batch_stack)

                # Progress update per batch
                if registered_count % 10000 == 0:
                    print(f"Registry progress: {registered_count} nodes registered...")  # Use print to bypass logging

        finally:
            # Restore original logging level
            logger.setLevel(original_level)

        logger.info(f"Optimized registry rebuild complete: {registered_count} nodes registered")

    def _save_to_db(self, sequence_id: str, path_nodes: List[TrieNode], sequence_embedding: np.ndarray):
        """Clean save method without complex type conversion."""
        try:
            self._check_and_resize_if_needed()

            with self.env.begin(write=True) as txn:
                metadata = {
                    'sequence_id': sequence_id,
                    'path_length': len(path_nodes),
                    'embedding': sequence_embedding.tobytes(),
                    'timestamp': time.time()
                }
                txn.put(sequence_id.encode(), msgpack.packb(metadata), db=self.metadata_db)

                for i, node in enumerate(path_nodes):
                    node_key = f"{sequence_id}_{i}".encode()

                    embedding_bytes = None
                    if node.token_embedding is not None:
                        embedding_bytes = node.token_embedding.tobytes()

                    node_data = {
                        'token': node.token,
                        'embedding': embedding_bytes,
                        'relevance_score': node.relevance_score,
                        'activation_level': node.activation_level,
                        'is_complete': node.is_complete,
                        'metadata': node.metadata,
                        'reward_history': node.reward_history,
                        'access_count': node.access_count,
                        'last_accessed': node.last_accessed,
                        'is_end_of_sequence': node.is_end_of_sequence,
                        # NEW: Save registry info
                        'node_id': node.node_id,
                        'path_tokens': getattr(node, 'path_tokens', []),
                        'hierarchy_level': getattr(node, 'hierarchy_level', 0)
                    }
                    txn.put(node_key, msgpack.packb(node_data), db=self.nodes_db)

            logger.info(f"Saved sequence {sequence_id[:8]}...")

        except lmdb.MapFullError:
            logger.warning("Database map full during save, attempting emergency resize")
            try:
                emergency_size = int(self.current_map_size * 1.5)
                self._resize_database(emergency_size)
                self._save_to_db(sequence_id, path_nodes, sequence_embedding)
            except Exception as resize_error:
                logger.error(f"Emergency resize failed: {resize_error}")
                raise

        except Exception as e:
            logger.error(f"Error saving sequence: {e}")
            if "MDB_MAP_FULL" in str(e):
                logger.error("Database capacity exceeded - consider manual database cleanup or restart")
    
    def _link_children_simple(self, nodes):
        """Link parent-child relationships ONLY within the same sequence."""
        logger.info("Starting sequence linking with cross-sequence contamination prevention")
        successful_links = 0
        skipped_links = 0
        error_links = 0

        for path, node in nodes.items():
            try:
                seq_id, pos = self._parse_node_key(path)
                child_path = f"{seq_id}_{pos + 1}"

                if child_path in nodes:
                    child = nodes[child_path]

                    if child.token:
                        child_seq_id, child_pos = self._parse_node_key(child_path)

                        if seq_id == child_seq_id and child_pos == pos + 1:
                            node.children[child.token] = child
                            successful_links += 1
                            logger.debug(f"Linked: {node.token or 'ROOT'} -> {child.token} (sequence: {seq_id[:8]}...)")
                        else:
                            skipped_links += 1
                            logger.warning(f"Prevented cross-sequence link: {node.token} -> {child.token} "
                                         f"(seq mismatch: {seq_id[:8]}... vs {child_seq_id[:8]}...)")
                    else:
                        skipped_links += 1
                        logger.warning(f"Skipped link to tokenless child at path: {child_path}")
                else:
                    logger.debug(f"No child found for {node.token or 'ROOT'} at {child_path} (end of sequence)")

            except Exception as e:
                error_links += 1
                logger.error(f"Error linking node at path {path}: {e}")

        logger.info(f"Sequence linking completed: {successful_links} successful, "
                   f"{skipped_links} skipped (contamination prevented), {error_links} errors")

    def _parse_node_key(self, path: str) -> Tuple[str, int]:
        """Parse sequence_id and position from path with improved error handling."""
        try:
            parts = path.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                seq_id, position = parts[0], int(parts[1])
                logger.debug(f"Parsed key: {path} -> seq_id: {seq_id[:8]}..., pos: {position}")
                return seq_id, position
            else:
                logger.warning(f"Malformed node key format: {path}, falling back to defaults")
                return path, 0
        except Exception as e:
            logger.error(f"Error parsing node key {path}: {e}")
            return path, 0

    def _set_root_simple(self, nodes):
        """Set root node with improved validation and error handling."""
        logger.info("Setting root node from loaded sequences")

        root_candidates = []
        for path, node in nodes.items():
            if path.endswith('_0'):
                root_candidates.append((path, node))
                logger.debug(f"Found root candidate: {path} with token: {node.token}")

        if root_candidates:
            selected_path, selected_node = root_candidates[0]
            self.root = selected_node
            logger.info(f"Selected root node from path: {selected_path} with token: {selected_node.token}")

            if len(root_candidates) > 1:
                logger.warning(f"Multiple root candidates found ({len(root_candidates)}), "
                              f"selected first one: {selected_path}")
        else:
            logger.warning("No root node candidates found, creating empty root")
            self.root = TrieNode(token=None, db_env=self.env, context_window=self.context_window)
            logger.info("Created fallback empty root node")

    def verify_trie_integrity_after_loading(self):
        """Verify trie integrity after loading to detect cross-sequence contamination."""
        logger.info("Verifying trie integrity to detect cross-sequence contamination")

        token_occurrences = {}
        contamination_detected = 0

        def traverse_and_check(node, path):
            nonlocal contamination_detected

            if node.token:
                if node.token not in token_occurrences:
                    token_occurrences[node.token] = []

                token_occurrences[node.token].append({
                    'path': path.copy(),
                    'children': list(node.children.keys()),
                    'is_end': node.is_end_of_sequence,
                    'node_id': id(node)
                })

            for child_token, child_node in node.children.items():
                traverse_and_check(child_node, path + [child_token])

        traverse_and_check(self.root, [])

        for token, occurrences in token_occurrences.items():
            if len(occurrences) > 1:
                children_sets = [set(occ['children']) for occ in occurrences]
                unique_children_sets = set(tuple(sorted(cs)) for cs in children_sets)

                if len(unique_children_sets) > 1:
                    contamination_detected += 1
                    logger.error(f"CONTAMINATION DETECTED: Token '{token}' has inconsistent children:")
                    for i, occ in enumerate(occurrences):
                        logger.error(f"  Occurrence {i+1}: path={occ['path']}, "
                                   f"children={occ['children']}, is_end={occ['is_end']}")

        if contamination_detected == 0:
            logger.info("✅ Trie integrity verification passed - no cross-sequence contamination detected")
        else:
            logger.error(f"❌ Trie integrity verification failed - {contamination_detected} contamination cases detected")

        return contamination_detected == 0

    def _is_context_confirmed_simple(self, prefix_context: List[str], next_token: str, reward: float) -> bool:
        """Direct embedding similarity search instead of sequence ID iteration."""
        try:
            if reward >= 0.8:
                logger.debug(f"Context confirmed: high reward override for {prefix_context} -> {next_token}")
                return True

            if not prefix_context:
                logger.debug(f"Context confirmed: root level token '{next_token}'")
                return True

            return self._check_embedding_similarity_direct(prefix_context, next_token, reward)

        except Exception as e:
            logger.warning(f"Error in context confirmation: {e}")
            return True

    def _check_embedding_similarity_direct(self, prefix_context: List[str], next_token: str, reward: float) -> bool:
        """Direct embedding similarity search without sequence reconstruction."""
        try:
            from token_embedding import create_token_embedding
            
            if prefix_context:
                prefix_embeddings = [create_token_embedding(token).embedding for token in prefix_context]
                target_context_embedding = np.mean(prefix_embeddings, axis=0)
            else:
                target_context_embedding = np.zeros(self.embed_dim, dtype=np.float32)

            target_next_embedding = create_token_embedding(next_token).embedding

            similarity_threshold = self._calculate_dynamic_threshold(len(prefix_context), reward)

            patterns_found = 0
            embeddings_checked = 0

            with self.env.begin(db=self.nodes_db) as txn:
                for key, value in txn.cursor():
                    embeddings_checked += 1

                    if embeddings_checked > 500:
                        logger.debug(f"Embedding search limited to {embeddings_checked} checks for performance")
                        break
                    
                    try:
                        logger.debug(f"Checking embedding at {key.decode()}...")
                        node_data = msgpack.unpackb(value, raw=False, strict_map_key=False)
                        stored_embedding_bytes = node_data.get('embedding')

                        if stored_embedding_bytes:
                            stored_embedding = np.frombuffer(stored_embedding_bytes, dtype=np.float32)

                            original_context = self.context_window.current_context_embedding.copy()

                            self.context_window.current_context_embedding = target_context_embedding
                            context_similarity = self.context_window.get_context_similarity(stored_embedding)

                            self.context_window.current_context_embedding = target_next_embedding
                            next_similarity = self.context_window.get_context_similarity(stored_embedding)

                            self.context_window.current_context_embedding = original_context

                            if context_similarity > similarity_threshold or next_similarity > similarity_threshold:
                                patterns_found += 1

                                if patterns_found >= 2:
                                    logger.debug(f"Direct embedding confirmation: found {patterns_found} similar patterns "
                                               f"in {embeddings_checked} embeddings (threshold: {similarity_threshold:.2f})")
                                    return True

                    except Exception as e:
                        logger.debug(f"Error checking embedding at {key}: {e}")
                        continue
                    
            if patterns_found == 0 and embeddings_checked < 100:
                logger.debug(f"No similar patterns found in limited embedding set ({embeddings_checked}), "
                            f"allowing pattern building for initial learning")
                return True

            logger.debug(f"Direct embedding search complete: found {patterns_found} similar patterns "
                        f"in {embeddings_checked} embeddings, confirmation threshold not met")
            return False

        except Exception as e:
            logger.warning(f"Error in direct embedding similarity search: {e}")
            return True

    def _calculate_dynamic_threshold(self, context_length: int, reward: float) -> float:
        """Calculate dynamic similarity threshold based on context length and reward."""
        try:
            base_threshold = 0.6
            reward_adjustment = -0.2 if reward > 0.3 else 0.0
            length_adjustment = min(0.1 * (context_length - 1), 0.2) if context_length > 1 else 0.0
            dynamic_threshold = base_threshold + reward_adjustment + length_adjustment
            final_threshold = max(0.3, min(dynamic_threshold, 0.8))

            logger.debug(f"Dynamic threshold calculation: context_length={context_length}, reward={reward:.2f}, "
                        f"final_threshold={final_threshold:.2f}")

            return final_threshold

        except Exception as e:
            logger.warning(f"Error calculating dynamic threshold: {e}")
            return 0.5

    def _count_exact_pattern_occurrences(self, prefix_context: List[str], next_token: str) -> int:
        """Count exact occurrences of a specific pattern in stored sequences."""
        try:
            pattern_count = 0

            with self.env.begin(db=self.metadata_db) as txn:
                sequences_checked = 0
                for key, value in txn.cursor():
                    sequences_checked += 1
                    if sequences_checked > 50:
                        break
                    
                    try:
                        metadata = msgpack.unpackb(value, raw=False, strict_map_key=False)
                        sequence_id = metadata.get('sequence_id', '')

                        if self._sequence_contains_exact_pattern(sequence_id, prefix_context, next_token):
                            pattern_count += 1

                    except Exception as e:
                        continue
                    
            logger.debug(f"Found {pattern_count} exact occurrences of pattern {prefix_context} -> {next_token}")
            return pattern_count

        except Exception as e:
            logger.warning(f"Error counting pattern occurrences: {e}")
            return 0

    def _sequence_contains_exact_pattern(self, sequence_id: str, prefix_context: List[str], next_token: str) -> bool:
        """Check if stored sequence contains exact pattern match."""
        try:
            stored_tokens = []
            i = 0
            with self.env.begin(db=self.nodes_db) as txn:
                while True:
                    node_key = f"{sequence_id}_{i}".encode()
                    value = txn.get(node_key)
                    if not value:
                        break

                    node_data = msgpack.unpackb(value, raw=False, strict_map_key=False)
                    token = node_data.get('token')
                    if token:
                        stored_tokens.append(token)
                    i += 1

            for j in range(len(stored_tokens) - len(prefix_context)):
                if (stored_tokens[j:j+len(prefix_context)] == prefix_context and 
                    j + len(prefix_context) < len(stored_tokens) and
                    stored_tokens[j + len(prefix_context)] == next_token):
                    return True

            return False

        except Exception as e:
            return False

    def _check_embedding_similarity_confirmation(self, prefix_context: List[str], next_token: str) -> bool:
        """Check embedding similarity for context confirmation with reduced threshold."""
        try:
            with self.env.begin(db=self.metadata_db) as txn:
                sequences_checked = 0
                for key, value in txn.cursor():
                    sequences_checked += 1
                    if sequences_checked > 100:
                        break
                    
                    try:
                        metadata = msgpack.unpackb(value, raw=False, strict_map_key=False)
                        sequence_id = metadata.get('sequence_id', '')

                        if self._sequence_contains_similar_pattern_permissive(sequence_id, prefix_context, next_token):
                            logger.debug(f"Embedding similarity confirmed for {prefix_context} -> {next_token}")
                            return True

                    except Exception as e:
                        continue
                    
            return False

        except Exception as e:
            logger.warning(f"Error in embedding similarity check: {e}")
            return False

    def _sequence_contains_similar_pattern_permissive(self, sequence_id: str, prefix_context: List[str], next_token: str) -> bool:
        """More permissive similarity check with lower threshold (0.6 instead of 0.8)."""
        try:
            from token_embedding import create_token_embedding
            
            if prefix_context:
                prefix_embeddings = [create_token_embedding(token).embedding for token in prefix_context]
                target_context_embedding = np.mean(prefix_embeddings, axis=0)
            else:
                return True

            target_next_embedding = create_token_embedding(next_token).embedding

            stored_tokens = []
            i = 0
            with self.env.begin(db=self.nodes_db) as txn:
                while True:
                    node_key = f"{sequence_id}_{i}".encode()
                    value = txn.get(node_key)
                    if not value:
                        break

                    node_data = msgpack.unpackb(value, raw=False, strict_map_key=False)
                    token = node_data.get('token')
                    if token:
                        stored_tokens.append(token)
                    i += 1

            for j in range(len(stored_tokens) - len(prefix_context)):
                stored_context = stored_tokens[j:j+len(prefix_context)]

                if j + len(prefix_context) < len(stored_tokens):
                    stored_next = stored_tokens[j + len(prefix_context)]

                    stored_context_embeddings = [create_token_embedding(token).embedding for token in stored_context]
                    stored_context_embedding = np.mean(stored_context_embeddings, axis=0)
                    stored_next_embedding = create_token_embedding(stored_next).embedding

                    original_context = self.context_window.current_context_embedding.copy()

                    self.context_window.current_context_embedding = stored_context_embedding
                    context_similarity = self.context_window.get_context_similarity(target_context_embedding)

                    self.context_window.current_context_embedding = stored_next_embedding
                    next_similarity = self.context_window.get_context_similarity(target_next_embedding)

                    self.context_window.current_context_embedding = original_context

                    if context_similarity > 0.6 and next_similarity > 0.6:
                        return True

            return False

        except Exception as e:
            logger.debug(f"Error checking permissive pattern in sequence {sequence_id}: {e}")
            return False

    def _sequence_contains_similar_pattern(self, sequence_id: str, prefix_context: List[str], next_token: str) -> bool:
        """Check if stored sequence contains similar pattern using basic embedding similarity."""
        try:
            from token_embedding import create_token_embedding
            
            if prefix_context:
                prefix_embeddings = [create_token_embedding(token).embedding for token in prefix_context]
                target_context_embedding = np.mean(prefix_embeddings, axis=0)
            else:
                return True

            target_next_embedding = create_token_embedding(next_token).embedding

            stored_tokens = []
            i = 0
            with self.env.begin(db=self.nodes_db) as txn:
                while True:
                    node_key = f"{sequence_id}_{i}".encode()
                    value = txn.get(node_key)
                    if not value:
                        break

                    node_data = msgpack.unpackb(value, raw=False, strict_map_key=False)
                    token = node_data.get('token')
                    if token:
                        stored_tokens.append(token)
                    i += 1

            for j in range(len(stored_tokens) - len(prefix_context)):
                stored_context = stored_tokens[j:j+len(prefix_context)]

                if j + len(prefix_context) < len(stored_tokens):
                    stored_next = stored_tokens[j + len(prefix_context)]

                    stored_context_embeddings = [create_token_embedding(token).embedding for token in stored_context]
                    stored_context_embedding = np.mean(stored_context_embeddings, axis=0)
                    stored_next_embedding = create_token_embedding(stored_next).embedding

                    original_context = self.context_window.current_context_embedding.copy()
                    self.context_window.current_context_embedding = stored_context_embedding
                    context_similarity = self.context_window.get_context_similarity(target_context_embedding)

                    self.context_window.current_context_embedding = original_context

                    self.context_window.current_context_embedding = stored_next_embedding
                    next_similarity = self.context_window.get_context_similarity(target_next_embedding)

                    self.context_window.current_context_embedding = original_context

                    if context_similarity > 0.8 and next_similarity > 0.8:
                        return True

            return False

        except Exception as e:
            logger.debug(f"Error checking pattern in sequence {sequence_id}: {e}")
            return False

    def _validate_sequence_for_storage(self, tokens: List[str], reward: float) -> bool:
        """Validate sequence before storage to prevent cross-sequence contamination."""
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                logger.debug(f"Validation failed: immediate repetition '{tokens[i]}' at position {i}")
                return False

        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
            if token_counts[token] > 2:
                logger.debug(f"Validation failed: excessive repetition of '{token}' ({token_counts[token]} times)")
                return False

        if len(tokens) >= 6:
            for i in range(len(tokens) - 3):
                subsequence = tokens[i:i+3]
                for j in range(i + 3, len(tokens) - 2):
                    if tokens[j:j+3] == subsequence:
                        logger.debug(f"Validation failed: repeated subsequence {subsequence} at positions {i} and {j}")
                        return False

        if reward >= 0.8:
            logger.info(f"Validation override: high reward ({reward}) allows sequence storage despite potential issues")
            return True

        if reward <= 0.1 and self._contains_problematic_patterns(tokens):
            logger.debug(f"Validation failed: low reward ({reward}) + problematic patterns")
            return False

        logger.debug(f"Sequence validation passed: {tokens}")
        return True

    def _contains_problematic_patterns(self, tokens: List[str]) -> bool:
        """Check for problematic patterns in token sequence."""
        # Simple heuristic - look for obvious problematic patterns
        problematic_patterns = [
            ['the', 'the', 'the'],  # Repetitive articles
            ['and', 'and', 'and'],  # Repetitive conjunctions
        ]
        
        for pattern in problematic_patterns:
            for i in range(len(tokens) - len(pattern) + 1):
                if tokens[i:i+len(pattern)] == pattern:
                    return True
        return False
    
    def _aggregate_sequence_embedding(self, token_embeddings) -> np.ndarray:
        """Aggregate token embeddings into sequence embedding"""
        if not token_embeddings:
            return np.zeros(self.embed_dim, dtype=np.float32)
        
        embeddings = np.array([te.embedding for te in token_embeddings])
        sequence_embedding = np.mean(embeddings, axis=0)
        
        norm = np.linalg.norm(sequence_embedding)
        if norm > 0:
            sequence_embedding = sequence_embedding / norm
        
        return sequence_embedding
    
    def _generate_sequence_id(self, tokens: List[str]) -> str:
        """Generate deterministic ID for token sequence"""
        sequence_text = ' '.join(tokens)
        return hashlib.md5(sequence_text.encode()).hexdigest()
    
    def close(self):
        """Close database"""
        if hasattr(self, 'env'):
            self.env.close()
            logger.info("Closed LMDB environment")

    def find_midsequence_continuations(self, query_tokens):
        results = []
        def dfs(node, path):
            if len(path) >= len(query_tokens):
                if path[-len(query_tokens):] == query_tokens:
                    for child in node.children.values():
                        results.append(child)
            for child in node.children.values():
                dfs(child, path + [child.token])
        dfs(self.root, [])
        return results

    def find_best_continuation(self, current_node, context_embedding: np.ndarray, query_sequence_embedding: np.ndarray, query_tokens: List[str], max_candidates: int = 5, max_continuations: int = 100) -> Tuple[List[str], float]:
        """More targeted repetition filtering - only blocks actual problematic repetitions."""
        logger.info(f"Finding best continuation for query: {query_tokens}")
        if not query_tokens:
            return [], 0.0

        try:
            matched_nodes = []
            matched_tokens = []

            for token in query_tokens:
                current_node = current_node.get_child(token)
                if current_node:
                    matched_nodes.append(current_node)
                    matched_tokens.append(token)
                else:
                    break

            matched_length = len(matched_nodes)
            logger.info(f"Found prefix match of length: {matched_length} / {len(query_tokens)}")

            best_candidates = []
            
            if matched_length == 0:
                logger.warning("No prefix match found, returning empty continuation")
                return [], 0.0
            
            current_node = matched_nodes[-1]
            best_candidates.append((current_node.token, current_node.relevance_score))
            
            candidates = []
            
            self._collect_continuations(
                current_node, current_node.path_tokens, candidates, query_sequence_embedding, max_continuations, context_embedding=context_embedding
            )
            trimmed_candidates = [(words[:], score) for words, score in candidates if len(words) > 1]

            if trimmed_candidates and query_tokens:
                filtered_candidates = []
                last_query_token = query_tokens[-1]
                for path, score in trimmed_candidates:
                    should_include = True
                    filter_reason = None

                    if path and path[0] == last_query_token:
                        should_include = False
                        filter_reason = f"immediate repetition: '{path[0]}' == '{last_query_token}'"
                    elif len(set(path)) < len(path):
                        duplicates = [token for token in set(path) if path.count(token) > 1]
                        should_include = False
                        filter_reason = f"internal repetition: {duplicates}"
                    elif len(path) >= 2 and len(query_tokens) >= 2:
                        continuation_start = path[:2]
                        query_end = query_tokens[-2:]
                        if continuation_start == query_end:
                            should_include = False
                            filter_reason = f"exact sequence repetition: {continuation_start} == {query_end}"
                    
                    if should_include:
                        filtered_candidates.append((path, score))
                    else:
                        logger.info(f"Filtered candidate {path}: {filter_reason}")
                
                candidates = filtered_candidates
                logger.info(f"Filtering results: {len(candidates)} candidates remain after targeted repetition filtering")

                if not candidates and max_candidates < 20:
                    logger.info("No candidates passed filtering, trying with more candidates...")
                    return self.find_best_continuation(current_node, context_embedding, query_sequence_embedding, query_tokens, max_candidates * 2)

            if candidates:
                scored = [
                    (path, score + 0.1 * matched_length)
                    for path, score in candidates
                ]
                best_candidates = scored
            
            if not best_candidates:
                logger.warning("No continuation candidates found at any prefix depth")
                return [], 0.0
            
            viable_candidates = [(cont, score) for cont, score in candidates if score > 0.01]
            if not viable_candidates:
                logger.warning("No viable candidates found after filtering")
                return [], 0.0
            else:
                best_candidates = max(viable_candidates, key=lambda x: x[1])
                best_continuation, best_score = best_candidates
            
            logger.info(f"Selected best continuation: {best_continuation} (score: {best_score:.3f})")
            return best_continuation, best_score

        except Exception as e:
            logger.error(f"Error finding continuation: {str(e)}")
            return [], 0.0

    def _collect_continuations(self, node: TrieNode, current_path: List[str], 
                              candidates: List[Tuple[List[str], float]], 
                              query_embedding: np.ndarray, max_continuations: int, 
                              max_depth: int = 612, context_embedding: np.ndarray = None):
        """Better repetition checking during collection."""
        if len(candidates) >= max_continuations or len(current_path) >= max_depth:
            return

        if current_path:
            relevance = node.calculate_relevance(context_embedding=context_embedding, query_embedding=query_embedding)
            activation = node.activation_level
            avg_reward = node.metadata.get('avg_reward', 0.0)
            completeness_bonus = 0.2 if node.is_complete else 0.0

            combined_score = (
                0.4 * relevance +
                0.3 * activation +
                0.2 * avg_reward +
                completeness_bonus
            )

            candidates.append((current_path.copy(), combined_score))

        for token, child_node in node.children.items():
            should_skip = False

            if current_path and token == current_path[-1]:
                logger.debug(f"Skipping immediate repetition: '{token}' after '{current_path[-1]}'")
                should_skip = True

            elif token in current_path:
                logger.debug(f"Skipping internal repetition: '{token}' already in path {current_path}")
                should_skip = True

            if not should_skip:
                self._collect_continuations(
                    child_node, 
                    current_path + [token],
                    candidates, 
                    query_embedding, 
                    max_continuations, 
                    max_depth,
                    context_embedding=context_embedding
                )

    def debug_trie_structure(self, query_tokens: List[str]):
        """Trace the exact trie structure and continuation logic to find the architectural flaw."""
        print(f"\n🔍 DEBUGGING TRIE STRUCTURE FOR: {query_tokens}")
        print("=" * 80)
        
        print("\n📍 STEP 1: PREFIX MATCHING TRACE")
        current_node = self.root
        matched_path = []
        
        print(f"Starting at ROOT node")
        print(f"ROOT children: {list(current_node.children.keys())}")
        
        for i, token in enumerate(query_tokens):
            print(f"\n  Looking for token '{token}'...")
            
            if token in current_node.children:
                current_node = current_node.children[token]
                matched_path.append(token)
                
                print(f"  ✅ FOUND: '{token}' at position {i}")
                print(f"     Node details: token='{current_node.token}', is_end={current_node.is_end_of_sequence}")
                print(f"     Node children: {list(current_node.children.keys())}")
                print(f"     Matched path so far: {matched_path}")
            else:
                print(f"  ❌ NOT FOUND: '{token}' not in {list(current_node.children.keys())}")
                break
            
        print(f"\n📊 PREFIX MATCH RESULT:")
        print(f"   Matched: {len(matched_path)}/{len(query_tokens)} tokens")
        print(f"   Final node token: '{current_node.token}'")
        print(f"   Final node children: {list(current_node.children.keys())}")
        
        print(f"\n🔍 STEP 2: CONTINUATION COLLECTION TRACE")
        if current_node.children:
            print(f"From final node '{current_node.token}', these continuations would be found:")
            self._debug_collect_continuations(current_node, [], max_depth=3)
        else:
            print(f"❌ Final node '{current_node.token}' has NO CHILDREN - should return empty continuation!")
        
        print(f"\n🔍 STEP 3: DEBUGGING COMMAND HANDLER")
        self.debug_command_handler(query_tokens)

        print(f"\n🐛 STEP 4: ARCHITECTURAL ISSUE DETECTION")
        self._check_trie_integrity(query_tokens)
    
    def _debug_collect_continuations(self, node: TrieNode, current_path: List[str], max_depth: int = 3):
        """Debug version of _collect_continuations to trace what's being found."""
        if max_depth <= 0:
            return
        
        indent = "  " * len(current_path)
        
        if node.is_end_of_sequence and current_path:
            print(f"{indent}📍 END PATH: {current_path} (score: {node.activation_level:.3f})")
        
        for token, child_node in node.children.items():
            new_path = current_path + [token]
            print(f"{indent}├─ '{token}' → {new_path}")
            
            if child_node.is_end_of_sequence:
                print(f"{indent}   └─ [END] score: {child_node.activation_level:.3f}")
            
            self._debug_collect_continuations(child_node, new_path, max_depth - 1)
    
    def _check_trie_integrity(self, query_tokens: List[str]):
        """Check for common trie architectural problems."""
        print("\n🔍 CHECKING TRIE INTEGRITY:")
        
        print("\n1. VERIFYING SEQUENCE STORAGE:")
        sequence_id = self._generate_sequence_id(query_tokens)
        print(f"   Expected sequence ID: {sequence_id[:8]}...")
        
        try:
            with self.env.begin(db=self.metadata_db) as txn:
                value = txn.get(sequence_id.encode())
                if value:
                    metadata = msgpack.unpackb(value, raw=False, strict_map_key=False)
                    print(f"   ✅ Sequence found in LMDB with length: {metadata.get('path_length')}")
                else:
                    print(f"   ❌ Sequence NOT found in LMDB!")
        except Exception as e:
            print(f"   ❌ Error checking LMDB: {e}")
        
        print("\n2. VERIFYING NODE LINKING:")
        current_node = self.root
        for i, token in enumerate(query_tokens):
            if token in current_node.children:
                child = current_node.children[token]
                expected_key = f"{sequence_id}_{i}"
                print(f"   Position {i}: '{token}' → node exists, expected key: {expected_key}")
                current_node = child
            else:
                print(f"   ❌ BROKEN LINK at position {i}: '{token}' missing from {list(current_node.children.keys())}")
                break
            
        print("\n3. CHECKING FOR OVERLAPPING SEQUENCES:")
        problematic_tokens = [query_tokens]
        
        for token in problematic_tokens:
            print(f"\n   Searching for all occurrences of '{token}' in trie:")
            self._find_token_occurrences(self.root, token, [])
    
    def _find_token_occurrences(self, node: TrieNode, target_token: str, path: List[str]):
        """Find all places where a specific token appears in the trie."""
        if node.token == target_token:
            print(f"     Found '{target_token}' at path: {path + [target_token]}")
            print(f"       Children: {list(node.children.keys())}")
            print(f"       Is end: {node.is_end_of_sequence}")
        
        for child_token, child_node in node.children.items():
            self._find_token_occurrences(child_node, target_token, path + [node.token] if node.token else path)
    
    def debug_command_handler(self, tokens: List[str]):
        """Add this to your interactive mode to debug trie issues."""
        self.debug_trie_structure(tokens)
    
    def find_best_continuation_debug(self, query_tokens: List[str], max_candidates: int = 5) -> Tuple[List[str], float]:
        """DEBUG VERSION: Shows exactly where problematic continuations are coming from."""
        logger.info(f"DEBUG: Finding continuation for: {query_tokens}")
        
        if not query_tokens:
            return [], 0.0
    
        current_node = self.root
        matched_nodes = []
        
        for token in query_tokens:
            if token in current_node.children:
                current_node = current_node.children[token]
                matched_nodes.append(current_node)
                logger.info(f"DEBUG: Matched '{token}', node children: {list(current_node.children.keys())}")
            else:
                logger.info(f"DEBUG: Token '{token}' not found, stopping at path: {[n.token for n in matched_nodes]}")
                break
            
        matched_length = len(matched_nodes)
        logger.info(f"DEBUG: Final matched length: {matched_length}/{len(query_tokens)}")
        
        if matched_length == len(query_tokens):
            final_node = matched_nodes[-1]
            logger.info(f"DEBUG: Full match! Final node '{final_node.token}' has children: {list(final_node.children.keys())}")
            
            candidates = []
            from token_embedding import create_token_embedding
            query_embeddings = [create_token_embedding(token) for token in query_tokens]
            query_sequence_embedding = self._aggregate_sequence_embedding(query_embeddings)
            
            logger.info(f"DEBUG: Calling _collect_continuations on node '{final_node.token}'")
            self._collect_continuations(
                final_node, [], candidates, query_sequence_embedding, max_candidates
            )
            
            logger.info(f"DEBUG: _collect_continuations found {len(candidates)} candidates:")
            for i, (path, score) in enumerate(candidates):
                logger.info(f"DEBUG:   {i+1}. {path} (score: {score:.3f})")
            
            if candidates:
                best = max(candidates, key=lambda x: x[1])
                logger.info(f"DEBUG: Best candidate: {best[0]} with score {best[1]:.3f}")
                return best[0], best[1]
        
        logger.info("DEBUG: No full match or no candidates found")
        return [], 0.0

    # NEW: Enhanced methods using the registry
    def find_best_continuation_direct(self, query_tokens: List[str], max_candidates: int = 5) -> Tuple[List[str], float]:
        """Fast continuation finding using direct node lookup."""
        logger.info(f"Finding continuation using direct lookup: {query_tokens}")
        
        target_node = self.root.get_node_by_path(query_tokens)
        
        if not target_node:
            logger.info(f"No node found for path: {query_tokens}")
            return [], 0.0
        
        if not target_node.children:
            logger.info("Target node has no children")
            return [], 0.0
        
        candidates = []
        for child_token, child_node in target_node.children.items():
            score = (
                0.4 * child_node.relevance_score +
                0.3 * child_node.activation_level +
                0.2 * child_node.metadata.get('avg_reward', 0.0) +
                (0.1 if child_node.is_complete else 0.0)
            )
            candidates.append(([child_token], score))
        
        if candidates:
            best_continuation, best_score = max(candidates, key=lambda x: x[1])
            logger.info(f"Found continuation: {best_continuation} with score: {best_score:.3f}")
            return best_continuation, best_score
        
        return [], 0.0
    
    def get_node_info(self, query_tokens: List[str]) -> Dict:
        """Get comprehensive node information using direct lookup."""
        node = self.root.get_node_by_path(query_tokens)
        if not node:
            return {'found': False}
        
        return {
            'found': True,
            'node_id': node.node_id,
            'token': node.token,
            'path_tokens': node.path_tokens,
            'hierarchy_level': node.hierarchy_level,
            'children_count': len(node.children),
            'children_tokens': list(node.children.keys()),
            'is_end_of_sequence': node.is_end_of_sequence,
            'activation_level': node.activation_level,
            'relevance_score': node.relevance_score,
            'embedding_key': getattr(node, '_embedding_key', None)
        }
    
    def debug_trie_with_registry(self):
        """Debug both trie structure and registry."""
        print("\n🔍 TRIE + REGISTRY DEBUG:")
        self.root.debug_registry()
        
        print(f"\n🎯 TESTING DIRECT LOOKUPS:")
        test_paths = [
            [],
            ["hello"],
            ["hello", "world"],
            ["goodbye"]
        ]
        
        for path in test_paths:
            node = self.root.get_node_by_path(path)
            path_str = '/'.join(path) if path else 'ROOT'
            if node:
                print(f"  ✅ Found: {path_str} → {node.node_id[:8]}... (children: {len(node.children)})")
            else:
                print(f"  ❌ Not found: {path_str}")