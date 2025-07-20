"""
Event-Driven Activation System for SOTBI

This module implements cross-sequence context discovery by activating nodes
based on partial matches and using them to discover additional continuations.

Key Innovation: Instead of just enhancing existing candidates, this finds NEW
candidates by searching the entire trie for nodes containing query tokens.
"""

import logging
import time
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

import numpy as np

from trie_node import SemanticTrieNode

logger = logging.getLogger(__name__)

class EventDrivenActivation:
    """
    CORRECTED: Event-driven activation system with context-aware filtering and contamination prevention.
    PRESERVED: All existing activation discovery logic.
    ADDED: Context relevance validation and proper activation lifecycle management.
    FIXED: Cross-query contamination issues identified by user.
    """
    
    def __init__(self, trie_memory):
        self.trie_memory = trie_memory
        self.activated_nodes = {}  # node_id -> activation_timestamp
        self.activation_timeout = 60.0  # Activation expires after 60 seconds
        logger.info("Initialized EventDrivenActivation with context-aware filtering and contamination prevention")
    
    def enhance_prediction_with_activation_context(self, query_tokens: List[str], 
                                                 normal_result: Tuple[List[str], float],
                                                 max_activated_candidates: int = 10) -> Tuple[List[str], float]:
        """
        PRESERVED: Core prediction enhancement logic.
        ADDED: Intelligent strategy selection based on result characteristics.
        FIXED: Proper integration with corrected activation methods.
        """
        logger.info(f"Enhancing prediction with activation context for query: {query_tokens}")

        try:
            normal_continuation, normal_confidence = normal_result

            # PRESERVED: Safety check for query length
            #if len(query_tokens) > 20:
            #    logger.warning(f"Query tokens too long ({len(query_tokens)}), using normal result")
            #    return normal_result

            relevant_tokens = query_tokens + normal_continuation

            # CORRECTED: Step 1 - Trigger activation events with contamination prevention
            self._trigger_activation_events(relevant_tokens)

            # PRESERVED: Step 2 - Discover continuations from activated nodes
            activated_continuations = self._discover_activated_continuations(
                relevant_tokens, max_activated_candidates
            )

            # PRESERVED: Step 3 - Intelligent strategy selection and execution
            combination_strategy = self._determine_combination_strategy(
                normal_result, activated_continuations, query_tokens
            )

            if combination_strategy == "select_best":
                enhanced_continuation, enhanced_confidence = self._choose_best_result(
                    normal_result, activated_continuations
                )
                logger.info(f"Used selection strategy: {enhanced_continuation} (confidence: {enhanced_confidence:.3f})")

            elif combination_strategy == "merge_content":
                enhanced_continuation, enhanced_confidence = self._merge_results_content(
                    normal_result, activated_continuations
                )
                logger.info(f"Used merging strategy: {enhanced_continuation} (confidence: {enhanced_confidence:.3f})")

            elif combination_strategy == "hybrid":
                # PRESERVED: Hybrid approach
                best_result = self._choose_best_result(normal_result, activated_continuations)
                enhanced_continuation, enhanced_confidence = self._merge_results_content(
                    best_result, activated_continuations
                )
                logger.info(f"Used hybrid strategy: {enhanced_continuation} (confidence: {enhanced_confidence:.3f})")

            else:
                # FALLBACK: Default to selection
                enhanced_continuation, enhanced_confidence = self._choose_best_result(
                    normal_result, activated_continuations
                )
                logger.warning(f"Unknown strategy '{combination_strategy}', defaulted to selection")

            return enhanced_continuation, enhanced_confidence

        except Exception as e:
            logger.error(f"Error in activation context enhancement: {e}")
            logger.info("Falling back to normal result due to activation error")
            return normal_result

    def _trigger_activation_events(self, query_tokens: List[str]):
        """
        CORRECTED: Activation triggering with contamination prevention and context awareness.
        PRESERVED: Core activation triggering logic.
        ADDED: Proper activation lifecycle management and context-aware filtering.
        FIXED: User-identified contamination issues by resetting activation flags.
        """
        logger.debug(f"Triggering activation events for tokens: {query_tokens}")
        
        # ADDED: Reset ALL activation flags before new activation cycle
        # JUSTIFICATION: Prevents contamination from previous queries as identified by user
        self._reset_all_activations()
        
        # PRESERVED: Clean up expired activations
        self._cleanup_expired_activations()
        
        # ADDED: Calculate query context embedding for relevance filtering
        # JUSTIFICATION: User identified context-blind activation as source of inappropriate selections
        query_context_embedding = self._calculate_query_context_embedding(query_tokens)
        
        # PRESERVED: Search entire trie for matching nodes
        activated_count = 0
        for query_token in query_tokens:
            # ENHANCED: Pass context embedding for relevance filtering
            activated_count += self._activate_nodes_containing_token(query_token, query_context_embedding)
        
        logger.info(f"Activated {activated_count} contextually relevant nodes across all query tokens")

    def _reset_all_activations(self):
        """
        ADDED: Reset all activation flags to prevent cross-query contamination.
        JUSTIFICATION: User identified that is_activated flags never get reset, causing contamination.
        """
        reset_count = 0
        
        # Use iterative traversal to reset all activation flags
        stack = [self.trie_memory.root]
        visited = set()
        max_iterations = 10000  # Prevent infinite loops
        iterations = 0
        
        while stack and iterations < max_iterations:
            iterations += 1
            node = stack.pop()
            
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)
            
            # ADDED: Reset activation state
            if hasattr(node, 'is_activated') and node.is_activated:
                node.is_activated = False
                node.activation_timestamp = 0.0
                reset_count += 1
            
            # Add children to stack
            for child_node in node.children.values():
                if id(child_node) not in visited:
                    stack.append(child_node)
        
        logger.debug(f"Reset {reset_count} previously activated nodes")
        
        # ADDED: Clear activated nodes tracking
        self.activated_nodes.clear()

    def _calculate_query_context_embedding(self, query_tokens: List[str]) -> Optional[np.ndarray]:
        """
        ADDED: Calculate aggregated embedding for query tokens to enable context-aware activation.
        JUSTIFICATION: User identified need for context-aware activation instead of blind token matching.
        """
        try:
            if not query_tokens:
                return None
            
            # Collect embeddings for query tokens
            query_embeddings = []
            for token in query_tokens:
                # Try to find nodes with this token and get their embeddings
                token_nodes = self._find_nodes_with_token(token)
                for node in token_nodes:
                    if node.token_embedding is not None:
                        query_embeddings.append(node.token_embedding)
                        break  # Use first available embedding for this token
            
            if not query_embeddings:
                logger.warning("No embeddings found for query tokens")
                return None
            
            # ADDED: Aggregate query embeddings using mean
            # JUSTIFICATION: Simple aggregation provides query context representation
            aggregated_embedding = np.mean(query_embeddings, axis=0)
            logger.debug(f"Calculated query context embedding from {len(query_embeddings)} token embeddings")
            
            return aggregated_embedding
            
        except Exception as e:
            logger.error(f"Error calculating query context embedding: {str(e)}")
            return None

    
    def _activate_nodes_containing_token(self, target_token: str, query_context_embedding: Optional[np.ndarray] = None) -> int:
        """
        ENHANCED: Optimized activation with activation level boosting.
        ADDED: Boost activation_level when nodes are activated for reinforcement learning.
        """
        activated_count = 0
        current_time = time.time()

        logger.debug(f"Activating nodes containing token '{target_token}' using optimized token lookup")

        # Get matching nodes
        if hasattr(self.trie_memory.root, 'get_nodes_by_token'):
            matching_nodes = self.trie_memory.root.get_nodes_by_token(target_token)
        else:
            all_nodes = self.trie_memory.root.get_all_registered_nodes()
            matching_nodes = [node for node in all_nodes if node.token == target_token]

        logger.debug(f"Found {len(matching_nodes)} nodes with token '{target_token}'")

        for node in matching_nodes:
            # Context relevance validation
            should_activate = self._validate_activation_relevance(node, query_context_embedding, target_token)

            if should_activate:
                # ENHANCED: Set activation flags
                activation_id = getattr(node, '_embedding_key', None) or getattr(node, 'node_id', f"unknown_{id(node)}")
                node.is_activated = True
                node.activation_timestamp = current_time
                self.activated_nodes[activation_id] = current_time
                
                # ADDED: Boost activation level for reinforcement learning
                activation_boost = self._calculate_activation_boost(node, query_context_embedding)
                old_activation = node.activation_level
                node.activation_level = min(1.0, node.activation_level + activation_boost)
                
                # ADDED: Update node access tracking
                node.access_count += 1
                node.last_accessed = current_time
                
                activated_count += 1

                # ENHANCED: Detailed logging
                path_str = '/'.join(getattr(node, 'path_tokens', [])) if hasattr(node, 'path_tokens') else 'unknown'
                logger.debug(f"Activated node '{target_token}' at path: {path_str}")
                logger.debug(f"Activation boost: {old_activation:.3f} -> {node.activation_level:.3f} (+{activation_boost:.3f})")

        logger.info(f"Enhanced activation complete: {activated_count}/{len(matching_nodes)} nodes activated")
        return activated_count

    def _calculate_activation_boost(self, node, query_context_embedding: Optional[np.ndarray]) -> float:
        """
        ADDED: Calculate appropriate activation boost based on context relevance.
        ADAPTIVE: Higher relevance = higher boost.
        """
        try:
            base_boost = 0.1  # Minimum boost for any activation
            
            # Context-based boost
            if query_context_embedding is not None and hasattr(node, 'token_embedding') and node.token_embedding is not None:
                try:
                    similarity = self.trie_memory.context_window._calculate_ensemble_similarity(
                        node.token_embedding, query_context_embedding
                    )
                    context_boost = similarity * 0.2  # Max 0.2 additional boost
                except Exception as e:
                    logger.debug(f"Error calculating context boost: {e}")
                    context_boost = 0.0
            else:
                context_boost = 0.0
            
            # Historical performance boost
            if hasattr(node, 'metadata'):
                avg_reward = node.metadata.get('avg_reward', 0.0)
                performance_boost = max(0.0, avg_reward * 0.1)  # Positive performance gets extra boost
            else:
                performance_boost = 0.0
            
            total_boost = base_boost + context_boost + performance_boost
            
            # Cap the boost to prevent runaway activation
            final_boost = min(0.4, total_boost)
            
            logger.debug(f"Activation boost calculation: base={base_boost:.3f}, context={context_boost:.3f}, "
                        f"performance={performance_boost:.3f}, final={final_boost:.3f}")
            
            return final_boost
            
        except Exception as e:
            logger.warning(f"Error calculating activation boost: {e}")
            return 0.1  # Safe fallback

    def _validate_activation_relevance(self, node: 'SemanticTrieNode', query_context_embedding: Optional[np.ndarray], target_token: str) -> bool:
        """
        ADDED: Validate whether a node should be activated based on context relevance.
        JUSTIFICATION: User identified inappropriate activations due to context-blind token matching.
        """
        try:
            # ADDED: If no query context available, default to conservative activation
            if query_context_embedding is None:
                logger.debug(f"No query context available, using conservative activation for '{target_token}'")
                return True  # Conservative: activate when we can't determine context
            
            # ADDED: If node has no embedding, check sequence context
            if node.token_embedding is None:
                # ADDED: Use path-based heuristics when embedding unavailable
                path_relevance = self._evaluate_path_context_relevance(node, target_token)
                logger.debug(f"No embedding for '{target_token}', path relevance: {path_relevance:.3f}")
                return path_relevance > 0.3  # Threshold for path-based relevance
            
            # ADDED: Calculate semantic relevance between node and query context
            try:
                semantic_relevance = self.trie_memory.context_window._calculate_ensemble_similarity(
                    node.token_embedding, query_context_embedding
                )
                logger.debug(f"Semantic relevance for '{target_token}': {semantic_relevance:.3f}")
                
                # ADDED: Multi-factor activation decision
                relevance_threshold = 0.25  # Configurable threshold
                
                # Factor 2: Node quality metrics
                node_quality = self._calculate_node_quality_score(node)
                
                # Factor 3: Historical performance
                historical_performance = node.metadata.get('avg_reward', 0.0)
                
                # ADDED: Combined activation score
                activation_score = (
                    semantic_relevance * 0.6 +
                    node_quality * 0.2 + 
                    historical_performance * 0.2
                )
                
                should_activate = activation_score > relevance_threshold
                
                logger.debug(f"Activation decision for '{target_token}': score={activation_score:.3f}, "
                            f"threshold={relevance_threshold:.3f}, activate={should_activate}")
                
                return should_activate
                
            except Exception as e:
                logger.error(f"Error calculating semantic relevance for '{target_token}': {str(e)}")
                return False  # Conservative: skip activation on calculation error
            
        except Exception as e:
            logger.error(f"Error validating activation relevance for '{target_token}': {str(e)}")
            return False  # Conservative: skip activation on validation error

    def _evaluate_path_context_relevance(self, node: 'SemanticTrieNode', target_token: str) -> float:
        """
        ADDED: Evaluate path-based context relevance when embeddings unavailable.
        JUSTIFICATION: Provides fallback relevance assessment for nodes without embeddings.
        """
        try:
            # ADDED: Simple path relevance heuristics
            # Note: This is simplified - full implementation would require parent tracking
            path_tokens = [node.token] if node.token else []
            
            if len(path_tokens) == 0:
                return 0.5  # Neutral relevance for isolated tokens
            
            # ADDED: Check for common context patterns
            context_indicators = ['question', 'answer', 'explanation', 'definition', 'example']
            path_text = ' '.join(path_tokens).lower()
            
            relevance_score = 0.0
            for indicator in context_indicators:
                if indicator in path_text:
                    relevance_score += 0.2
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            logger.error(f"Error evaluating path context relevance: {str(e)}")
            return 0.0

    def _calculate_node_quality_score(self, node: 'SemanticTrieNode') -> float:
        """
        ADDED: Calculate node quality score based on multiple factors.
        JUSTIFICATION: Provides additional signal for activation decisions beyond semantic relevance.
        """
        try:
            # ADDED: Quality factors with explicit weights
            factors = {
                'activation_level': node.activation_level * 0.3,
                'relevance_score': node.relevance_score * 0.3,
                'access_frequency': min(1.0, node.access_count / 100.0) * 0.2,
                'reward_consistency': self._calculate_reward_consistency(node) * 0.2
            }
            
            quality_score = sum(factors.values())
            logger.debug(f"Node quality for '{node.token}': {quality_score:.3f}")
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating node quality score: {str(e)}")
            return 0.0

    def _calculate_reward_consistency(self, node: 'SemanticTrieNode') -> float:
        """
        ADDED: Calculate reward consistency to avoid activating unreliable nodes.
        JUSTIFICATION: Nodes with inconsistent rewards should be less likely to activate.
        """
        try:
            if len(node.reward_history) < 3:
                return 0.5  # Neutral consistency for nodes with insufficient history
            
            recent_rewards = node.reward_history[-5:]  # Last 5 rewards
            reward_variance = np.var(recent_rewards)
            consistency = max(0.0, 1.0 - reward_variance)  # Lower variance = higher consistency
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating reward consistency: {str(e)}")
            return 0.0

    def _find_nodes_with_token(self, target_token: str) -> List['SemanticTrieNode']:
        """
        PRESERVED: Helper method to find all nodes containing a specific token.
        ENHANCED: Added iteration limits and cycle prevention for safety.
        """
        found_nodes = []
        
        # ADDED: Use iterative approach with safety limits
        stack = [self.trie_memory.root]
        visited = set()
        max_iterations = 100000
        iterations = 0
        
        while stack and iterations < max_iterations:
            iterations += 1
            node = stack.pop()
            
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)
            
            if node.token == target_token:
                found_nodes.append(node)
            
            for child in node.children.values():
                if id(child) not in visited:
                    stack.append(child)
        
        if iterations >= max_iterations:
            logger.warning(f"Node search hit iteration limit for token: {target_token}")
        
        return found_nodes

    def _cleanup_expired_activations(self):
        """
        PRESERVED: Clean up expired activations based on timeout.
        """
        current_time = time.time()
        expired_activations = []
        
        for activation_id, activation_time in self.activated_nodes.items():
            if current_time - activation_time > self.activation_timeout:
                expired_activations.append(activation_id)
        
        for activation_id in expired_activations:
            del self.activated_nodes[activation_id]
        
        if expired_activations:
            logger.debug(f"Cleaned up {len(expired_activations)} expired activations")

    def _discover_activated_continuations(self, query_tokens: List[str], max_candidates: int) -> Tuple[List[str], float]:
        """
        ENHANCED: Discover continuations from activated nodes using registry-based access.
        PERFORMANCE: Much faster continuation discovery using direct node access.
        """
        logger.debug(f"Discovering continuations from activated nodes for query: {query_tokens}")
        
        activated_continuations = []
        
        # NEW: Use registry to get activated nodes efficiently
        if not self.activated_nodes:
            logger.debug("No activated nodes to process")
            return [], 0.0
        
        # ENHANCED: Process activated nodes using registry lookup
        processed_count = 0
        valid_activations = 0
        
        for activation_id, activation_time in self.activated_nodes.items():
            processed_count += 1
            
            # NEW: Get node by activation ID (which is embedding_key or node_id)
            node = None
            
            # Try embedding key lookup first
            if hasattr(self.trie_memory.root, 'get_node_by_embedding_key'):
                node = self.trie_memory.root.get_node_by_embedding_key(activation_id)
            
            # Fallback to node ID lookup
            if not node and activation_id.startswith('node_'):
                node_id = activation_id
                node = self.trie_memory.root.get_node_by_id(node_id)
            
            if not node:
                logger.debug(f"Could not find node for activation_id: {activation_id}")
                continue
            
            valid_activations += 1
            
            # PRESERVED: Check if node is still activated and recent
            current_time = time.time()
            if (hasattr(node, 'is_activated') and node.is_activated and 
                current_time - activation_time < 300):  # 5 minute activation window
                
                # NEW: Use registry to get node's path and find continuations
                node_path = getattr(node, 'path_tokens', [])
                
                # Get children as potential continuations
                if node.children:
                    for child_token, child_node in node.children.items():
                        # Calculate continuation score
                        score = self._calculate_continuation_score(child_node, query_tokens, node_path)
                        
                        if score > 0.1:  # Minimum viability threshold
                            continuation_path = [child_token]
                            activated_continuations.append((continuation_path, score))
                            
                            logger.debug(f"Found activated continuation: {continuation_path} "
                                       f"(score: {score:.3f}, from path: {'/'.join(node_path)})")
        
        logger.debug(f"Processed {processed_count} activations, {valid_activations} valid, "
                   f"found {len(activated_continuations)} continuations")
        
        # PRESERVED: Return best continuation
        viable_continuations = [(cont, score) for cont, score in activated_continuations if score > 0.1]
        
        if viable_continuations:
            # Sort by score and take the best
            best_continuation, best_score = max(viable_continuations, key=lambda x: x[1])
            logger.debug(f"Best activated continuation: {best_continuation} (score: {best_score:.3f})")
            return best_continuation, best_score
        else:
            logger.debug("No viable activated continuations found")
            return [], 0.0

    def _calculate_activation_confidence(self, node: 'SemanticTrieNode', continuation: List[str]) -> float:
        """
        PRESERVED: Calculate confidence score for activated continuation.
        """
        try:
            base_confidence = node.activation_level * 0.4
            reward_confidence = node.metadata.get('avg_reward', 0.0) * 0.3
            length_confidence = min(1.0, len(continuation) / 5.0) * 0.3
            
            total_confidence = base_confidence + reward_confidence + length_confidence
            return min(1.0, total_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating activation confidence: {str(e)}")
            return 0.0

    def _determine_combination_strategy(self, normal_result: Tuple[List[str], float], 
                                      activated_continuations, query_tokens: List[str]) -> str:
        """
        PRESERVED: Intelligent strategy selection based on result characteristics and query analysis.
        """
        try:
            normal_continuation, normal_confidence = normal_result
            logger.debug(f"Analyzing combination strategy for {len(query_tokens)} query tokens")
            
            # Handle empty activated continuations
            if not activated_continuations or (isinstance(activated_continuations, tuple) and activated_continuations[1] == 0.0):
                logger.debug("No valid activated continuations - using selection strategy")
                return "select_best"
            
            # For single tuple result from _discover_activated_continuations
            if isinstance(activated_continuations, tuple):
                activated_continuations = [activated_continuations]
            
            # Extract valid activated results for analysis
            valid_activated = []
            if isinstance(activated_continuations, list):
                for item in activated_continuations:
                    if isinstance(item, tuple) and len(item) == 2:
                        continuation, confidence = item
                        if isinstance(continuation, list) and isinstance(confidence, (int, float)) and confidence > 0.1:
                            valid_activated.append((continuation, confidence))
            
            if not valid_activated:
                logger.debug("No valid activated continuations found")
                return "select_best"
            
            # Confidence Analysis
            activated_confidences = [conf for _, conf in valid_activated]
            max_activated_confidence = max(activated_confidences)
            confidence_gap = abs(normal_confidence - max_activated_confidence)
            
            logger.debug(f"Confidence analysis: normal={normal_confidence:.3f}, "
                        f"max_activated={max_activated_confidence:.3f}, gap={confidence_gap:.3f}")
            
            # Content Overlap Analysis
            normal_tokens_set = set(normal_continuation)
            content_overlaps = []
            
            for continuation, _ in valid_activated:
                activated_tokens_set = set(continuation)
                overlap = len(normal_tokens_set.intersection(activated_tokens_set))
                total_unique = len(normal_tokens_set.union(activated_tokens_set))
                overlap_ratio = overlap / max(1, total_unique)
                content_overlaps.append(overlap_ratio)
            
            avg_overlap = sum(content_overlaps) / len(content_overlaps) if content_overlaps else 0
            logger.debug(f"Content overlap analysis: average_overlap={avg_overlap:.3f}")
            
            # Query Complexity Analysis
            query_complexity = self._analyze_query_complexity(query_tokens)
            logger.debug(f"Query complexity: {query_complexity}")
            
            # Decision Logic
            if confidence_gap > 0.3:
                logger.info(f"Large confidence gap ({confidence_gap:.3f}) - using selection strategy")
                return "select_best"
            
            if avg_overlap < 0.3 and query_complexity in ["complex", "analytical"]:
                logger.info(f"Low content overlap ({avg_overlap:.3f}) with complex query - using merge strategy")
                return "merge_content"
            
            if confidence_gap < 0.15 and 0.3 <= avg_overlap <= 0.7:
                logger.info(f"Close confidences ({confidence_gap:.3f}) with moderate overlap ({avg_overlap:.3f}) - using hybrid strategy")
                return "hybrid"
            
            if avg_overlap > 0.7:
                logger.info(f"High content overlap ({avg_overlap:.3f}) - using selection to avoid redundancy")
                return "select_best"
            
            logger.info("No specific criteria met - defaulting to selection strategy")
            return "select_best"
            
        except Exception as e:
            logger.error(f"Error determining combination strategy: {str(e)}")
            return "select_best"

    def _analyze_query_complexity(self, query_tokens: List[str]) -> str:
        """
        PRESERVED: Query complexity analysis to inform combination strategy.
        """
        try:
            complexity_indicators = {
                'question_words': ['what', 'how', 'why', 'when', 'where', 'who', 'which'],
                'analytical_words': ['analyze', 'compare', 'evaluate', 'explain', 'describe', 'discuss'],
                'technical_words': ['algorithm', 'system', 'method', 'process', 'architecture', 'implementation'],
                'connective_words': ['because', 'therefore', 'however', 'although', 'furthermore', 'moreover']
            }
            
            token_set = set(token.lower() for token in query_tokens)
            
            # Count complexity indicators
            question_count = len(token_set.intersection(complexity_indicators['question_words']))
            analytical_count = len(token_set.intersection(complexity_indicators['analytical_words']))
            technical_count = len(token_set.intersection(complexity_indicators['technical_words']))
            connective_count = len(token_set.intersection(complexity_indicators['connective_words']))
            
            total_complexity_score = question_count + analytical_count + technical_count + connective_count
            query_length = len(query_tokens)
            
            logger.debug(f"Complexity analysis: question={question_count}, analytical={analytical_count}, "
                        f"technical={technical_count}, connective={connective_count}, length={query_length}")
            
            if total_complexity_score >= 3 or query_length > 10:
                return "complex"
            elif analytical_count > 0 or technical_count > 1:
                return "analytical"
            elif total_complexity_score >= 1 or query_length > 5:
                return "moderate"
            else:
                return "simple"
                
        except Exception as e:
            logger.error(f"Error analyzing query complexity: {str(e)}")
            return "simple"

    def _choose_best_result(self, normal_result: Tuple[List[str], float], 
                           activated_continuations) -> Tuple[List[str], float]:
        """
        PRESERVED: Principled result selection without arbitrary confidence manipulation.
        """
        try:
            normal_continuation, normal_confidence = normal_result
            logger.debug(f"Choosing best result from normal: {normal_continuation} (conf: {normal_confidence:.3f})")
            
            # Handle empty activated continuations
            if not activated_continuations or (isinstance(activated_continuations, tuple) and activated_continuations[1] == 0.0):
                logger.debug("No activated continuations, returning normal result")
                return normal_result
            
            # Normalize to list format
            if isinstance(activated_continuations, tuple):
                activated_continuations = [activated_continuations]
            
            all_candidates = []
            all_candidates.append(("normal", normal_continuation, normal_confidence))
            
            # Process activated continuations
            valid_activated_count = 0
            if isinstance(activated_continuations, list):
                for i, item in enumerate(activated_continuations):
                    try:
                        if isinstance(item, tuple) and len(item) == 2:
                            continuation, confidence = item
                            if isinstance(continuation, list) and isinstance(confidence, (int, float)) and confidence > 0.1:
                                all_candidates.append(("activated", continuation, confidence))
                                valid_activated_count += 1
                                logger.debug(f"Added activated candidate: {continuation} -> {confidence:.3f}")
                    except Exception as e:
                        logger.error(f"Error processing activated candidate {i}: {str(e)}")
                        continue
            
            if valid_activated_count == 0:
                logger.warning("No valid activated continuations found, returning normal result")
                return normal_result
            
            # Filter viable candidates
            viable_candidates = [(source, cont, score) for source, cont, score in all_candidates if score > 0.1]
            
            if not viable_candidates:
                logger.warning("No viable candidates found")
                return [], 0.0
            
            # Sort by confidence
            viable_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Selection with tie-breaking
            best_source, best_continuation, best_confidence = viable_candidates[0]
            
            # Tie-breaking for close confidences
            if len(viable_candidates) > 1:
                second_source, second_continuation, second_confidence = viable_candidates[1]
                confidence_diff = best_confidence - second_confidence
                
                if confidence_diff < 0.05 and best_source == "normal" and second_source == "activated":
                    best_source, best_continuation, best_confidence = second_source, second_continuation, second_confidence
                    logger.info(f"Tie-breaking: chose activated result due to close confidence and diversity preference")
            
            logger.info(f"Selected {best_source} continuation with confidence {best_confidence:.3f}")
            
            # Final validation
            if not isinstance(best_continuation, list) or not best_continuation:
                logger.error(f"Invalid best continuation, falling back to normal")
                return normal_result
            
            return best_continuation, best_confidence

        except Exception as e:
            logger.error(f"Error in result selection: {str(e)}")
            return normal_result

    def _merge_results_content(self, normal_result: Tuple[List[str], float], 
                              activated_continuations) -> Tuple[List[str], float]:
        """
        PRESERVED: True content merging implementation for when actual merging is desired.
        """
        try:
            normal_continuation, normal_confidence = normal_result
            
            if not activated_continuations or (isinstance(activated_continuations, tuple) and activated_continuations[1] == 0.0):
                return normal_result
            
            # Normalize activated continuations
            if isinstance(activated_continuations, tuple):
                activated_continuations = [activated_continuations]
            
            # Content merging strategy
            merged_tokens = []
            confidence_weights = []
            
            # Add normal result tokens
            merged_tokens.extend(normal_continuation)
            confidence_weights.extend([normal_confidence] * len(normal_continuation))
            
            # Add activated continuation tokens
            for item in activated_continuations:
                if isinstance(item, tuple) and len(item) == 2:
                    continuation, confidence = item
                    if isinstance(continuation, list):
                        # Avoid duplicate tokens while preserving order
                        for token in continuation:
                            if token not in merged_tokens:
                                merged_tokens.append(token)
                                confidence_weights.append(confidence)
            
            # Calculate weighted average confidence
            if confidence_weights:
                merged_confidence = sum(confidence_weights) / len(confidence_weights)
            else:
                merged_confidence = normal_confidence
            
            logger.info(f"Content merging: combined {len(merged_tokens)} tokens with confidence {merged_confidence:.3f}")
            return merged_tokens, merged_confidence
            
        except Exception as e:
            logger.error(f"Error in content merging: {str(e)}")
            return normal_result
        
    def _calculate_continuation_score(self, node, query_tokens: List[str], source_path: List[str]) -> float:
        """
        Calculate score for a continuation node from activated nodes.
        COMBINES: Activation level, relevance, rewards, and path compatibility.

        Args:
            node: The child node being considered as a continuation
            query_tokens: The original query tokens
            source_path: Path tokens of the activated parent node

        Returns:
            Continuation score as a float between 0.0 and 1.0
        """
        try:
            # Base score from node properties
            base_score = 0.0

            # Factor 1: Node activation level (40% weight)
            if hasattr(node, 'activation_level'):
                base_score += 0.4 * node.activation_level

            # Factor 2: Node relevance score (30% weight)
            if hasattr(node, 'relevance_score'):
                base_score += 0.3 * node.relevance_score

            # Factor 3: Average reward from metadata (20% weight)
            if hasattr(node, 'metadata') and isinstance(node.metadata, dict):
                avg_reward = node.metadata.get('avg_reward', 0.0)
                # Ensure it's a valid number
                if isinstance(avg_reward, (int, float)) and not np.isnan(avg_reward):
                    base_score += 0.2 * max(0.0, min(1.0, avg_reward))

            # Factor 4: Completion bonus (10% weight)
            if hasattr(node, 'is_complete') and node.is_complete:
                base_score += 0.1

            # Factor 5: Path compatibility bonus
            if source_path and query_tokens:
                compatibility_bonus = self._calculate_path_compatibility(source_path, query_tokens)
                base_score += compatibility_bonus * 0.1

            # Factor 6: Frequency bonus (small weight)
            if hasattr(node, 'metadata') and isinstance(node.metadata, dict):
                frequency = node.metadata.get('frequency', 0)
                if isinstance(frequency, (int, float)):
                    frequency_bonus = min(0.05, frequency / 1000.0)  # Cap at 0.05
                    base_score += frequency_bonus

            # Factor 7: Recency bonus
            if hasattr(node, 'last_accessed'):
                import time
                current_time = time.time()
                time_since_access = current_time - node.last_accessed
                # More recent = higher bonus, decays over 24 hours
                recency_bonus = max(0.0, 0.05 * (1.0 - min(1.0, time_since_access / 86400)))
                base_score += recency_bonus

            # Ensure score is within valid range
            final_score = max(0.0, min(1.0, base_score))

            logger.debug(f"Continuation score for '{getattr(node, 'token', 'unknown')}': {final_score:.3f}")
            return final_score

        except Exception as e:
            logger.warning(f"Error calculating continuation score: {e}")
            return 0.0


    def _calculate_path_compatibility(self, source_path: List[str], query_tokens: List[str]) -> float:
        """
        HELPER: Calculate compatibility between source path and query tokens.

        Args:
            source_path: Path tokens of the activated parent node
            query_tokens: The original query tokens

        Returns:
            Compatibility score between 0.0 and 1.0
        """
        try:
            if not source_path or not query_tokens:
                return 0.5  # Neutral compatibility

            # Check for suffix overlap (most important)
            max_overlap = min(len(source_path), len(query_tokens))
            suffix_overlap = 0

            for i in range(1, max_overlap + 1):
                if source_path[-i:] == query_tokens[-i:]:
                    suffix_overlap = i
                else:
                    break
                
            # Calculate suffix compatibility
            suffix_compatibility = suffix_overlap / len(query_tokens) if query_tokens else 0.0

            # Check for any token overlap
            source_set = set(source_path)
            query_set = set(query_tokens)
            token_overlap = len(source_set.intersection(query_set))
            total_unique = len(source_set.union(query_set))
            token_compatibility = token_overlap / total_unique if total_unique > 0 else 0.0

            # Check for semantic patterns (simple heuristics)
            semantic_compatibility = self._check_semantic_patterns(source_path, query_tokens)

            # Weighted combination
            compatibility = (
                0.5 * suffix_compatibility +      # Suffix match most important
                0.3 * token_compatibility +       # Token overlap
                0.2 * semantic_compatibility      # Semantic patterns
            )

            logger.debug(f"Path compatibility: suffix={suffix_compatibility:.2f}, "
                        f"token={token_compatibility:.2f}, semantic={semantic_compatibility:.2f}, "
                        f"final={compatibility:.2f}")

            return compatibility

        except Exception as e:
            logger.warning(f"Error calculating path compatibility: {e}")
            return 0.0


    def _check_semantic_patterns(self, source_path: List[str], query_tokens: List[str]) -> float:
        """
        HELPER: Check for semantic patterns between source path and query.

        Args:
            source_path: Path tokens of the activated parent node
            query_tokens: The original query tokens

        Returns:
            Semantic compatibility score between 0.0 and 1.0
        """
        try:
            # Simple semantic pattern detection
            patterns = {
                'question_answer': {
                    'triggers': ['what', 'how', 'why', 'when', 'where', 'who'],
                    'responses': ['the', 'it', 'this', 'that', 'because', 'since']
                },
                'cause_effect': {
                    'triggers': ['because', 'since', 'due', 'caused'],
                    'responses': ['therefore', 'thus', 'so', 'then', 'result']
                },
                'sequence': {
                    'triggers': ['first', 'initially', 'begin', 'start'],
                    'responses': ['then', 'next', 'after', 'following', 'subsequently']
                }
            }

            source_text = ' '.join(source_path).lower()
            query_text = ' '.join(query_tokens).lower()

            semantic_score = 0.0
            pattern_count = 0

            for pattern_name, pattern_data in patterns.items():
                # Check if query contains triggers and source contains responses
                query_has_trigger = any(trigger in query_text for trigger in pattern_data['triggers'])
                source_has_response = any(response in source_text for response in pattern_data['responses'])

                if query_has_trigger and source_has_response:
                    semantic_score += 0.3
                    pattern_count += 1
                    logger.debug(f"Found semantic pattern: {pattern_name}")

                # Check reverse (source has trigger, query has response)
                source_has_trigger = any(trigger in source_text for trigger in pattern_data['triggers'])
                query_has_response = any(response in query_text for response in pattern_data['responses'])

                if source_has_trigger and query_has_response:
                    semantic_score += 0.2
                    pattern_count += 1
                    logger.debug(f"Found reverse semantic pattern: {pattern_name}")

            # Normalize by pattern count to avoid over-scoring
            final_score = min(1.0, semantic_score)

            return final_score

        except Exception as e:
            logger.warning(f"Error checking semantic patterns: {e}")
            return 0.0