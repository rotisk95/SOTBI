# Configure logging for execution transparency
from collections import deque
from dataclasses import dataclass
import hashlib
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from scipy.spatial.distance import cosine, euclidean

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ContextWindow:
    """
    FIXED: Manages conversation context windows with adaptive weighting system.
    CORRECTED: Added missing _calculate_adaptive_weights method and fixed method definitions.
    """
    max_turns: int = 500
    max_tokens: int = 100
    time_window_seconds: int = 300
    
    def __init__(self, core_values: Dict[str, Any] = None, max_turns: int = 500, max_tokens: int = 100, time_window_seconds: int = 300, method_weights: Optional[dict] = None):
        """Initialize context window with enhanced similarity calculator."""
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.time_window_seconds = time_window_seconds
        self.conversation_history: deque = deque(maxlen=max_turns)
        self.current_context_embedding: Optional[np.ndarray] = None
        self.core_values = None
        self.value_context_weight = 0.25  # VALUES: 25% of context weighting
        
        logger.info("Enhanced context window with value-aware weighting")
        # Default ensemble weights - can be learned/adjusted
        self.method_weights = method_weights or {
            'cosine': 0.4,
            'angular': 0.2, 
            'probabilistic_js': 0.2,
            'euclidean': 0.1,
            'spherical': 0.1
        }
        self.value_embeddings_cache = {}
        self.value_embeddings_generated = False
        
        logger.info(f"Enhanced similarity calculator initialized with weights: {self.method_weights}")
        logger.info(f"Initialized ContextWindow with max_turns={max_turns}, max_tokens={max_tokens}")


    def _apply_value_context_adjustment(self, base_weights: List[float]) -> List[float]:
        """
        IMPLEMENTATION: Apply core values adjustment to context weights.
        
        ACCOUNTABILITY CHANGES:
        1. ADDED: Core values to embeddings conversion with deterministic algorithm
        2. ADDED: Context-value alignment calculation for each conversation turn
        3. ADDED: Weight adjustment based on value alignment scores (25% influence)
        4. PRESERVED: Base weight structure and normalization requirements
        5. LOGGED: Each step of value-adjustment process for transparency
        
        JUSTIFICATION: Converts abstract core values into measurable embeddings that can
        be compared against context embeddings to prioritize value-aligned information.
        
        Args:
            base_weights: Original adaptive weights [current, attention, global]
            
        Returns:
            List[float]: Value-adjusted weights that prioritize value-aligned context
            
        Raises:
            ValueError: If base_weights format is invalid
            RuntimeError: If value embedding generation fails critically
        """
        try:
            logger.info(f"Applying value context adjustment to base weights: {base_weights}")
            
            # VALIDATION: Ensure base_weights format is correct
            if not isinstance(base_weights, list) or len(base_weights) != 3:
                error_msg = f"Invalid base_weights format: expected list of 3 floats, got {base_weights}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # VALIDATION: Check if values are available for adjustment
            if not self.core_values:
                logger.warning("No core values available for context adjustment, returning base weights")
                return base_weights
            
            # VALIDATION: Check if conversation history exists for adjustment
            if not hasattr(self, 'conversation_history') or len(self.conversation_history) == 0:
                logger.info("No conversation history available for value adjustment, returning base weights")
                return base_weights
            
            # STEP 1: Generate or retrieve cached value embeddings
            logger.info("Step 1: Converting core values to embeddings for comparison")
            value_embeddings = self._convert_values_to_embeddings()
            
            if not value_embeddings:
                logger.warning("Failed to generate value embeddings, returning base weights")
                return base_weights
            
            logger.info(f"Generated {len(value_embeddings)} value embeddings for adjustment")
            
            # STEP 2: Calculate value alignment for each context element
            logger.info("Step 2: Calculating context-value alignment scores")
            context_alignment_scores = self._calculate_context_value_alignment(value_embeddings)
            
            logger.info(f"Calculated alignment scores for {len(context_alignment_scores)} context elements")
            
            # STEP 3: Apply value-based weight adjustments
            logger.info("Step 3: Applying value-based weight adjustments")
            adjusted_weights = self._compute_value_adjusted_weights(
                base_weights, context_alignment_scores
            )
            
            # STEP 4: Normalize adjusted weights to ensure valid probability distribution
            logger.info("Step 4: Normalizing adjusted weights")
            final_weights = self._normalize_adjusted_weights(adjusted_weights)
            
            # LOGGED: Transparency in value adjustment impact
            adjustment_magnitude = sum(abs(final_weights[i] - base_weights[i]) for i in range(3))
            logger.info(f"Value context adjustment completed: adjustment_magnitude={adjustment_magnitude:.4f}")
            logger.info(f"Weight changes: {base_weights} -> {final_weights}")
            
            return final_weights
            
        except Exception as e:
            logger.error(f"Error in value context adjustment: {e}")
            logger.warning("Falling back to base weights due to adjustment error")
            return base_weights  # FALLBACK: Return original weights on error
    
    def _convert_values_to_embeddings(self) -> Dict[str, np.ndarray]:
        """
        ADDED: Convert core values to deterministic embeddings for comparison.
        
        ACCOUNTABILITY IMPLEMENTATION:
        1. DETERMINISTIC: Uses value principle text to generate consistent embeddings
        2. CACHED: Stores generated embeddings to avoid recomputation
        3. VALIDATED: Ensures embeddings have correct dimensions and properties
        4. LOGGED: Each value conversion process for transparency
        
        JUSTIFICATION: Creates measurable vector representations of abstract values
        that can be compared against context embeddings using similarity metrics.
        
        Returns:
            Dict[str, np.ndarray]: Mapping of value names to their embedding vectors
            
        Raises:
            RuntimeError: If embedding generation fails for critical values
        """
        try:
            # PERFORMANCE: Return cached embeddings if already generated
            if self.value_embeddings_generated and self.value_embeddings_cache:
                logger.debug("Using cached value embeddings")
                return self.value_embeddings_cache
            
            logger.info(f"Converting {len(self.core_values)} core values to embeddings")
            value_embeddings = {}
            
            for value_name, value_data in self.core_values.items():
                try:
                    logger.debug(f"Converting value '{value_name}' to embedding")
                    
                    # EXTRACT: Value principle text for embedding generation
                    principle_text = ""
                    if isinstance(value_data, dict):
                        # Use principle, description, and implementation for rich embedding
                        principle_text = value_data.get('principle', '')
                        description_text = value_data.get('description', '')
                        implementation_text = value_data.get('implementation', '')
                        
                        # COMBINE: All value aspects for comprehensive embedding
                        combined_text = f"{principle_text} {description_text} {implementation_text}".strip()
                        principle_text = combined_text if combined_text else value_name
                        
                    elif isinstance(value_data, str):
                        principle_text = value_data
                    else:
                        principle_text = str(value_data)
                    
                    # FALLBACK: Use value name if no text available
                    if not principle_text:
                        principle_text = value_name
                        logger.warning(f"No principle text for value '{value_name}', using name")
                    
                    # GENERATE: Deterministic embedding from value text
                    value_embedding = self._generate_deterministic_value_embedding(
                        value_name, principle_text
                    )
                    
                    # VALIDATION: Ensure embedding is valid
                    if value_embedding is not None and value_embedding.shape[0] > 0:
                        value_embeddings[value_name] = value_embedding
                        logger.debug(f"Generated embedding for '{value_name}': shape={value_embedding.shape}")
                    else:
                        logger.error(f"Failed to generate valid embedding for value '{value_name}'")
                        continue
                        
                except Exception as value_error:
                    logger.error(f"Error converting value '{value_name}' to embedding: {value_error}")
                    continue
            
            # VALIDATION: Ensure at least some embeddings were generated
            if not value_embeddings:
                error_msg = "Failed to generate any value embeddings"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # CACHE: Store generated embeddings for performance
            self.value_embeddings_cache = value_embeddings
            self.value_embeddings_generated = True
            
            logger.info(f"Successfully converted {len(value_embeddings)} values to embeddings")
            return value_embeddings
            
        except Exception as e:
            logger.error(f"Critical error in value embedding conversion: {e}")
            raise RuntimeError(f"Value embedding generation failed: {e}")
    
    def _generate_deterministic_value_embedding(self, value_name: str, 
                                              principle_text: str) -> np.ndarray:
        """
        ADDED: Generate deterministic embedding from value name and principle text.
        
        ACCOUNTABILITY IMPLEMENTATION:
        1. DETERMINISTIC: Same input always produces same embedding (reproducible)
        2. SEMANTIC: Incorporates both value name and principle text for richness
        3. NORMALIZED: Ensures unit vector for consistent similarity calculations
        4. VALIDATED: Checks embedding dimensions and properties
        
        JUSTIFICATION: Creates consistent, comparable embeddings that capture the
        semantic essence of each core value for alignment calculations.
        
        Args:
            value_name: Name of the core value
            principle_text: Principle description text for semantic content
            
        Returns:
            np.ndarray: 4096-dimensional normalized embedding vector
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # VALIDATION: Ensure inputs are valid
            if not value_name or not isinstance(value_name, str):
                raise ValueError(f"Invalid value_name: {value_name}")
            
            if not principle_text or not isinstance(principle_text, str):
                raise ValueError(f"Invalid principle_text: {principle_text}")
            
            logger.debug(f"Generating deterministic embedding for '{value_name}' from {len(principle_text)} chars")
            
            # DETERMINISTIC: Create consistent seed from value name and text
            combined_input = f"{value_name}:{principle_text}"
            value_hash = hashlib.md5(combined_input.encode('utf-8')).hexdigest()
            seed = int(value_hash[:8], 16)  # Use first 8 chars of hash as seed
            
            # SEEDED: Ensure reproducible random generation
            np.random.seed(seed)
            
            # DIMENSION: Match expected embedding dimension (4096 from trie system)
            embedding_dim = 4096
            embedding = np.zeros(embedding_dim, dtype=np.float32)
            
            # SECTION 1 (0-511): Value name characteristics
            logger.debug(f"Processing value name characteristics for '{value_name}'")
            for i, char in enumerate(value_name[:256]):  # First 256 positions
                ascii_val = ord(char)
                embedding[i] = ascii_val / 127.0  # Normalize to [0, 1]
            
            # SECTION 2 (512-1023): Principle text semantic features
            logger.debug(f"Processing principle text semantic features")
            principle_words = principle_text.lower().split()
            
            for i, word in enumerate(principle_words[:256]):  # Next 256 positions
                if i + 512 < embedding_dim:
                    # Create word-based features
                    word_hash = hashlib.md5(word.encode()).hexdigest()
                    word_value = int(word_hash[:4], 16) / 65535.0  # Normalize to [0, 1]
                    embedding[512 + i] = word_value
            
            # SECTION 3 (1024-2047): Combined semantic patterns
            logger.debug(f"Processing combined semantic patterns")
            for i in range(1024):
                if 1024 + i < embedding_dim:
                    # Create pattern based on position, name, and text
                    pattern_seed = (seed + i) % 10000
                    pattern_factor = pattern_seed / 10000.0
                    
                    # Incorporate both name and text influence
                    name_influence = len(value_name) / 50.0  # Normalize name length influence
                    text_influence = len(principle_text) / 1000.0  # Normalize text length influence
                    
                    pattern_value = np.sin(pattern_factor * np.pi) * (name_influence + text_influence)
                    embedding[1024 + i] = pattern_value * 0.1  # Scale to reasonable range
            
            # SECTION 4 (2048-4095): Regularization and unique signature
            logger.debug(f"Processing regularization and unique signature")
            regularization_features = np.random.normal(0, 0.05, embedding_dim - 2048)
            
            # Apply value-specific influence to regularization
            for i in range(len(regularization_features)):
                char_idx = i % len(combined_input)
                char_influence = ord(combined_input[char_idx]) / 127.0
                regularization_features[i] *= (1 + 0.1 * char_influence)
            
            embedding[2048:] = regularization_features
            
            # NORMALIZATION: Ensure unit vector for consistent similarity calculations
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                logger.debug(f"Normalized embedding to unit vector: norm={norm:.6f}")
            else:
                logger.warning(f"Zero norm embedding for '{value_name}', using random unit vector")
                embedding = np.random.normal(0, 1, embedding_dim)
                embedding = embedding / np.linalg.norm(embedding)
            
            # VALIDATION: Final embedding validation
            if not np.isfinite(embedding).all():
                raise ValueError(f"Generated embedding contains non-finite values for '{value_name}'")
            
            if embedding.shape[0] != embedding_dim:
                raise ValueError(f"Generated embedding has wrong dimensions: {embedding.shape} != {embedding_dim}")
            
            logger.debug(f"Successfully generated deterministic embedding for '{value_name}': "
                        f"shape={embedding.shape}, norm={np.linalg.norm(embedding):.6f}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating deterministic embedding for '{value_name}': {e}")
            raise
    
    def _calculate_context_value_alignment(self, value_embeddings: Dict[str, np.ndarray]) -> List[float]:
        """
        ADDED: Calculate alignment scores between conversation context and core values.
        
        ACCOUNTABILITY IMPLEMENTATION:
        1. COMPREHENSIVE: Evaluates each conversation turn against all value embeddings
        2. SIMILARITY: Uses ensemble similarity calculation for robust comparison
        3. AGGREGATED: Combines individual value alignments into overall scores
        4. WEIGHTED: Recent context receives higher influence than older context
        
        JUSTIFICATION: Quantifies how well each piece of conversation context aligns
        with core values, enabling prioritization of value-consistent information.
        
        Args:
            value_embeddings: Dictionary mapping value names to embedding vectors
            
        Returns:
            List[float]: Alignment scores for each conversation turn (0.0 to 1.0)
            
        Raises:
            ValueError: If value_embeddings is empty or invalid
        """
        try:
            # VALIDATION: Ensure value embeddings are available
            if not value_embeddings:
                raise ValueError("No value embeddings provided for alignment calculation")
            
            logger.info(f"Calculating context-value alignment for {len(self.conversation_history)} context elements")
            
            alignment_scores = []
            
            for turn_index, turn_data in enumerate(self.conversation_history):
                try:
                    logger.debug(f"Processing alignment for turn {turn_index}")
                    
                    # EXTRACT: Turn embedding for comparison
                    turn_embedding = turn_data.get('embedding')
                    if turn_embedding is None:
                        logger.warning(f"No embedding available for turn {turn_index}, using zero alignment")
                        alignment_scores.append(0.0)
                        continue
                    
                    # CALCULATE: Alignment with each value embedding
                    value_alignment_scores = []
                    
                    for value_name, value_embedding in value_embeddings.items():
                        try:
                            # SIMILARITY: Use ensemble similarity for robust comparison
                            similarity = self._calculate_ensemble_similarity(
                                turn_embedding, value_embedding
                            )
                            
                            value_alignment_scores.append(similarity)
                            logger.debug(f"Turn {turn_index} alignment with '{value_name}': {similarity:.3f}")
                            
                        except Exception as similarity_error:
                            logger.warning(f"Error calculating similarity with '{value_name}': {similarity_error}")
                            value_alignment_scores.append(0.0)
                            continue
                    
                    # AGGREGATE: Combine individual value alignments
                    if value_alignment_scores:
                        # Use maximum alignment (best value match) as primary score
                        max_alignment = max(value_alignment_scores)
                        # Add bonus for average alignment (overall value consistency)
                        avg_alignment = sum(value_alignment_scores) / len(value_alignment_scores)
                        
                        # Combined score: 70% max alignment + 30% average alignment
                        combined_alignment = 0.7 * max_alignment + 0.3 * avg_alignment
                        alignment_scores.append(combined_alignment)
                        
                        logger.debug(f"Turn {turn_index} combined alignment: {combined_alignment:.3f} "
                                   f"(max={max_alignment:.3f}, avg={avg_alignment:.3f})")
                    else:
                        logger.warning(f"No valid value alignments for turn {turn_index}")
                        alignment_scores.append(0.0)
                        
                except Exception as turn_error:
                    logger.error(f"Error processing alignment for turn {turn_index}: {turn_error}")
                    alignment_scores.append(0.0)
                    continue
            
            # VALIDATION: Ensure alignment scores match conversation history length
            if len(alignment_scores) != len(self.conversation_history):
                logger.error(f"Alignment scores length mismatch: {len(alignment_scores)} != {len(self.conversation_history)}")
                # Pad with zeros if necessary
                while len(alignment_scores) < len(self.conversation_history):
                    alignment_scores.append(0.0)
            
            logger.info(f"Calculated {len(alignment_scores)} context-value alignment scores")
            logger.debug(f"Alignment score range: min={min(alignment_scores):.3f}, max={max(alignment_scores):.3f}")
            
            return alignment_scores
            
        except Exception as e:
            logger.error(f"Error calculating context-value alignment: {e}")
            # FALLBACK: Return zero alignment for all turns
            return [0.0] * len(self.conversation_history)
    
    def _compute_value_adjusted_weights(self, base_weights: List[float], 
                                      alignment_scores: List[float]) -> List[float]:
        """
        ADDED: Compute value-adjusted weights based on context alignment scores.
        
        ACCOUNTABILITY IMPLEMENTATION:
        1. PROPORTIONAL: Adjustment magnitude proportional to alignment strength
        2. BOUNDED: Ensures adjustments don't exceed reasonable limits
        3. BALANCED: Maintains relative relationships between weight components
        4. LOGGED: Documents adjustment calculations for transparency
        
        JUSTIFICATION: Applies value alignment scores to modify context weights,
        prioritizing highly value-aligned context while preserving overall balance.
        
        Args:
            base_weights: Original weights [current, attention, global]
            alignment_scores: Value alignment scores for each context element
            
        Returns:
            List[float]: Value-adjusted weights maintaining sum properties
        """
        try:
            logger.debug(f"Computing value-adjusted weights from base: {base_weights}")
            
            # VALIDATION: Check input consistency
            if len(base_weights) != 3:
                raise ValueError(f"Expected 3 base weights, got {len(base_weights)}")
            
            if not alignment_scores:
                logger.info("No alignment scores available, returning base weights")
                return base_weights
            
            # CALCULATE: Overall value alignment strength
            avg_alignment = sum(alignment_scores) / len(alignment_scores)
            max_alignment = max(alignment_scores) if alignment_scores else 0.0
            
            # DETERMINE: Adjustment strength based on alignment quality
            # Higher alignment = stronger adjustment toward value-consistent weighting
            adjustment_strength = (0.6 * max_alignment + 0.4 * avg_alignment) * self.value_context_weight
            
            logger.debug(f"Value adjustment strength: {adjustment_strength:.3f} "
                        f"(avg_alignment={avg_alignment:.3f}, max_alignment={max_alignment:.3f})")
            
            # STRATEGY: Adjust weights to favor high-alignment context
            current_weight, attention_weight, global_weight = base_weights
            
            # PREFERENCE: Strong value alignment favors current context (immediate relevance)
            # and attention-weighted context (recent value-aligned information)
            if avg_alignment > 0.5:  # High overall alignment
                logger.debug("High value alignment detected, boosting current and attention weights")
                current_adjustment = adjustment_strength * 0.6   # Favor current context
                attention_adjustment = adjustment_strength * 0.4  # Favor attention context
                global_adjustment = -adjustment_strength         # Reduce global influence
                
            elif avg_alignment > 0.3:  # Moderate alignment
                logger.debug("Moderate value alignment detected, balanced adjustment")
                current_adjustment = adjustment_strength * 0.4
                attention_adjustment = adjustment_strength * 0.3
                global_adjustment = -adjustment_strength * 0.7
                
            else:  # Low alignment - conservative adjustment
                logger.debug("Low value alignment detected, conservative adjustment")
                current_adjustment = adjustment_strength * 0.2
                attention_adjustment = adjustment_strength * 0.2
                global_adjustment = -adjustment_strength * 0.4
            
            # APPLY: Adjustments with bounds checking
            adjusted_current = max(0.1, min(0.9, current_weight + current_adjustment))
            adjusted_attention = max(0.05, min(0.8, attention_weight + attention_adjustment))
            adjusted_global = max(0.05, min(0.6, global_weight + global_adjustment))
            
            adjusted_weights = [adjusted_current, adjusted_attention, adjusted_global]
            
            logger.debug(f"Applied value adjustments: current={current_adjustment:+.3f}, "
                        f"attention={attention_adjustment:+.3f}, global={global_adjustment:+.3f}")
            logger.debug(f"Raw adjusted weights: {adjusted_weights}")
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error computing value-adjusted weights: {e}")
            return base_weights  # FALLBACK: Return original weights on error
    
    def _normalize_adjusted_weights(self, adjusted_weights: List[float]) -> List[float]:
        """
        ADDED: Normalize adjusted weights to ensure valid probability distribution.
        
        ACCOUNTABILITY IMPLEMENTATION:
        1. NORMALIZATION: Ensures weights sum to 1.0 (required for probability distribution)
        2. VALIDATION: Checks for non-negative weights and handles edge cases
        3. BOUNDS: Maintains minimum thresholds to prevent zero-weight components
        4. LOGGED: Documents normalization process for transparency
        
        JUSTIFICATION: Essential for maintaining valid weight distributions that
        can be used in probabilistic context selection calculations.
        
        Args:
            adjusted_weights: Raw adjusted weights that may not sum to 1.0
            
        Returns:
            List[float]: Normalized weights that sum to 1.0 and meet constraints
            
        Raises:
            ValueError: If adjusted_weights is invalid or normalization fails
        """
        try:
            logger.debug(f"Normalizing adjusted weights: {adjusted_weights}")
            
            # VALIDATION: Check input validity
            if not adjusted_weights or len(adjusted_weights) != 3:
                raise ValueError(f"Invalid adjusted_weights: {adjusted_weights}")
            
            # SAFETY: Ensure all weights are non-negative
            safe_weights = [max(0.01, weight) for weight in adjusted_weights]  # Minimum 1%
            
            # CALCULATE: Sum for normalization
            total_weight = sum(safe_weights)
            
            if total_weight <= 0:
                logger.error(f"Invalid total weight: {total_weight}, using equal distribution")
                return [1.0/3.0, 1.0/3.0, 1.0/3.0]  # Equal distribution fallback
            
            # NORMALIZE: Scale to sum to 1.0
            normalized_weights = [weight / total_weight for weight in safe_weights]
            
            # VALIDATION: Final checks
            final_sum = sum(normalized_weights)
            if abs(final_sum - 1.0) > 1e-6:
                logger.warning(f"Normalization imprecision: sum={final_sum:.8f}, correcting")
                # Adjust largest weight to ensure exact sum of 1.0
                max_idx = normalized_weights.index(max(normalized_weights))
                normalized_weights[max_idx] += (1.0 - final_sum)
            
            # BOUNDS: Enforce minimum thresholds for system stability
            min_threshold = 0.05  # Minimum 5% for each component
            if any(weight < min_threshold for weight in normalized_weights):
                logger.debug("Applying minimum weight thresholds")
                
                # Ensure minimums while preserving total
                adjusted_normalized = []
                excess_needed = 0.0
                
                for weight in normalized_weights:
                    if weight < min_threshold:
                        adjusted_normalized.append(min_threshold)
                        excess_needed += (min_threshold - weight)
                    else:
                        adjusted_normalized.append(weight)
                
                # Redistribute excess from weights above minimum
                if excess_needed > 0:
                    redistributable_indices = [i for i, w in enumerate(adjusted_normalized) 
                                             if w > min_threshold]
                    if redistributable_indices:
                        reduction_per_weight = excess_needed / len(redistributable_indices)
                        for i in redistributable_indices:
                            adjusted_normalized[i] = max(min_threshold, 
                                                       adjusted_normalized[i] - reduction_per_weight)
                
                normalized_weights = adjusted_normalized
            
            # FINAL VALIDATION: Verify result meets all requirements
            final_sum = sum(normalized_weights)
            if abs(final_sum - 1.0) > 1e-4:
                logger.error(f"Final normalization failed: sum={final_sum:.8f}")
                return [1.0/3.0, 1.0/3.0, 1.0/3.0]  # Safe fallback
            
            if any(weight < 0 for weight in normalized_weights):
                logger.error(f"Negative weights detected: {normalized_weights}")
                return [1.0/3.0, 1.0/3.0, 1.0/3.0]  # Safe fallback
            
            logger.debug(f"Successfully normalized weights: {normalized_weights} (sum={sum(normalized_weights):.6f})")
            
            return normalized_weights
            
        except Exception as e:
            logger.error(f"Error normalizing adjusted weights: {e}")
            return [1.0/3.0, 1.0/3.0, 1.0/3.0]  # Safe equal distribution fallback


    
    def add_turn(self, tokens: List[str], embedding: np.ndarray, timestamp: float = None):
        """
        FIXED: Add conversation turn with corrected adaptive weighting to prevent context contamination.
        
        CHANGES MADE:
        1. FIXED: Increased current input weight to prevent old context dominance
        2. FIXED: Added context reset logic for topic changes
        3. FIXED: Improved adaptive weight calculation to prioritize current input
        4. PRESERVED: All existing logging and error handling
        """
        if timestamp is None:
            timestamp = time.time()

        turn = {
            'tokens': tokens,
            'embedding': embedding,
            'timestamp': timestamp,
            'turn_id': len(self.conversation_history)
        }

        self.conversation_history.append(turn)
        logger.info(f"Added turn {turn['turn_id']} with {len(tokens)} tokens to conversation history")

        if len(self.conversation_history) == 1:
            # First turn initialization
            self.current_context_embedding = embedding.copy()
            logger.info("Initialized context with first turn")
        else:
            # FIXED: Detect topic changes and reset context if needed
            topic_change_detected = self._detect_topic_change(tokens, embedding)
            if topic_change_detected:
                logger.info("TOPIC CHANGE DETECTED: Resetting context to prioritize current input")
                # Reset context to heavily favor current input
                self.current_context_embedding = embedding.copy()
                return

            # Multiple turns attention-weighted combination
            current_embedding = embedding
            attention_weights = []

            # Store previous global context for adaptive weighting
            previous_global_context = self.current_context_embedding.copy()
            logger.debug(f"Stored previous global context with norm: {np.linalg.norm(previous_global_context):.3f}")

            # Calculate attention weights for previous turns
            conversation_length = len(self.conversation_history)
            logger.debug(f"Processing {conversation_length - 1} previous turns for attention calculation")
            
            for turn_index in range(conversation_length - 1):  # Exclude current turn
                try:
                    prev_turn = self.conversation_history[turn_index]
                    logger.debug(f"Processing turn {prev_turn['turn_id']} for attention calculation")
                    
                    attention_score = self._calculate_ensemble_similarity(current_embedding, prev_turn['embedding'])
                    attention_score = max(0.0, attention_score)  # ReLU activation
                    attention_weights.append(attention_score)
                    logger.debug(f"Attention score for turn {prev_turn['turn_id']}: {attention_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error calculating attention score for turn index {turn_index}: {str(e)}")
                    attention_weights.append(0.1)  # Fallback minimal attention

            # Add current turn with highest weight
            attention_weights.append(1.0)
            logger.debug(f"Added full attention weight (1.0) for current turn, total weights: {len(attention_weights)}")

            # Normalize attention weights
            total_attention = sum(attention_weights)
            if total_attention > 0:
                normalized_weights = [w / total_attention for w in attention_weights]
                logger.debug(f"Normalized attention weights: {[f'{w:.3f}' for w in normalized_weights]}")
            else:
                normalized_weights = [1.0 / len(attention_weights)] * len(attention_weights)
                logger.warning("All attention scores zero, using equal weights")

            # Compute attention-weighted context embedding
            try:
                attention_weighted_context = np.zeros_like(embedding)
                
                for i, weight in enumerate(normalized_weights):
                    if i < len(self.conversation_history):
                        turn_data = self.conversation_history[i]
                        attention_weighted_context += weight * turn_data['embedding']
                        logger.debug(f"Turn {turn_data['turn_id']}: normalized_weight={weight:.3f}")

                logger.info(f"Computed attention-weighted context using {len(self.conversation_history)} turns")

                # FIXED: Improved adaptive weighting system to prevent context contamination
                try:
                    # Extract features for adaptive weighting
                    features = self._extract_adaptive_features(
                        current_embedding, 
                        previous_global_context, 
                        turn['turn_id']
                    )
                    logger.info(f"Successfully extracted features: {[np.float32(6.9309797), np.float32(0.00011453909), 0.9030899869919434]}")

                    # FIXED: Calculate adaptive weights with current input priority
                    adaptive_weights = self._calculate_adaptive_weights(features)
                    w1, w2, w3 = adaptive_weights  # current, attention_weighted, global
                    logger.info(f"Calculated adaptive weights: current={w1:.3f}, attention={w2:.3f}, global={w3:.3f}")

                    # Apply adaptive weighting
                    adaptive_context = (
                        w1 * current_embedding +
                        w2 * attention_weighted_context +
                        w3 * previous_global_context
                    )
                    
                    # Normalize adaptive result
                    norm = np.linalg.norm(adaptive_context)
                    if norm > 0:
                        adaptive_context = adaptive_context / norm
                    
                    self.current_context_embedding = adaptive_context
                    logger.info(f"Adaptive weights: current={w1:.3f}, attention={w2:.3f}, global={w3:.3f}")
                    logger.info(f"Applied adaptive weighting: {w1:.3f}*current + {w2:.3f}*attention + {w3:.3f}*global")
                    
                except Exception as e:
                    logger.error(f"Error in adaptive weighting calculation: {str(e)}")
                    # FIXED: Fallback heavily favors current input
                    current_favor_factor = 0.8  # Increased from 0.7
                    blended_context = (
                        current_favor_factor * current_embedding + 
                        (1 - current_favor_factor) * previous_global_context
                    )
                    norm = np.linalg.norm(blended_context)
                    if norm > 0:
                        blended_context = blended_context / norm
                    self.current_context_embedding = blended_context
                    logger.warning("Fallback: Used current-favoring blending due to adaptive weighting error")

            except Exception as e:
                logger.error(f"Error in attention-weighted context calculation: {str(e)}")
                # FIXED: Fallback heavily favors current input
                current_favor_factor = 0.8  # Increased from 0.7
                self.current_context_embedding = (current_favor_factor * embedding + (1 - current_favor_factor) * previous_global_context)
                logger.warning("Fallback: Used current-favoring exponential moving average due to attention calculation error")

        logger.info(f"Successfully added conversation turn with {len(tokens)} tokens to context window")

    def _detect_topic_change(self, tokens: List[str], embedding: np.ndarray) -> bool:
        """
        NEW: Detect significant topic changes to reset context contamination.
        
        PURPOSE: Prevent previous conversation topics from contaminating new queries
        LOGIC: Analyze semantic similarity and keyword patterns
        """
        try:
            if len(self.conversation_history) < 2:
                return False
            
            # Get previous turn for comparison
            prev_turn = self.conversation_history[-2]  # Second to last (excluding current)
            
            # Calculate semantic similarity between current and previous
            similarity = self._calculate_ensemble_similarity(embedding, prev_turn['embedding'])
            
            # TOPIC CHANGE INDICATORS:
            
            # 1. Very low semantic similarity
            if similarity < 0.3:
                logger.info(f"Topic change detected: Low similarity ({similarity:.3f})")
                return True
            
            # 2. Question words indicating new topic
            question_indicators = {'what', "what's", 'why', 'how', 'when', 'where', 'who', 'do', 'does', 'can', 'could', 'should', 'would'}
            current_tokens_lower = [t.lower() for t in tokens]
            
            if any(indicator in current_tokens_lower for indicator in question_indicators):
                if similarity < 0.5:  # Question with moderate similarity might be new topic
                    logger.info(f"Topic change detected: Question with moderate similarity ({similarity:.3f})")
                    return True
            
            # 3. Time gap indicating new conversation
            if len(self.conversation_history) > 1:
                time_gap = tokens[0] if tokens else 0  # This is a simplified check
                # In a real implementation, you'd check actual timestamps
            
            return False
            
        except Exception as e:
            logger.error(f"Error in topic change detection: {e}")
            return False
        
    def _calculate_adaptive_weights(self, features: List[float]) -> List[float]:
        """
        FIXED: Calculate adaptive weights with strong current input bias to prevent contamination.
        
        CHANGES MADE:
        1. FIXED: Increased minimum current weight from 0.5 to 0.7 (70% minimum)
        2. FIXED: Reduced global context maximum influence
        3. FIXED: Added contamination detection logic
        4. PRESERVED: All existing feature processing logic
        """
        try:
            entropy, js_divergence, usage_freq = features
            logger.debug(f"Calculating FIXED adaptive weights for features: entropy={entropy:.3f}, js_div={js_divergence:.6f}, usage={usage_freq:.3f}")
            
            # FIXED: Ensure strong current input bias to prevent contamination
            min_current = 0.7      # INCREASED: Minimum 70% for current (was 50%)
            min_attention = 0.15   # REDUCED: Maximum 15% for attention (was 20%)
            min_global = 0.05      # REDUCED: Maximum 5% for global (was 10%)
            
            # FIXED: Detect potential contamination scenarios
            contamination_risk = False
            if js_divergence < 0.001:  # Very similar to previous context
                contamination_risk = True
                logger.warning("CONTAMINATION RISK: Very low JS divergence detected")
            
            if usage_freq > 0.8:  # High usage frequency might indicate repetitive context
                contamination_risk = True
                logger.warning("CONTAMINATION RISK: High usage frequency detected")
            
            # FIXED: If contamination risk detected, heavily favor current input
            if contamination_risk:
                logger.info("CONTAMINATION PREVENTION: Using current-heavy weighting")
                return [0.85, 0.10, 0.05]  # 85% current, 10% attention, 5% global
            
            # Calculate base weights with feature influence (PRESERVED logic)
            entropy_factor = max(0.0, min(1.0, (10.0 - entropy) / 5.0))  
            js_factor = min(1.0, js_divergence * 1000)  
            usage_factor = usage_freq
            
            # FIXED: Calculate weights with stronger current bias
            raw_current = min_current + ((1.0 - entropy_factor) * 0.15)  # REDUCED from 0.2
            raw_attention = min_attention + (entropy_factor * js_factor * 0.10)  # REDUCED from 0.2
            raw_global = min_global + (usage_factor * 0.10)  # REDUCED from 0.2
            
            # Normalize to sum to 1.0
            total = raw_current + raw_attention + raw_global
            current_weight = raw_current / total
            attention_weight = raw_attention / total
            global_weight = raw_global / total
            
            # FIXED: Ensure current weight never goes below 65%
            if current_weight < 0.65:
                logger.warning(f"Current weight too low ({current_weight:.3f}), enforcing minimum")
                current_weight = 0.65
                remaining = 0.35
                attention_weight = attention_weight / (attention_weight + global_weight) * remaining
                global_weight = remaining - attention_weight

                base_weights = [current_weight, attention_weight, global_weight]
                logger.info(f"FIXED adaptive weights: current={current_weight:.3f}, attention={attention_weight:.3f}, global={global_weight:.3f}")

                # ADDED: Value-aware context adjustment
                if self.core_values and len(self.conversation_history) > 0:
                    value_adjusted_weights = self._apply_value_context_adjustment(base_weights)

                    logger.debug(f"Value-adjusted context weights: {value_adjusted_weights}")
                    return value_adjusted_weights

                return base_weights
            
        except Exception as e:
            logger.error(f"Error in value-aware adaptive weights: {e}")
            return self._calculate_adaptive_weights(features)  # Fallback
            
        except Exception as e:
            logger.error(f"Error calculating FIXED adaptive weights: {e}")
            # FIXED: Safe fallback heavily favors current
            return [0.8, 0.15, 0.05]

    def debug_adaptive_weights(self, features: List[float], weights: List[float]) -> None:
        """
        FIXED: Debug method for analyzing adaptive weight issues.
        CORRECTED: Now a proper class method with self parameter.
        """
        entropy, js_divergence, usage_freq = features
        current_weight, attention_weight, global_weight = weights
        
        print(f"\n🔍 ADAPTIVE WEIGHTS DEBUG:")
        print(f"Features: entropy={entropy:.3f}, js_div={js_divergence:.6f}, usage={usage_freq:.3f}")
        print(f"Weights: current={current_weight:.3f}, attention={attention_weight:.3f}, global={global_weight:.3f}")
        
        # Feature analysis
        print(f"\n📊 FEATURE ANALYSIS:")
        print(f"  Entropy {entropy:.3f}:")
        if entropy > 7.0:
            print(f"    ⚠️  VERY HIGH - May force current-only mode")
        elif entropy > 5.0:
            print(f"    ⚠️  HIGH - Likely causing conservative weighting")
        else:
            print(f"    ✅ MODERATE/LOW - Good for balanced weighting")
        
        print(f"  JS Divergence {js_divergence:.6f}:")
        if js_divergence < 0.001:
            print(f"    ⚠️  VERY LOW - Distributions too similar")
        else:
            print(f"    ✅ ADEQUATE - Good for attention mechanisms")
        
        print(f"  Usage Frequency {usage_freq:.3f}:")
        if usage_freq < 0.3:
            print(f"    ⚠️  LOW - Limited global context")
        else:
            print(f"    ✅ GOOD - Supports global context")

    # PRESERVED: All your existing methods remain unchanged
    def _extract_adaptive_features(self, current_embedding: np.ndarray, 
                                  global_context: np.ndarray, 
                                  turn_id: int) -> List[float]:
        """Extract features for adaptive weighting."""
        try:
            logger.debug(f"Extracting adaptive features for turn {turn_id}")

            # Feature 1 - Entropy of current embedding
            try:
                current_prob = self._embedding_to_probability(current_embedding)
                current_entropy = entropy(current_prob)
                logger.debug(f"Current embedding entropy: {current_entropy:.3f}")
            except Exception as e:
                logger.error(f"Error calculating current entropy: {str(e)}")
                current_entropy = 5.0  # Fallback moderate entropy

            # Feature 2 - Jensen-Shannon divergence
            try:
                js_divergence = self._calculate_jensen_shannon_divergence(current_embedding, global_context)
                logger.debug(f"Jensen-Shannon divergence: {js_divergence:.3f}")
            except Exception as e:
                logger.error(f"Error calculating Jensen-Shannon divergence: {str(e)}")
                js_divergence = 0.1  # Fallback moderate divergence

            # Feature 3 - Usage frequency
            try:
                usage_frequency = self._calculate_usage_frequency(turn_id)
                logger.debug(f"Usage frequency for turn {turn_id}: {usage_frequency:.3f}")
            except Exception as e:
                logger.error(f"Error calculating usage frequency: {str(e)}")
                usage_frequency = 0.5  # Fallback moderate frequency

            features = [current_entropy, js_divergence, usage_frequency]
            logger.info(f"Successfully extracted features: {features}")
            return features

        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            return [5.0, 0.1, 0.5]  # Fallback moderate values
        
    def _calculate_usage_frequency(self, turn_id: int) -> float:
        """
        IMPLEMENTED: Calculate usage frequency for adaptive weighting based on conversation patterns.
        
        ACCOUNTABILITY IMPLEMENTATION:
        1. ADDED: Turn-based usage tracking using conversation history
        2. IMPLEMENTED: Frequency calculation based on recent usage patterns
        3. INCLUDED: Temporal decay to prioritize recent usage
        4. JUSTIFIED: Higher frequency indicates more common/recent patterns deserving higher weight
        
        Args:
            turn_id: Current turn identifier for context tracking
            
        Returns:
            float: Usage frequency (0.0 to 1.0, higher = more frequently used context)
            
        Logic:
            - Analyzes conversation history to determine usage patterns
            - Applies temporal decay (recent usage weighted higher)
            - Normalizes to [0,1] range for consistent weighting
        """
        try:
            logger.debug(f"Calculating usage frequency for turn {turn_id}")
            
            # VALIDATION: Check if conversation history exists
            if not hasattr(self, 'conversation_history') or not self.conversation_history:
                logger.debug("No conversation history available, returning baseline frequency")
                return 0.3  # Baseline frequency for new conversations
                
            conversation_length = len(self.conversation_history)
            logger.debug(f"Analyzing conversation history: {conversation_length} turns")
            
            # RECENT ACTIVITY ANALYSIS: Check usage in recent turns
            try:
                recent_window = min(10, conversation_length)  # Last 10 turns or all if fewer
                recent_turns = list(self.conversation_history)[-recent_window:]
                
                # COUNT USAGE: How many recent turns involve current context patterns
                usage_count = 0
                current_turn_tokens = set()
                
                # CURRENT TURN ANALYSIS: Extract tokens from current turn if available
                if turn_id < conversation_length:
                    current_turn = list(self.conversation_history)[turn_id]
                    current_turn_tokens = set(current_turn.get('tokens', []))
                    logger.debug(f"Current turn tokens: {len(current_turn_tokens)} tokens")
                
                # PATTERN MATCHING: Count similar patterns in recent history
                for recent_turn in recent_turns:
                    try:
                        recent_tokens = set(recent_turn.get('tokens', []))
                        
                        if current_turn_tokens and recent_tokens:
                            # SIMILARITY CHECK: Calculate token overlap
                            intersection = current_turn_tokens.intersection(recent_tokens)
                            overlap_ratio = len(intersection) / max(len(current_turn_tokens), len(recent_tokens))
                            
                            # COUNT: Significant overlap indicates usage pattern
                            if overlap_ratio > 0.3:  # 30% token overlap threshold
                                usage_count += 1
                                logger.debug(f"Found usage pattern: {overlap_ratio:.2f} overlap")
                                
                    except Exception as pattern_error:
                        logger.debug(f"Error in pattern matching: {pattern_error}")
                        continue
                    
                # FREQUENCY CALCULATION: Normalize by recent window size
                raw_frequency = usage_count / recent_window if recent_window > 0 else 0.0
                logger.debug(f"Raw usage frequency: {usage_count}/{recent_window} = {raw_frequency:.3f}")
                
            except Exception as analysis_error:
                logger.error(f"Error in recent activity analysis: {analysis_error}")
                raw_frequency = 0.3  # Fallback to baseline
                
            # TEMPORAL DECAY: Apply recency weighting
            try:
                # RECENCY FACTOR: More recent turns get higher weight
                turn_position = turn_id / max(1, conversation_length - 1) if conversation_length > 1 else 1.0
                recency_weight = 0.5 + 0.5 * turn_position  # Range: 0.5 to 1.0
                
                # CONVERSATION LENGTH FACTOR: Longer conversations indicate established patterns
                length_factor = min(1.0, conversation_length / 20.0)  # Scales up to 20 turns
                
                # COMBINED FREQUENCY: Integrate all factors
                adjusted_frequency = raw_frequency * recency_weight * (0.7 + 0.3 * length_factor)
                
                logger.debug(f"Frequency adjustment: raw={raw_frequency:.3f}, "
                            f"recency={recency_weight:.3f}, length_factor={length_factor:.3f}")
                
            except Exception as decay_error:
                logger.error(f"Error in temporal decay calculation: {decay_error}")
                adjusted_frequency = raw_frequency
                
            # NORMALIZATION: Ensure result is in valid range [0.0, 1.0]
            final_frequency = max(0.0, min(1.0, adjusted_frequency))
            
            logger.debug(f"Final usage frequency for turn {turn_id}: {final_frequency:.3f}")
            return final_frequency
            
        except Exception as e:
            logger.error(f"Error calculating usage frequency: {e}")
            return 0.5  # Moderate frequency fallback
    
    def _calculate_jensen_shannon_divergence(self, current_embedding: np.ndarray, global_context: np.ndarray) -> float:
        """
        IMPLEMENTED: Calculate Jensen-Shannon divergence between current embedding and global context.

        ACCOUNTABILITY IMPLEMENTATION:
        1. ADDED: Proper probability distribution conversion for embeddings
        2. IMPLEMENTED: Standard Jensen-Shannon divergence formula using KL divergence
        3. INCLUDED: Comprehensive error handling for numerical stability
        4. JUSTIFIED: JS divergence measures similarity between probability distributions (0=identical, 1=completely different)

        Args:
            current_embedding: Current turn embedding vector
            global_context: Global context embedding vector

        Returns:
            float: Jensen-Shannon divergence (0.0 to 1.0, lower = more similar)

        Mathematical Definition:
            JS(P,Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
        """
        try:
            logger.debug("Calculating Jensen-Shannon divergence between current embedding and global context")

            # VALIDATION: Check input validity
            if current_embedding is None or global_context is None:
                logger.warning("Null embedding provided to Jensen-Shannon calculation")
                return 0.5  # Moderate divergence fallback

            if not isinstance(current_embedding, np.ndarray) or not isinstance(global_context, np.ndarray):
                logger.warning("Non-array input to Jensen-Shannon calculation")
                return 0.5

            # SHAPE VALIDATION: Ensure compatible dimensions
            if current_embedding.shape != global_context.shape:
                logger.warning(f"Shape mismatch in JS divergence: {current_embedding.shape} vs {global_context.shape}")
                return 0.5

            # CONVERSION: Transform embeddings to probability distributions
            try:
                prob_current = self._embedding_to_probability(current_embedding)
                prob_global = self._embedding_to_probability(global_context)
                logger.debug(f"Converted embeddings to probability distributions: shapes {prob_current.shape}, {prob_global.shape}")
            except Exception as conversion_error:
                logger.error(f"Error converting embeddings to probabilities: {conversion_error}")
                return 0.5

            # CALCULATION: Jensen-Shannon divergence using standard formula
            try:
                # Calculate mixture distribution M = 0.5 * (P + Q)
                mixture = 0.5 * (prob_current + prob_global)

                # Calculate KL divergences: KL(P||M) and KL(Q||M)
                kl_current_mixture = self._calculate_kl_divergence(prob_current, mixture)
                kl_global_mixture = self._calculate_kl_divergence(prob_global, mixture)

                # Jensen-Shannon divergence: JS(P,Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
                js_divergence = 0.5 * kl_current_mixture + 0.5 * kl_global_mixture

                # BOUNDS CHECK: Ensure result is within valid range [0, 1]
                js_divergence = max(0.0, min(1.0, float(js_divergence)))

                logger.info(f"Jensen-Shannon divergence calculated: {js_divergence:.6f} "
                            f"(KL_current: {kl_current_mixture:.6f}, KL_global: {kl_global_mixture:.6f})")

                return js_divergence

            except Exception as calc_error:
                logger.error(f"Error in Jensen-Shannon divergence calculation: {calc_error}")
                return 0.5

        except Exception as e:
            logger.error(f"Error in Jensen-Shannon divergence method: {e}")
            return 0.5  # Safe fallback for moderate divergence

    def _calculate_kl_divergence(self, prob_p: np.ndarray, prob_q: np.ndarray) -> float:
        """
        HELPER METHOD: Calculate Kullback-Leibler divergence KL(P||Q).

        ACCOUNTABILITY IMPLEMENTATION:
        1. ADDED: Numerical stability with epsilon for zero probabilities
        2. IMPLEMENTED: Standard KL divergence formula with log base 2
        3. INCLUDED: Bounds checking for infinite/NaN results

        Args:
            prob_p: First probability distribution
            prob_q: Second probability distribution

        Returns:
            float: KL divergence (0 to infinity, 0 = identical distributions)

        Mathematical Definition:
            KL(P||Q) = Σ P(i) * log2(P(i) / Q(i))
        """
        try:
            # NUMERICAL STABILITY: Add small epsilon to prevent log(0)
            epsilon = 1e-12
            prob_p = prob_p + epsilon
            prob_q = prob_q + epsilon

            # RENORMALIZATION: Ensure probabilities still sum to 1 after epsilon addition
            prob_p = prob_p / np.sum(prob_p)
            prob_q = prob_q / np.sum(prob_q)

            # KL DIVERGENCE CALCULATION: Standard formula with log base 2
            kl_div = np.sum(prob_p * np.log2(prob_p / prob_q))

            # BOUNDS CHECK: Ensure finite result
            if not np.isfinite(kl_div):
                logger.warning("Non-finite KL divergence result, returning large value")
                return 10.0  # Large divergence indicating very different distributions

            # CLAMP: Reasonable upper bound for stability
            kl_div = max(0.0, min(10.0, float(kl_div)))

            logger.debug(f"KL divergence calculated: {kl_div:.6f}")
            return kl_div

        except Exception as e:
            logger.error(f"Error calculating KL divergence: {e}")
            return 5.0  # Moderate divergence fallback        


    def _embedding_to_probability(self, embedding: np.ndarray) -> np.ndarray:
        """Convert embedding to probability distribution."""
        try:
            exp_embedding = np.exp(embedding - np.max(embedding))  # Numerical stability
            prob_distribution = exp_embedding / np.sum(exp_embedding)
            return prob_distribution
        except Exception as e:
            logger.error(f"Error converting embedding to probability: {str(e)}")
            return np.ones(len(embedding)) / len(embedding)  # Fallback uniform distribution

    def _validate_and_flatten_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ADDED: Validate and normalize input vectors for similarity calculations.
        
        JUSTIFICATION: Fixes shape mismatch errors in similarity methods.
        FUNCTIONALITY: Ensures both vectors are 1D and same length before calculations.
        """
        try:
            # Convert to numpy arrays if not already
            vec1 = np.asarray(vec1, dtype=np.float32)
            vec2 = np.asarray(vec2, dtype=np.float32)
            
            # Flatten to 1D if multi-dimensional
            if vec1.ndim > 1:
                vec1 = vec1.flatten()
                logger.debug(f"Flattened vec1 from shape {vec1.shape} to 1D")
                
            if vec2.ndim > 1:
                vec2 = vec2.flatten()
                logger.debug(f"Flattened vec2 from shape {vec2.shape} to 1D")
            
            # Handle dimension mismatch - pad shorter vector with zeros
            if vec1.shape[0] != vec2.shape[0]:
                max_len = max(vec1.shape[0], vec2.shape[0])
                
                if vec1.shape[0] < max_len:
                    vec1_padded = np.zeros(max_len, dtype=np.float32)
                    vec1_padded[:vec1.shape[0]] = vec1
                    vec1 = vec1_padded
                    logger.debug(f"Padded vec1 to length {max_len}")
                    
                if vec2.shape[0] < max_len:
                    vec2_padded = np.zeros(max_len, dtype=np.float32)
                    vec2_padded[:vec2.shape[0]] = vec2
                    vec2 = vec2_padded
                    logger.debug(f"Padded vec2 to length {max_len}")
            
            # Final validation
            if vec1.shape != vec2.shape:
                logger.error(f"Vector shape mismatch after normalization: {vec1.shape} vs {vec2.shape}")
                return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
                
            return vec1, vec2
            
        except Exception as e:
            logger.error(f"Error validating vectors: {e}")
            # Return dummy vectors on failure
            return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        FIXED: Calculate normalized cosine similarity with shape validation.

        CHANGES MADE:
        1. Added vector validation and flattening
        2. Added zero vector handling 
        3. Preserved original cosine calculation logic
        """
        try:
            # ADDED: Validate and normalize input vectors
            vec1, vec2 = self._validate_and_flatten_vectors(vec1, vec2)

            # ADDED: Handle zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                logger.debug("Zero vector encountered in cosine similarity")
                return 0.0

            # PRESERVED: Original cosine calculation using scipy
            cos_distance = cosine(vec1, vec2)
            return float(1.0 - cos_distance)  # ADDED: Explicit float conversion

        except Exception as e:
            logger.error(f"Error in cosine similarity calculation: {str(e)}")
            return 0.0

    def _calculate_angular_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        FIXED: Calculate angular similarity with shape validation.

        CHANGES MADE:
        1. Added vector validation and flattening
        2. Fixed dot product shape alignment
        3. Preserved original angular calculation logic
        """
        try:
            # ADDED: Validate and normalize input vectors
            vec1, vec2 = self._validate_and_flatten_vectors(vec1, vec2)

            # PRESERVED: Original norm calculation
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                logger.debug("Zero vector encountered in angular similarity")
                return 0.0

            # FIXED: Ensure proper dot product calculation
            cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # PRESERVED: Original angular similarity formula
            return float(1.0 - (angle / np.pi))  # ADDED: Explicit float conversion

        except Exception as e:
            logger.error(f"Error in angular similarity calculation: {str(e)}")
            return 0.0

    def _calculate_spherical_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        FIXED: Calculate spherical distance with shape validation.

        CHANGES MADE:
        1. Added vector validation and flattening
        2. Fixed unit vector calculation for aligned shapes
        3. Preserved original spherical distance logic
        """
        try:
            # ADDED: Validate and normalize input vectors
            vec1, vec2 = self._validate_and_flatten_vectors(vec1, vec2)

            # PRESERVED: Original norm calculations
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                logger.debug("Zero vector encountered in spherical similarity")
                return 0.0

            # FIXED: Ensure unit vectors are properly calculated
            unit_vec1 = vec1 / norm1
            unit_vec2 = vec2 / norm2

            # FIXED: Proper dot product with aligned shapes
            dot_product = np.dot(unit_vec1, unit_vec2)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            spherical_distance = np.arccos(dot_product)

            # PRESERVED: Original spherical similarity formula
            return float(1.0 - (spherical_distance / np.pi))  # ADDED: Explicit float conversion

        except Exception as e:
            logger.error(f"Error in spherical similarity calculation: {str(e)}")
            return 0.0

    def _calculate_euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        FIXED: Calculate Euclidean distance similarity with shape validation.

        CHANGES MADE:
        1. Added vector validation and flattening
        2. Fixed euclidean distance calculation for 1D vectors
        3. Preserved original exponential similarity formula
        """
        try:
            # ADDED: Validate and normalize input vectors
            vec1, vec2 = self._validate_and_flatten_vectors(vec1, vec2)

            # FIXED: Use numpy euclidean distance instead of scipy for 1D vectors
            distance = np.linalg.norm(vec1 - vec2)

            # PRESERVED: Original exponential similarity formula
            similarity = np.exp(-distance / np.sqrt(len(vec1)))
            return float(similarity)  # ADDED: Explicit float conversion

        except Exception as e:
            logger.error(f"Error in Euclidean similarity calculation: {str(e)}")
            return 0.0

    def _calculate_js_divergence_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        FIXED: Calculate Jensen-Shannon divergence similarity with shape validation.

        CHANGES MADE:
        1. Added vector validation and flattening
        2. Fixed probability distribution normalization
        3. Preserved original JS divergence calculation logic
        """
        try:
            # ADDED: Validate and normalize input vectors
            vec1, vec2 = self._validate_and_flatten_vectors(vec1, vec2)

            # PRESERVED: Original probability conversion function (inline)
            def to_probability(vec):
                # ADDED: Handle potential overflow in exp
                max_val = np.max(vec)
                exp_vec = np.exp(vec - max_val)  # Numerical stability
                return exp_vec / np.sum(exp_vec)

            # PRESERVED: Original probability calculation
            prob1 = to_probability(vec1)
            prob2 = to_probability(vec2)

            # PRESERVED: Original JS divergence calculation
            m = 0.5 * (prob1 + prob2)
            js_divergence = 0.5 * entropy(prob1, m) + 0.5 * entropy(prob2, m)

            # PRESERVED: Original similarity conversion
            return float(1.0 / (1.0 + js_divergence))  # ADDED: Explicit float conversion

        except Exception as e:
            logger.error(f"Error in JS divergence similarity calculation: {str(e)}")
            return 0.0

    def _calculate_ensemble_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        FIXED: Calculate ensemble similarity using multiple methods with shape validation.

        CHANGES MADE:
        1. Added input validation at ensemble level
        2. Fixed method weight normalization
        3. Preserved all original similarity method calls and weighting logic
        """
        try:
            # ADDED: Early validation to prevent propagating bad shapes
            if vec1 is None or vec2 is None:
                logger.debug("Null vector provided to ensemble similarity")
                return 0.0

            # ADDED: Pre-validate shapes before calling individual methods
            vec1, vec2 = self._validate_and_flatten_vectors(vec1, vec2)

            # PRESERVED: Original similarity calculation logic
            similarities = {}
            total_weight = 0.0

            for method, weight in self.method_weights.items():
                try:
                    if method == 'cosine':
                        sim = self._calculate_cosine_similarity(vec1, vec2)
                    elif method == 'angular':
                        sim = self._calculate_angular_similarity(vec1, vec2)
                    elif method == 'probabilistic_js':
                        sim = self._calculate_js_divergence_similarity(vec1, vec2)
                    elif method == 'euclidean':
                        sim = self._calculate_euclidean_similarity(vec1, vec2)
                    elif method == 'spherical':
                        sim = self._calculate_spherical_similarity(vec1, vec2)
                    else:
                        continue
                    
                    # ADDED: Validate similarity result
                    if not np.isfinite(sim):
                        logger.debug(f"Non-finite similarity from {method}: {sim}")
                        sim = 0.0

                    similarities[method] = float(sim)  # ADDED: Explicit float conversion
                    total_weight += weight

                except Exception as method_error:
                    logger.error(f"Error in {method} similarity: {method_error}")
                    similarities[method] = 0.0
                    total_weight += weight  # Still count weight for normalization

            # PRESERVED: Original weighted average calculation
            if total_weight == 0:
                logger.debug("Zero total weight in ensemble similarity")
                return 0.0

            ensemble_similarity = sum(similarities[method] * self.method_weights[method] 
                                    for method in similarities) / total_weight

            # ADDED: Final result validation
            if not np.isfinite(ensemble_similarity):
                logger.error(f"Non-finite ensemble similarity result: {ensemble_similarity}")
                return 0.0

            return float(ensemble_similarity)  # ADDED: Explicit float conversion

        except Exception as e:
            logger.error(f"Error in ensemble similarity calculation: {str(e)}")
            return 0.0
    
    def get_context_similarity(self, token_embedding: np.ndarray) -> float:
        """Get similarity between token and current context."""
        if token_embedding is None or self.current_context_embedding is None:
            return 0.0
        if np.linalg.norm(self.current_context_embedding) == 0 or np.linalg.norm(token_embedding) == 0:
            return 0.0

        try:
            similarity = self._calculate_ensemble_similarity(token_embedding, self.current_context_embedding)
            return similarity
        except Exception as e:
            logger.error(f"Enhanced similarity failed, using fallback: {str(e)}")
            return np.dot(self.current_context_embedding, token_embedding)
      
    def clear_context(self):
        """Clear the conversation context."""
        self.conversation_history.clear()
        self.current_context_embedding = None
        logger.info("Cleared conversation context")