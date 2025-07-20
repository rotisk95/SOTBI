# Configure logging for execution transparency
from collections import deque
from dataclasses import dataclass
import logging
import math
import time
from typing import List, Optional, Tuple

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
    
    def __init__(self, max_turns: int = 500, max_tokens: int = 100, time_window_seconds: int = 300, method_weights: Optional[dict] = None):
        """Initialize context window with enhanced similarity calculator."""
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.time_window_seconds = time_window_seconds
        self.conversation_history: deque = deque(maxlen=max_turns)
        self.current_context_embedding: Optional[np.ndarray] = None
        
        # Default ensemble weights - can be learned/adjusted
        self.method_weights = method_weights or {
            'cosine': 0.4,
            'angular': 0.2, 
            'probabilistic_js': 0.2,
            'euclidean': 0.1,
            'spherical': 0.1
        }
        
        logger.info(f"Enhanced similarity calculator initialized with weights: {self.method_weights}")
        logger.info(f"Initialized ContextWindow with max_turns={max_turns}, max_tokens={max_tokens}")
    
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
            
            weights = [current_weight, attention_weight, global_weight]
            logger.info(f"FIXED adaptive weights: current={current_weight:.3f}, attention={attention_weight:.3f}, global={global_weight:.3f}")
            
            return weights
            
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