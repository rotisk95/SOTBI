# Configure logging for execution transparency
from collections import deque
from dataclasses import dataclass
import logging
import math
import time
from typing import List, Optional

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
        """Add conversation turn with adaptive weighting system."""
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
    
                # FIXED: Adaptive weighting system with proper method call
                try:
                    # Extract features for adaptive weighting
                    features = self._extract_adaptive_features(
                        current_embedding, 
                        previous_global_context, 
                        turn['turn_id']
                    )
                    logger.info(f"Extracted adaptive features: entropy={features[0]:.3f}, js_divergence={features[1]:.3f}, usage_freq={features[2]:.3f}")
    
                    # FIXED: Calculate adaptive weights using the new method
                    adaptive_weights = self._calculate_adaptive_weights(features)
                    w1, w2, w3 = adaptive_weights  # current, attention_weighted, global
                    logger.info(f"Adaptive weights: current={w1:.3f}, attention={w2:.3f}, global={w3:.3f}")
    
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
                    logger.info(f"Applied adaptive weighting: {w1:.3f}*current + {w2:.3f}*attention + {w3:.3f}*global")
                    
                except Exception as e:
                    logger.error(f"Error in adaptive weighting calculation: {str(e)}")
                    # Fallback to static blending
                    global_blend_factor = 0.3
                    blended_context = (
                        (1 - global_blend_factor) * attention_weighted_context + 
                        global_blend_factor * previous_global_context
                    )
                    norm = np.linalg.norm(blended_context)
                    if norm > 0:
                        blended_context = blended_context / norm
                    self.current_context_embedding = blended_context
                    logger.warning("Fallback: Used static blending due to adaptive weighting error")
    
            except Exception as e:
                logger.error(f"Error in attention-weighted context calculation: {str(e)}")
                # Fallback to exponential moving average
                decay_factor = 0.7
                self.current_context_embedding = (decay_factor * previous_global_context + (1 - decay_factor) * embedding)
                logger.warning("Fallback: Used exponential moving average due to attention calculation error")
    
        logger.info(f"Successfully added conversation turn with {len(tokens)} tokens to context window")

    def _calculate_adaptive_weights(self, features: List[float]) -> List[float]:
        """
        ADDED: Calculate adaptive weights using improved balanced approach.
        FIXES: The original missing method that was causing the error.
        """
        try:
            entropy, js_divergence, usage_freq = features
            logger.debug(f"Calculating adaptive weights for features: entropy={entropy:.3f}, js_div={js_divergence:.6f}, usage={usage_freq:.3f}")
            
            # BALANCED APPROACH: Ensure minimum weights for all components
            min_current = 0.5      # Minimum 50% for current
            min_attention = 0.2    # Minimum 20% for attention
            min_global = 0.1       # Minimum 10% for global
            
            # Calculate base weights with feature influence
            entropy_factor = max(0.0, min(1.0, (10.0 - entropy) / 5.0))  # Normalize entropy (lower = better)
            js_factor = min(1.0, js_divergence * 1000)  # Amplify JS divergence
            usage_factor = usage_freq
            
            # Calculate raw weights
            raw_current = min_current + ((1.0 - entropy_factor) * 0.2)  # High entropy = more current focus
            raw_attention = min_attention + (entropy_factor * js_factor * 0.2)  # Good for complex, diverse contexts
            raw_global = min_global + (usage_factor * 0.2)  # More usage = more global context
            
            # Normalize to sum to 1.0
            total = raw_current + raw_attention + raw_global
            current_weight = raw_current / total
            attention_weight = raw_attention / total
            global_weight = raw_global / total
            
            weights = [current_weight, attention_weight, global_weight]
            logger.info(f"Calculated adaptive weights: current={current_weight:.3f}, attention={attention_weight:.3f}, global={global_weight:.3f}")
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating adaptive weights: {e}")
            # Safe fallback weights
            return [0.6, 0.3, 0.1]

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

    def _calculate_jensen_shannon_divergence(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between embeddings."""
        try:
            prob1 = self._embedding_to_probability(embedding1)
            prob2 = self._embedding_to_probability(embedding2)

            m = 0.5 * (prob1 + prob2)
            js_divergence = 0.5 * entropy(prob1, m) + 0.5 * entropy(prob2, m)
            return js_divergence
        except Exception as e:
            logger.error(f"Error in Jensen-Shannon divergence calculation: {str(e)}")
            return 0.1  # Fallback moderate divergence

    def _calculate_usage_frequency(self, turn_id: int) -> float:
        """Calculate usage frequency for long-term salience tracking."""
        try:
            conversation_length = len(self.conversation_history)
            if conversation_length <= 1:
                frequency = 1.0
            else:
                frequency = min(1.0, math.log(conversation_length + 1) / math.log(10))
            return frequency
        except Exception as e:
            logger.error(f"Error calculating usage frequency: {str(e)}")
            return 0.5  # Fallback moderate frequency

    def _calculate_ensemble_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate ensemble similarity using multiple methods."""
        try:
            similarities = {}
            total_weight = 0.0
            
            for method, weight in self.method_weights.items():
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
                
                similarities[method] = sim
                total_weight += weight
            
            if total_weight == 0:
                return 0.0
                
            ensemble_similarity = sum(similarities[method] * self.method_weights[method] 
                                    for method in similarities) / total_weight
            
            return ensemble_similarity
            
        except Exception as e:
            logger.error(f"Error in ensemble similarity calculation: {str(e)}")
            return 0.0

    # All your other similarity methods remain unchanged...
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate normalized cosine similarity."""
        try:
            cos_distance = cosine(vec1, vec2)
            return 1.0 - cos_distance
        except Exception as e:
            logger.error(f"Error in cosine similarity calculation: {str(e)}")
            return 0.0
    
    def _calculate_angular_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate angular similarity."""
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return 1.0 - (angle / np.pi)
            
        except Exception as e:
            logger.error(f"Error in angular similarity calculation: {str(e)}")
            return 0.0
    
    def _calculate_spherical_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate spherical distance."""
        try:
            unit_vec1 = vec1 / np.linalg.norm(vec1)
            unit_vec2 = vec2 / np.linalg.norm(vec2)
            
            dot_product = np.clip(np.dot(unit_vec1, unit_vec2), -1.0, 1.0)
            spherical_distance = np.arccos(dot_product)
            
            return 1.0 - (spherical_distance / np.pi)
            
        except Exception as e:
            logger.error(f"Error in spherical similarity calculation: {str(e)}")
            return 0.0
    
    def _calculate_js_divergence_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence similarity."""
        try:
            def to_probability(vec):
                exp_vec = np.exp(vec - np.max(vec))
                return exp_vec / np.sum(exp_vec)
            
            prob1 = to_probability(vec1)
            prob2 = to_probability(vec2)
            
            m = 0.5 * (prob1 + prob2)
            js_divergence = 0.5 * entropy(prob1, m) + 0.5 * entropy(prob2, m)
            
            return 1.0 / (1.0 + js_divergence)
            
        except Exception as e:
            logger.error(f"Error in JS divergence similarity calculation: {str(e)}")
            return 0.0
    
    def _calculate_euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance similarity."""
        try:
            distance = euclidean(vec1, vec2)
            return np.exp(-distance / np.sqrt(len(vec1)))
        except Exception as e:
            logger.error(f"Error in Euclidean similarity calculation: {str(e)}")
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