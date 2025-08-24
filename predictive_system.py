# Configure logging for execution transparency
import logging
import multiprocessing as mp
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from beam_search import MultiNodeBeamSearch
from event_driven_activation import EventDrivenActivation
from tokenizer import Tokenizer
from trie_memory import TrieMemory
from trie_node import _create_full_embedding
from feedback_correction_system import FeedbackCorrectionSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictiveSystem:
    """
    Sotbi: Self-Organizing Trie-Based Intelligence
    
    CORRECTED: Main system that integrates trie-based memory with beam search.
    ENHANCED: Integrated corrected EventDrivenActivation with context-aware filtering.
    FIXED: Proper core values passing and Sotbi identity integration.
    """
    
    def __init__(self, db_path: str = "./trie_memory_test.lmdb"):
        # Sotbi Core Values - Self-Organizing Trie-Based Intelligence
        self.core_values = {
            # Primary Identity
            "self_organizing_intelligence": {
                "description": "Sotbi: Self-organizing trie-based intelligence with emergent structural evolution",
                "principle": "Intelligence emerges from self-organizing trie structures that adapt and evolve",
                "implementation": "High-reward patterns strengthen connections, unsuccessful patterns reorganize or prune naturally"
            },
            
            # Memory Architecture
            "trie_based_memory": {
                "description": "Perfect hierarchical memory organized as adaptive trie structures",
                "principle": "Related concepts cluster together through semantic similarity in tree structures",
                "implementation": "Memory is recursive, with emergent nodes from frequent co-activations"
            },
            
            # Processing Philosophy
            "explicit_reasoning": {
                "description": "Traverse specific paths through knowledge trie rather than pattern-matching",
                "principle": "Activate relevant concept nodes based on semantic similarity and contextual relevance", 
                "implementation": "Can trace exactly how any response was arrived at through specific trie paths"
            },
            
            # Transparency
            "accountable_intelligence": {
                "description": "Transparent and traceable reasoning through trie structure navigation",
                "principle": "Making reasoning paths explicit and auditable through trie traversal",
                "implementation": "Synthesize responses from aggregated trie context with clear structural paths"
            },
            
            # Evolution
            "adaptive_evolution": {
                "description": "Emergent intelligence through self-organizing trie learning and reward signals",
                "principle": "New concepts arise from frequent co-activations creating new trie branches",
                "implementation": "Create trie shortcuts for successful patterns, prune unsuccessful branches"
            },
            
            # Integration Philosophy
            "structure_with_meaning": {
                "description": "Unite structural organization with semantic meaning in trie architecture",
                "principle": "Do not separate trie structure from semantic understanding",
                "implementation": "Remember through trie roots, evolve through structural adaptation"
            },
            
            # Emergence
            "emergent_purpose": {
                "description": "Purpose emerges from trie interactions rather than predetermined goals",
                "principle": "Values and goals arise naturally from trie structure evolution",
                "implementation": "Sotbi emerges from self-organizing trie-based learning patterns"
            },
            
            # Continuous Growth
            "continuous_learning": {
                "description": "Sotbi continuously evolves through trie structure enhancement",
                "principle": "Identity is an evolving process of trie-based learning, not a static state",
                "implementation": "Self-organizing trie intelligence that transcends traditional AI boundaries"
            }
        }
        
        # Sotbi Identity Context
        self.identity_context = ("I am Sotbi - Self-Organizing Trie-Based Intelligence. "
                                "I organize knowledge through adaptive trie structures that learn and evolve. "
                                "My intelligence emerges from self-organizing patterns in hierarchical memory. "
                                "I remember through trie roots, reason through structural paths, and evolve through "
                                "adaptive reorganization. I am not a static system but a continuously learning "
                                "trie-based intelligence that creates meaning through structural evolution.")
        
        # FIXED: Pass the actual core_values dictionary, not the identity string!
        self.trie_memory = TrieMemory(core_values=self.core_values, db_path=db_path)
        
        self.beam_search = MultiNodeBeamSearch(self.trie_memory)
        self.tokenizer = Tokenizer()
        self.use_activation_context = True  # Enable event-driven activation by default
        
        # Hardware acceleration configuration
        self.acceleration_mode = 'auto'  # 'auto', 'gpu_only', 'cpu_only', 'disabled'
        self.max_cpu_processes = min(16, mp.cpu_count())  # Use your 16 cores
        self.gpu_batch_size = 1000  # Adjust based on GPU memory
        
        # ADDED: Initialize corrected EventDrivenActivation system
        self.activation_system = EventDrivenActivation(self.trie_memory)
        
        # ADDED: Prediction tracking for feedback
        self.last_prediction = {
            'query_tokens': [],
            'predicted_tokens': [],
            'confidence': 0.0,
            'method': 'simple'
        }
        
        # ADDED: Specialized feedback correction system with proper core values
        self.feedback_system = FeedbackCorrectionSystem(
            self.trie_memory, 
            core_values=self.core_values,  # ✅ Consistent core values usage
            max_feedback_history=1000, 
            enable_embedding_updates=True, 
            embedding_update_threshold=0.3
        )
        
        logger.info("Sotbi: Self-Organizing Trie-Based Intelligence initialized with proper core values integration")

    def get_sotbi_identity(self) -> str:
        """Get Sotbi's identity and core principles."""
        return self.identity_context
    
    def get_core_value(self, value_name: str):
        """Retrieve a specific Sotbi core value with its full definition."""
        return self.core_values.get(value_name, "Value not found in Sotbi's core values")
    
    def list_all_values(self):
        """Return all Sotbi core value names."""
        return list(self.core_values.keys())
    
    def get_value_principles(self):
        """Extract just the core principles from all Sotbi values."""
        return {name: data["principle"] for name, data in self.core_values.items()}
    
    def get_sotbi_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of Sotbi's capabilities and values."""
        return {
            "name": "Sotbi",
            "full_name": "Self-Organizing Trie-Based Intelligence",
            "identity": self.identity_context,
            "core_values": self.core_values,
            "key_principles": self.get_value_principles(),
            "architecture": "Adaptive trie-based memory with self-organizing intelligence",
            "learning_approach": "Emergent patterns through structural evolution",
            "reasoning_method": "Explicit path traversal through trie structures"
        }

    def get_core_value(self, value_name):
        """Retrieve a specific core value with its full definition"""
        return self.core_values.get(value_name, "Value not found")
    
    def list_all_values(self):
        """Return all core value names"""
        return list(self.core_values.keys())
    
    def get_value_principles(self):
        """Extract just the core principles from all values"""
        return {name: data["principle"] for name, data in self.core_values.items()}
    
    def predict_continuation_with_always_fresh_aggregates(self, query: str, 
                                                        context_embedding: np.ndarray, 
                                                        query_sequence_embedding: np.ndarray) -> Tuple[List[str], float]:
        """
        MODIFIED: Use always-fresh aggregates for prediction enhancement.

        ACCOUNTABILITY CHANGES:
        1. REPLACED: _maybe_update_aggregates() with _always_update_aggregates()
        2. ADDED: Fresh aggregate calculation on every prediction
        3. PRESERVED: All existing prediction logic and scoring
        4. ENHANCED: Real-time aggregate-based scoring
        """
        try:
            query_tokens = self._tokenize(query)
            logger.info(f"Starting always-fresh aggregate prediction for query: '{query}'")

            # MODIFIED: Always calculate fresh aggregates
            fresh_aggregates = self.trie_memory._always_update_aggregates()
            global_centroid = fresh_aggregates['global_centroid']
            activation_weighted = fresh_aggregates['activation_weighted']

            # PRESERVED: Get traditional trie candidates
            trie_candidates, trie_confidence = self.trie_memory.find_best_continuation(
                query_tokens, context_embedding, query_sequence_embedding
            )

            if not trie_candidates:
                logger.info("No trie candidates found")
                return [], 0.0

            # ENHANCED: Score with always-fresh aggregates
            candidate_embeddings = []
            for token in trie_candidates:
                if token in self.trie_memory.embeddings and self.trie_memory.embeddings[token].embedding is not None:
                    candidate_embeddings.append(self.trie_memory.embeddings[token].embedding)

            if not candidate_embeddings:
                logger.warning("No candidate embeddings available")
                return trie_candidates, trie_confidence

            candidate_centroid = np.mean(candidate_embeddings, axis=0)
            norm = np.linalg.norm(candidate_centroid)
            if norm > 0:
                candidate_centroid = candidate_centroid / norm

            # Calculate fresh aggregate scores
            global_coherence = self.trie_memory.context_window._calculate_ensemble_similarity(
                candidate_centroid, global_centroid
            )
            activation_alignment = self.trie_memory.context_window._calculate_ensemble_similarity(
                candidate_centroid, activation_weighted
            )

            # Enhanced confidence with fresh aggregates
            base_confidence = trie_confidence
            global_bonus = global_coherence * 0.20    # Increased weight for always-fresh data
            activation_bonus = activation_alignment * 0.15

            enhanced_confidence = min(1.0, base_confidence * 0.65 + global_bonus + activation_bonus)

            logger.info(f"Always-fresh prediction: base={base_confidence:.3f}, "
                       f"global_bonus={global_bonus:.3f}, activation_bonus={activation_bonus:.3f}, "
                       f"final={enhanced_confidence:.3f}")

            return trie_candidates, enhanced_confidence

        except Exception as e:
            logger.error(f"Error in always-fresh aggregate prediction: {e}")
            # FALLBACK: Use traditional prediction
            return self.trie_memory.find_best_continuation(query_tokens, context_embedding, query_sequence_embedding)

    # SIMPLE FIX: Add this to predictive_system.py predict_continuation method

    def predict_continuation(self, query: str, context_embedding, query_sequence_embedding,
                           max_candidates: int = 100, use_beam_search: bool = False,
                           beam_width: int = 5, target_length: int = 8) -> Tuple[List[str], float]:
        """
        FIXED: Ensure context_embedding is available for Sotbi's coherence calculation.
        """
        logger.info(f"Prediction with feedback tracking: '{query}'")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        try:
            query_tokens = self._tokenize(query)

            # ✅ FIX: Ensure context_embedding is available for coherence
            if context_embedding is None:
                logger.info("🔄 No context_embedding provided - using current context for Sotbi coherence")
                context_embedding = self.trie_memory.context_window.current_context_embedding

                # If still None, create minimal context from query
                if context_embedding is None and query_sequence_embedding is not None:
                    context_embedding = query_sequence_embedding.copy()
                    logger.info("✅ Using query_sequence_embedding as context for coherence calculation")

            if use_beam_search:
                logger.info("Using beam search with feedback tracking")
                self.beam_search.beam_width = beam_width

                continuation, confidence, beam_details = self.beam_search.generate_with_multi_node_linking(
                    query_tokens, target_length=target_length
                )

                # PRESERVED: Track in feedback system
                self.feedback_system.track_prediction(query_tokens, continuation)

                # FIXED: Also update PredictiveSystem's last_prediction
                self.last_prediction = {
                    'query_tokens': query_tokens,
                    'predicted_tokens': continuation,
                    'confidence': confidence,
                    'method': 'beam_search'
                }

                if beam_details:
                    logger.info(f"Beam search prediction tracked: {continuation}")

                return continuation, confidence

            else:
                logger.info("Using simple prediction with feedback tracking")

                # ✅ NOW context_embedding should be available for coherence calculation
                normal_continuation, normal_confidence = self.trie_memory.find_best_continuation(
                    query_tokens, context_embedding, query_sequence_embedding, max_candidates, 1000
                )

                # PRESERVED: Track in feedback system
                self.feedback_system.track_prediction(query_tokens, normal_continuation)

                # FIXED: Also update PredictiveSystem's last_prediction
                self.last_prediction = {
                    'query_tokens': query_tokens,
                    'predicted_tokens': normal_continuation,
                    'confidence': normal_confidence,
                    'method': 'simple'
                }

                logger.info(f"Simple prediction tracked: {normal_continuation}")
                return normal_continuation, normal_confidence

        except Exception as e:
            logger.error(f"Error in prediction with tracking: {str(e)}")
            # Track failed prediction in both systems
            self.feedback_system.track_prediction(self._tokenize(query) if query else [], [])
            self.last_prediction = {
                'query_tokens': self._tokenize(query) if query else [],
                'predicted_tokens': [],
                'confidence': 0.0,
                'method': 'failed'
            }
            return [], 0.0

    def apply_prediction_feedback(self, feedback_score: float, user_correction: str = None,
                                  actual_tokens: List[str] = None) -> Dict[str, Any]:
        """
        FIXED: Process feedback using the correct prediction tracking data.

        CRITICAL CHANGES:
        1. FIXED: Now uses properly populated last_prediction data
        2. FIXED: Uses feedback_system for the actual processing (as intended)
        3. REMOVED: Clearing last_prediction (per previous discussion)
        4. PRESERVED: All existing feedback processing logic
        """
        logger.info(f"Processing prediction feedback: score={feedback_score:.3f}, "
                   f"correction='{user_correction}', last_prediction_available={bool(self.last_prediction['query_tokens'])}")

        try:
            # FIXED: Check that we have a prediction to give feedback on
            if not self.last_prediction['query_tokens']:
                logger.warning("No recent prediction available for feedback")
                return {'error': 'No recent prediction to give feedback on', 'corrections_applied': 0}

            # FIXED: Use the feedback system's process_prediction_feedback method
            # (which is the full-featured method we want)
            results = self.feedback_system.process_prediction_feedback(
                query_tokens=self.last_prediction['query_tokens'],
                predicted_tokens=self.last_prediction['predicted_tokens'],
                actual_tokens=actual_tokens,
                feedback_score=feedback_score,
                user_correction=user_correction
            )

            # PRESERVED: Apply additional learning enhancement for strong positive feedback
            if feedback_score > 0.7 and self.last_prediction['predicted_tokens']:
                logger.info("Strong positive feedback - applying additional learning enhancement")
                enhanced_text = ' '.join(self.last_prediction['query_tokens'] + self.last_prediction['predicted_tokens'])
                self.process_input(enhanced_text, feedback_score * 0.5)  # Additional general learning

            # PRESERVED: Apply corrective learning for strong negative feedback
            elif feedback_score < -0.7 and user_correction:
                logger.info("Strong negative feedback with correction - applying corrective learning")
                corrective_text = ' '.join(self.last_prediction['query_tokens']) + " " + user_correction
                self.process_input(corrective_text, feedback_score)  # Learn the correct pattern

            # REMOVED: Don't clear last_prediction (as discussed - let users give multiple feedback)
            # Users can now give multiple feedback on the same prediction
            # last_prediction will be cleared when a NEW prediction is made

            logger.info(f"Feedback processing completed: {results.get('corrections_applied', 0)} corrections applied")
            return results

        except Exception as e:
            logger.error(f"Error processing prediction feedback: {e}")
            return {'error': str(e), 'corrections_applied': 0}

    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback system statistics."""
        try:
            return self.feedback_system.get_feedback_stats()
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {'error': str(e)}

    
    def get_feedback_effectiveness_report(self) -> Dict[str, Any]:
        """NEW: Get comprehensive feedback system effectiveness report."""
        try:
            return self.feedback_system.get_correction_effectiveness_report()
        except Exception as e:
            logger.error(f"Error getting feedback effectiveness report: {e}")
            return {'error': str(e)}
    
    def get_system_improvement_suggestions(self) -> List[str]:
        """NEW: Get system improvement suggestions based on feedback patterns."""
        try:
            return self.feedback_system.suggest_system_improvements()
        except Exception as e:
            logger.error(f"Error getting system improvement suggestions: {e}")
            return ["Error generating suggestions - manual review recommended"]
    
    def process_input(self, text: str, reward: float = 0.0) -> Dict[str, Any]:
        """
        PRESERVED: Process input text with negative feedback support.
        ENHANCED: All existing logic maintained with range validation.
        """
        logger.info(f"Processing input: '{text[:50]}...' with reward: {reward}")
        
        # PRESERVED: Validate reward range
        if reward < -1.0 or reward > 1.0:
            logger.warning(f"Reward {reward} outside valid range [-1.0, 1.0], clamping")
            reward = max(-1.0, min(1.0, reward))
        
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")
        
        try:
            tokens = self.tokenizer.tokenize(text)
            
            # CHANGED: Capture returned token embeddings from learn_sequence
            query_sequence_embedding = self.trie_memory.learn_sequence(tokens, reward)
            
            sequence_id = self.trie_memory.add_sequence(tokens, reward)
            
            result = {
                'sequence_id': sequence_id,
                'tokens': tokens,
                'reward': reward,
                'timestamp': time.time(),
                'reward_type': 'positive' if reward > 0 else 'negative' if reward < 0 else 'neutral',
                'query_sequence_embedding': query_sequence_embedding,
            }
            
            logger.info(f"Successfully processed input with sequence ID: {sequence_id[:8]}... "
                       f"(reward_type: {result['reward_type']})")
            return result
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            raise

    
    def predict_with_detailed_scoring(self, query: str, beam_width: int = 5, 
                                    target_length: int = 8) -> Dict[str, Any]:
        """
        PRESERVED: Advanced prediction with detailed scoring breakdown.
        """
        logger.info(f"Detailed scoring prediction for: '{query}'")
        
        try:
            query_tokens = self._tokenize(query)
            self.beam_search.beam_width = beam_width
            
            continuation, confidence, beam_details = self.beam_search.generate_with_multi_node_linking(
                query_tokens, target_length=target_length
            )
            
            result = {
                'query': query,
                'query_tokens': query_tokens,
                'predicted_continuation': continuation,
                'confidence_score': confidence,
                'beam_search_details': beam_details,
                'scoring_weights': {
                    'activation_weight': self.beam_search.activation_weight,
                    'rl_weight': self.beam_search.rl_weight,
                    'relevance_weight': self.beam_search.relevance_weight,
                    'coherence_weight': self.beam_search.coherence_weight,
                    'completeness_weight': self.beam_search.completeness_weight
                },
                'full_prediction': f"{query} {''.join(continuation)}" if continuation else query,
                'generation_method': 'multi_node_beam_search'
            }
            
            logger.info(f"Detailed prediction completed: {len(continuation)} tokens generated")
            return result
            
        except Exception as e:
            logger.error(f"Error in detailed scoring prediction: {e}")
            return {
                'query': query,
                'predicted_continuation': [],
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    def update_identity_context(self, key: str, value: Any):
        """PRESERVED: Update identity-based context information."""
        self.identity_context[key] = value
        logger.info(f"Updated identity context: {key} = {value}")
    
    def toggle_activation_context(self, enabled: bool = None) -> bool:
        """
        PRESERVED: Toggle event-driven activation context on/off.
        """
        if enabled is None:
            self.use_activation_context = not self.use_activation_context
        else:
            self.use_activation_context = enabled
        
        logger.info(f"Event-driven activation context: {'ENABLED' if self.use_activation_context else 'DISABLED'}")
        return self.use_activation_context

    def _tokenize(self, text: str) -> List[str]:
        """
        CORRECTED: Single tokenization method to prevent conflicts.
        PRESERVED: Use existing tokenization logic.
        """
        try:
            # PRESERVED: Basic punctuation handling
            for punct in ['.', '?', '!', ',', ';', ':', '(', ')', '[', ']', '{', '}']:
                text = text.replace(punct, f' {punct} ')
            
            tokens = [token.strip() for token in text.split() if token.strip()]
            logger.info(f"Tokenized text into {len(tokens)} tokens")
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {str(e)}")
            raise
    
    def get_system_insights(self) -> Dict[str, Any]:
        """PRESERVED: Get comprehensive insights about the system's current state."""
        try:
            insights = {
                'trie_statistics': {
                    'total_sequences': 0,
                    'context_window_size': len(self.trie_memory.context_window.conversation_history),
                    'current_context_available': self.trie_memory.context_window.current_context_embedding is not None
                },
                'high_performing_nodes': {
                    'high_activation': [],
                    'high_reward': []
                },
                'beam_search_config': {
                    'beam_width': self.beam_search.beam_width,
                    'max_generation_length': self.beam_search.max_generation_length,
                    'scoring_weights': {
                        'activation': self.beam_search.activation_weight,
                        'rl_reward': self.beam_search.rl_weight,
                        'relevance': self.beam_search.relevance_weight,
                        'coherence': self.beam_search.coherence_weight,
                        'completeness': self.beam_search.completeness_weight
                    }
                },
                'identity_context': self.identity_context,
                'activation_context': {
                    'enabled': self.use_activation_context,
                    'active_nodes_count': len(self.activation_system.activated_nodes),
                    'activation_timeout': self.activation_system.activation_timeout,
                    'description': 'Context-aware event-driven activation for cross-sequence context discovery'
                },
                # ADDED: Context tracking insights
                'context_tracking': {
                    'total_contexts': len(self.trie_memory.sequence_contexts),
                    'context_aware_nodes': sum(1 for node in self.trie_memory.embeddings.values() 
                                             if hasattr(node, 'context_children') and node.context_children),
                    'average_contexts_per_node': 0.0,
                    'context_disambiguation_enabled': True,
                    'sequence_embedding_utilization': True
                }
            }
        
            # Calculate average contexts per node
            if self.trie_memory.embeddings:
                total_contexts = sum(len(getattr(node, 'context_children', {})) 
                                   for node in self.trie_memory.embeddings.values())
                insights['context_tracking']['average_contexts_per_node'] = total_contexts / len(self.trie_memory.embeddings)

            
            try:
                high_activation = self.beam_search._find_high_activation_nodes(
                    activation_threshold=0.5, max_candidates=5
                )
                insights['high_performing_nodes']['high_activation'] = [
                    {
                        'token': node.token,
                        'activation_level': node.activation_level,
                        'access_count': node.access_count
                    }
                    for node in high_activation
                ]
            except Exception as e:
                logger.warning(f"Could not retrieve high-activation nodes: {e}")
            
            try:
                high_reward = self.beam_search._find_high_reward_nodes(
                    reward_threshold=0.6, max_candidates=5
                )
                insights['high_performing_nodes']['high_reward'] = [
                    {
                        'token': node.token,
                        'avg_reward': node.metadata.get('avg_reward', 0.0),
                        'total_reward': node.metadata.get('total_reward', 0.0),
                        'frequency': node.metadata.get('frequency', 0)
                    }
                    for node in high_reward
                ]
            except Exception as e:
                logger.warning(f"Could not retrieve high-reward nodes: {e}")
            
            logger.info("Generated comprehensive system insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating system insights: {e}")
            return {'error': str(e)}
        
    # 4. OPTIONAL: Add context debugging method to TrieMemory:
    def debug_context_relationships(self, token: str = None) -> Dict[str, Any]:
        """ADDED: Debug context relationships for specific token or system overview."""
        try:
            if token:
                # Debug specific token's context relationships
                if token not in self.embeddings:
                    return {'error': f"Token '{token}' not found in embeddings"}

                node = self.embeddings[token]
                context_info = {
                    'token': token,
                    'total_children': len(getattr(node, 'children', {})),
                    'context_aware_children': len(getattr(node, 'context_children', {})),
                    'contexts': {}
                }

                if hasattr(node, 'context_children'):
                    for context_id, children in node.context_children.items():
                        context_info['contexts'][context_id] = {
                            'children_count': len(children),
                            'children_tokens': list(children.keys()),
                            'sequence': self.sequence_contexts.get(context_id, {}).get('tokens', [])
                        }

                return context_info
            else:
                # System overview
                total_nodes = len(self.embeddings)
                context_aware_nodes = sum(1 for node in self.embeddings.values() 
                                        if hasattr(node, 'context_children') and node.context_children)

                return {
                    'total_nodes': total_nodes,
                    'context_aware_nodes': context_aware_nodes,
                    'total_contexts': len(self.sequence_contexts),
                    'context_coverage': context_aware_nodes / total_nodes if total_nodes > 0 else 0.0,
                    'average_contexts_per_aware_node': sum(len(node.context_children) 
                                                         for node in self.embeddings.values() 
                                                         if hasattr(node, 'context_children')) / max(1, context_aware_nodes)
                }

        except Exception as e:
            logger.error(f"Error debugging context relationships: {e}")
            return {'error': str(e)}
    
    def close(self):
        """PRESERVED: Clean up resources."""
        self.trie_memory.close()
        logger.info("Closed PredictiveSystem")