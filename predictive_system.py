# Configure logging for execution transparency
import logging
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from beam_search import MultiNodeBeamSearch
from event_driven_activation import EventDrivenActivation
from tokenizer import Tokenizer
from trie_memory import TrieMemory
from token_embedding import TokenEmbedding, create_token_embedding
from trie_node import TrieNode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictiveSystem:
    """
    CORRECTED: Main system that integrates trie-based memory with beam search.
    ENHANCED: Integrated corrected EventDrivenActivation with context-aware filtering.
    ADDED: Proper activation lifecycle management and contamination prevention.
    FIXED: Proper integration of EventDrivenActivation class methods.
    REMOVED: Duplicate _tokenize method to prevent conflicts.
    PRESERVED: All existing functionality and interfaces.
    """
    
    def __init__(self, db_path: str = "./trie_memory.lmdb"):
        # PRESERVED: Initialize core components
        self.trie_memory = TrieMemory(db_path)
        self.identity_context = {}
        self.beam_search = MultiNodeBeamSearch(self.trie_memory)
        self.tokenizer = Tokenizer()
        self.use_activation_context = True  # Enable event-driven activation by default
        
        # ADDED: Initialize corrected EventDrivenActivation system
        # JUSTIFICATION: Provides context-aware activation with contamination prevention
        self.activation_system = EventDrivenActivation(self.trie_memory)
        
        logger.info("Initialized PredictiveSystem with corrected trie-based memory, beam search, and context-aware activation")
    
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
            sequence_id = self.trie_memory.add_sequence(tokens, reward)
            
            result = {
                'sequence_id': sequence_id,
                'tokens': tokens,
                'reward': reward,
                'timestamp': time.time(),
                'reward_type': 'positive' if reward > 0 else 'negative' if reward < 0 else 'neutral'
            }
            
            logger.info(f"Successfully processed input with sequence ID: {sequence_id[:8]}... "
                       f"(reward_type: {result['reward_type']})")
            return result
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            raise
    
    def predict_continuation(self, query: str, max_candidates: int = 5, 
                           use_beam_search: bool = False, beam_width: int = 5,
                           target_length: int = 8) -> Tuple[List[str], float]:
        """
        CORRECTED: Predict continuation with option for beam search and event-driven activation.
        ENHANCED: Proper integration with corrected activation system.
        FIXED: Import dependencies and method calls.
        """
        logger.info(f"Predicting continuation for: '{query}' (beam_search={use_beam_search}, activation_context={self.use_activation_context})")
        
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        try:
            query_tokens = self._tokenize(query)
            
            if use_beam_search:
                logger.info("Using beam search with corrected trie traversal")
                self.beam_search.beam_width = beam_width
                
                continuation, confidence, beam_details = self.beam_search.generate_with_multi_node_linking(
                    query_tokens, target_length=target_length
                )
                
                if beam_details:
                    logger.info(f"Beam search completed in {len(beam_details)} steps")
                
                logger.info(f"Beam search result: {continuation} with confidence: {confidence:.3f}")
                return continuation, confidence
                
            else:
                logger.info("Using simple prediction method with optional activation context")
                current_node = self.trie_memory.root
                
                # CORRECTED: Get embeddings and context
                query_embeddings = [create_token_embedding(token) for token in query_tokens]
                context_embedding = self.trie_memory.context_window.current_context_embedding
                query_sequence_embedding = self.trie_memory._aggregate_sequence_embedding(query_embeddings)
                
                # PRESERVED: Get normal trie traversal result
                normal_continuation, normal_confidence = self.trie_memory.find_best_continuation(
                    current_node, context_embedding, query_sequence_embedding, query_tokens, max_candidates, 100
                )

                # CORRECTED: Apply event-driven activation enhancement with proper integration
                if self.use_activation_context:
                    logger.info("Applying corrected event-driven activation context enhancement")
                    enhanced_continuation, enhanced_confidence = self.activation_system.enhance_prediction_with_activation_context(
                        query_tokens, (normal_continuation, normal_confidence)
                    )
                    logger.info(f"Enhanced prediction result: {enhanced_continuation} with confidence: {enhanced_confidence:.3f}")
                    return enhanced_continuation, enhanced_confidence
                else:
                    logger.info(f"Normal prediction result: {normal_continuation} with confidence: {normal_confidence:.3f}")
                    return normal_continuation, normal_confidence
            
        except Exception as e:
            logger.error(f"Error predicting continuation: {str(e)}")
            return [], 0.0
    
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
                }
            }
            
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
    
    def close(self):
        """PRESERVED: Clean up resources."""
        self.trie_memory.close()
        logger.info("Closed PredictiveSystem")