# Configure logging for execution transparency
from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

import numpy as np

from trie_memory import TrieMemory
from trie_node import SemanticTrieNode


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiNodeBeamSearch:
    """
    FIXED: Multi-node beam search with corrected trie traversal logic.
    
    CRITICAL FIXES:
    1. _collect_multi_source_candidates now only uses actual trie children
    2. Removed all fallback logic that collected random nodes
    3. Restored basic trie traversal principles
    """
    
    def __init__(self, trie_memory: TrieMemory, beam_width: int = 5):
        self.trie_memory = trie_memory
        self.beam_width = beam_width
        self.max_generation_length = 20
        self.activation_weight = 0.25
        self.rl_weight = 0.25
        self.relevance_weight = 0.25
        self.coherence_weight = 0.15
        self.completeness_weight = 0.10
        logger.info(f"Initialized MultiNodeBeamSearch with beam_width={beam_width}")

    @dataclass
    class BeamState:
        """Represents a single beam in the search space."""
        path_tokens: List[str]
        path_nodes: List[SemanticTrieNode]
        cumulative_score: float
        last_node_embedding: np.ndarray
        generation_step: int = 0
        
        def __post_init__(self):
            if self.last_node_embedding is None and self.path_nodes:
                self.last_node_embedding = self.path_nodes[-1].token_embedding.embedding

    def generate_with_multi_node_linking(self, context_tokens: List[str], 
                                       target_length: int = 10) -> Tuple[List[str], float, List[Dict]]:
        """
        PRESERVED: Generate text using beam search (with corrected candidate collection).
        """
        logger.info(f"Starting multi-node beam search for context: {context_tokens}")
        
        try:
            initial_beams = self._initialize_beams_from_context(context_tokens)
            if not initial_beams:
                logger.warning("No initial beams found from context")
                return [], 0.0, []
            
            active_beams = initial_beams
            generation_details = []
            
            for step in range(target_length):
                logger.debug(f"Generation step {step + 1}: {len(active_beams)} active beams")
                
                all_candidates = []
                
                for beam in active_beams:
                    candidates = self._collect_multi_source_candidates(beam, context_tokens)
                    for candidate in candidates:
                        score_breakdown = self._calculate_comprehensive_score(
                            candidate, beam, context_tokens
                        )
                        
                        new_beam = self.BeamState(
                            path_tokens=beam.path_tokens + [candidate.token],
                            path_nodes=beam.path_nodes + [candidate],
                            cumulative_score=beam.cumulative_score + score_breakdown['total_score'],
                            last_node_embedding=candidate.token_embedding.embedding,
                            generation_step=step + 1
                        )
                        
                        all_candidates.append((new_beam, score_breakdown))
                
                if not all_candidates:
                    logger.warning(f"No candidates found at step {step + 1}")
                    break
                
                all_candidates.sort(key=lambda x: x[0].cumulative_score, reverse=True)
                active_beams = [candidate[0] for candidate in all_candidates[:self.beam_width]]
                
                step_details = {
                    'step': step + 1,
                    'num_candidates': len(all_candidates),
                    'selected_beams': len(active_beams),
                    'top_scores': [beam.cumulative_score for beam in active_beams[:3]],
                    'top_tokens': [beam.path_tokens[-1] if beam.path_tokens else None for beam in active_beams[:3]]
                }
                generation_details.append(step_details)
                
                if self._should_stop_generation(active_beams):
                    logger.info(f"Natural stopping point detected at step {step + 1}")
                    break
                
                logger.debug(f"Step {step + 1} completed. Top beam score: {active_beams[0].cumulative_score:.3f}")
            
            if active_beams:
                best_beam = active_beams[0]
                final_tokens = best_beam.path_tokens[len(context_tokens):]
                final_score = best_beam.cumulative_score / max(1, len(final_tokens))
                
                logger.info(f"Generation completed: {len(final_tokens)} tokens, final score: {final_score:.3f}")
                return final_tokens, final_score, generation_details
            else:
                logger.warning("No valid beams remaining")
                return [], 0.0, generation_details
                
        except Exception as e:
            logger.error(f"Error in multi-node beam search: {e}")
            return [], 0.0, []

    def _collect_multi_source_candidates(self, beam: BeamState, context_tokens: List[str]) -> List[SemanticTrieNode]:
        """
        CRITICAL FIX: Only collect actual trie children as continuations.
        
        REMOVED:
        - Semantic similarity candidates (not actual continuations)
        - High-activation nodes from elsewhere (not continuations)  
        - Emergency fallback candidates (fake continuations)
        - Broad trie search (random nodes, not continuations)
        
        PRESERVED:
        - ONLY direct children of current node are valid continuations
        - If node has no children → return empty list (natural sequence end)
        - Basic trie traversal principles
        """
        candidates = []
        
        try:
            logger.debug(f"Collecting candidates for beam with {len(beam.path_tokens)} tokens")
            
            if not beam.path_nodes:
                logger.debug("No path nodes in beam, cannot collect candidates")
                return candidates
            
            last_node = beam.path_nodes[-1]
            logger.debug(f"Collecting children of '{last_node.token}': {len(last_node.children)} children available")
            
            # CRITICAL FIX: ONLY collect direct children (actual continuations)
            if last_node.children:
                child_tokens = list(last_node.children.keys())
                logger.debug(f"Available children: {child_tokens}")
                
                for child_token, child_node in last_node.children.items():
                    candidates.append(child_node)
                    logger.debug(f"Added actual continuation candidate: '{child_token}'")
                
                logger.info(f"Collected {len(candidates)} actual continuation children")
            else:
                logger.info(f"Node '{last_node.token}' has no children - natural sequence end")
            
            # REMOVED: All fallback logic that was collecting random nodes
            # This was the core issue causing nonsensical continuations
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error collecting candidates: {e}")
            return []

    def _calculate_comprehensive_score(self, candidate_node: SemanticTrieNode, 
                                     current_beam: BeamState, 
                                     context_tokens: List[str]) -> Dict[str, float]:
        """
        PRESERVED: Multi-dimensional scoring with activation and RL.
        """
        try:
            scores = {}
            
            activation_score = candidate_node.activation_level
            scores['activation'] = activation_score
            
            avg_reward = candidate_node.metadata.get('avg_reward', 0.0)
            if candidate_node.reward_history:
                recent_rewards = candidate_node.reward_history[-5:]
                recent_avg = sum(recent_rewards) / len(recent_rewards)
                rl_score = 0.6 * recent_avg + 0.4 * avg_reward
            else:
                rl_score = avg_reward
            scores['rl_reward'] = rl_score
            
            if self.trie_memory.context_window.current_context_embedding is not None:
                relevance_score = candidate_node.calculate_relevance(
                    context_embedding=self.trie_memory.context_window.current_context_embedding
                )
            else:
                relevance_score = 0.5
            scores['relevance'] = relevance_score
            
            if current_beam.path_nodes:
                last_embedding = current_beam.last_node_embedding
                candidate_embedding = candidate_node.token_embedding.embedding
                coherence_score = np.dot(last_embedding, candidate_embedding)
                
                if candidate_node.token not in ['.', '!', '?', ',', ';', ':', '"', "'", '(', ')', '[', ']']:
                    coherence_score += 0.15
                
                if self._is_natural_transition(
                    current_beam.path_tokens[-1] if current_beam.path_tokens else "", 
                    candidate_node.token
                ):
                    coherence_score += 0.1
            else:
                coherence_score = 0.5
            scores['coherence'] = coherence_score
            
            completeness_score = 0.0
            if candidate_node.is_complete:
                sequence_length = len(current_beam.path_tokens)
                if candidate_node.token in ['.', '!', '?']:
                    if current_beam.path_tokens and current_beam.path_tokens[-1] in ['.', '!', '?']:
                        completeness_score = -0.3
                    elif sequence_length >= 5:
                        completeness_score = 0.2
                    else:
                        completeness_score = -0.1
                else:
                    if sequence_length >= 4:
                        completeness_score = 0.3
            scores['completeness'] = completeness_score
            
            repetition_penalty = 0.0
            if current_beam.path_tokens:
                recent_tokens = current_beam.path_tokens[-3:]
                if candidate_node.token in recent_tokens:
                    repetition_count = recent_tokens.count(candidate_node.token)
                    repetition_penalty = repetition_count * 0.2
            
            total_score = (
                scores['activation'] * self.activation_weight +
                scores['rl_reward'] * self.rl_weight +
                scores['relevance'] * self.relevance_weight +
                scores['coherence'] * self.coherence_weight +
                scores['completeness'] * self.completeness_weight
            ) - repetition_penalty
            
            scores['total_score'] = total_score
            
            logger.debug(f"Scoring '{candidate_node.token}': total={total_score:.3f}")
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive score: {e}")
            return {'total_score': 0.0, 'activation': 0.0, 'rl_reward': 0.0, 
                   'relevance': 0.0, 'coherence': 0.0, 'completeness': 0.0}

    def _initialize_beams_from_context(self, context_tokens: List[str]) -> List[BeamState]:
        """
        PRESERVED: Initialize beam search from context tokens.
        """
        try:
            current_node = self.trie_memory.root
            path_nodes = []
            
            for token in context_tokens:
                if token in current_node.children:
                    current_node = current_node.children[token]
                    path_nodes.append(current_node)
                else:
                    break
            
            if path_nodes:
                matched_length = len(path_nodes)
                logger.info(f"Found context path of length {matched_length}")
                
                last_embedding = path_nodes[-1].token_embedding.embedding
                initial_beam = self.BeamState(
                    path_tokens=context_tokens[:matched_length],
                    path_nodes=path_nodes,
                    cumulative_score=0.0,
                    last_node_embedding=last_embedding
                )
                return [initial_beam]
            else:
                logger.info("No context match found, starting with high-activation nodes")
                high_activation_starts = self._find_high_activation_nodes(
                    activation_threshold=0.3, max_candidates=3
                )
                
                beams = []
                for node in high_activation_starts:
                    beam = self.BeamState(
                        path_tokens=[node.token],
                        path_nodes=[node],
                        cumulative_score=node.activation_level,
                        last_node_embedding=node.token_embedding.embedding
                    )
                    beams.append(beam)
                
                logger.info(f"Initialized {len(beams)} beams from high-activation nodes")
                return beams
                
        except Exception as e:
            logger.error(f"Error initializing beams: {e}")
            return []

    def _should_stop_generation(self, active_beams: List[BeamState]) -> bool:
        """
        PRESERVED: Check if generation should stop naturally.
        """
        try:
            if not active_beams:
                return True
            
            best_beam = active_beams[0]
            
            if not best_beam.path_tokens or not best_beam.path_nodes:
                return False
            
            last_token = best_beam.path_tokens[-1]
            last_node = best_beam.path_nodes[-1]
            sequence_length = len(best_beam.path_tokens)
            
            if sequence_length < 4:
                return False
            
            if last_token in ['.', '!', '?', ',', ';', ':']:
                previous_tokens = best_beam.path_tokens[:-1]
                if previous_tokens:
                    prev_token = previous_tokens[-1]
                    if prev_token in ['.', '!', '?']:
                        return False
                    
                    content_tokens = [t for t in previous_tokens if t not in ['.', '!', '?', ',', ';', ':', '"', "'", '(', ')', '[', ']']]
                    if len(content_tokens) < 4:
                        return False
            
            if last_node.is_complete:
                if last_token in ['.', '!', '?']:
                    content_before_punct = [t for t in best_beam.path_tokens[:-1] 
                                          if t not in ['.', '!', '?', ',', ';', ':', '"', "'", '(', ')', '[', ']']]
                    
                    if len(content_before_punct) >= 4:
                        return True
                    else:
                        return False
                else:
                    if sequence_length >= 6:
                        return True
                    else:
                        return False
            
            if sequence_length >= 4:
                recent_tokens = best_beam.path_tokens[-4:]
                if len(set(recent_tokens)) <= 2:
                    return True
            
            if sequence_length >= 12:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking stopping condition: {e}")
            return False

    def _is_natural_transition(self, prev_token: str, next_token: str) -> bool:
        """PRESERVED: Simple heuristic for natural language transitions."""
        patterns = [
            (prev_token.endswith('ing'), next_token in ['a', 'the', 'and']),
            (prev_token in ['the', 'a', 'an'], next_token.isalpha()),
            (prev_token in ['i', 'you', 'we', 'they'], next_token in ['am', 'are', 'will', 'can']),
        ]
        
        return any(condition for condition in patterns)

    def _find_high_activation_nodes(self, activation_threshold: float = 0.6, 
                                  max_candidates: int = 5) -> List[SemanticTrieNode]:
        """PRESERVED: Find nodes with high activation levels."""
        high_activation_nodes = []
        
        try:
            def traverse_for_activation(node: SemanticTrieNode):
                if node.activation_level >= activation_threshold:
                    high_activation_nodes.append(node)
                
                for child in node.children.values():
                    traverse_for_activation(child)
                    if len(high_activation_nodes) >= max_candidates * 2:
                        break
            
            traverse_for_activation(self.trie_memory.root)
            
            high_activation_nodes.sort(key=lambda x: x.activation_level, reverse=True)
            return high_activation_nodes[:max_candidates]
            
        except Exception as e:
            logger.error(f"Error finding high activation nodes: {e}")
            return []

    def _find_high_reward_nodes(self, reward_threshold: float = 0.7, 
                              max_candidates: int = 5) -> List[SemanticTrieNode]:
        """PRESERVED: Find nodes with high RL reward scores."""
        high_reward_nodes = []
        
        try:
            def traverse_for_rewards(node: SemanticTrieNode):
                avg_reward = node.metadata.get('avg_reward', 0.0)
                if avg_reward >= reward_threshold:
                    high_reward_nodes.append(node)
                
                for child in node.children.values():
                    traverse_for_rewards(child)
                    if len(high_reward_nodes) >= max_candidates * 2:
                        break
            
            traverse_for_rewards(self.trie_memory.root)
            
            high_reward_nodes.sort(key=lambda x: x.metadata.get('avg_reward', 0.0), reverse=True)
            return high_reward_nodes[:max_candidates]
            
        except Exception as e:
            logger.error(f"Error finding high reward nodes: {e}")
            return []
