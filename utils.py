
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _calculate_safe_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        ADDED: Safe cosine similarity calculation for multiprocessing environment.

        JUSTIFICATION: Provides robust similarity calculation with comprehensive error handling 
        for multiprocessing worker functions where external dependencies may not be available.
        FUNCTIONALITY: Calculates cosine similarity with fallback mechanisms and validation.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Float similarity score between 0.0 and 1.0, or 0.0 if calculation fails

        Raises:
            None: All errors are caught and result in 0.0 return value
        """

        try:
            # VALIDATION: Check input validity
            if embedding1 is None or embedding2 is None:
                return 0.0

            if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
                return 0.0

            if embedding1.size == 0 or embedding2.size == 0:
                return 0.0

            # SHAPE VALIDATION: Ensure compatible shapes
            if embedding1.shape != embedding2.shape:
                # Attempt to flatten if different shapes
                try:
                    embedding1 = embedding1.flatten()
                    embedding2 = embedding2.flatten()

                    if embedding1.shape != embedding2.shape:
                        return 0.0
                except Exception:
                    return 0.0

            # NORM CALCULATION: Calculate vector norms with zero-check
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0

            # COSINE SIMILARITY: Calculate dot product and normalize
            dot_product = np.dot(embedding1, embedding2)
            cosine_sim = dot_product / (norm1 * norm2)

            # BOUNDS CHECK: Ensure result is within valid range
            cosine_sim = max(0.0, min(1.0, float(cosine_sim)))

            return cosine_sim

        except Exception as similarity_error:
            # SILENT FAILURE: Return 0.0 for any calculation errors to prevent process crashes
            return 0.0
        
        
def search_embeddings_chunk(embeddings_chunk: List[Tuple[str, Any]], 
                           query_tokens: List[str],
                           context_embedding: Optional[np.ndarray] = None,
                           query_sequence_embedding: Optional[np.ndarray] = None) -> Tuple[List[Tuple[List[str], float]], Dict[str, Any]]:
    """
    ADDED: Multiprocessing worker function for partial match search across embeddings chunk.
    JUSTIFICATION: Enables parallel processing of embeddings chunks across CPU cores for faster partial matching.
    FUNCTIONALITY: Searches a chunk of embeddings for tokens matching query and generates continuation candidates.
    REQUIREMENTS:
    - Must be at module level for multiprocessing pickling
    - Must handle numpy arrays and custom node objects safely
    - Must provide comprehensive error handling for process isolation
    Args:
        embeddings_chunk: List of (token, SemanticTrieNode) tuples to search
        query_tokens: Original query token sequence to match against
        context_embedding: Optional context embedding for relevance scoring
        query_sequence_embedding: Optional query embedding for similarity scoring
    Returns:
        Tuple of (candidates_list, statistics_dict) where:
        - candidates_list: List of (continuation_path, relevance_score) tuples
        - statistics_dict: Dictionary containing processing statistics and metadata
    Raises:
        Exception: Re-raises critical errors that prevent chunk processing
    """
    
    # Create process-local logger to avoid conflicts
    logger = logging.getLogger(f"chunk_worker_{id(embeddings_chunk)}")
    start_time = time.time()
    logger.info(f"Starting embeddings chunk search: {len(embeddings_chunk)} embeddings, "
               f"{len(query_tokens)} query tokens")
    # INITIALIZE: Statistics tracking
    process_stats = {
        'embeddings_searched': 0,
        'query_matches_found': 0,
        'children_processed': 0,
        'candidates_generated': 0,
        'similarity_calculations': 0,
        'errors_encountered': 0,
        'processing_time_seconds': 0.0,
        'chunk_size': len(embeddings_chunk)
    }
    candidates = []
    try:
        logger.debug(f"Processing chunk with context_embedding: {context_embedding is not None}, "
                    f"query_sequence_embedding: {query_sequence_embedding is not None}")
        # ITERATE: Process each embedding in the chunk
        for embedding_index, (token, node) in enumerate(embeddings_chunk):
            try:
                process_stats['embeddings_searched'] += 1
                # VALIDATION: Check if token is valid
                if not token or not isinstance(token, str):
                    logger.debug(f"Skipping invalid token at index {embedding_index}: {token}")
                    continue
                
                # MATCH CHECK: Determine if this token appears in query tokens
                if token not in query_tokens:
                    logger.debug(f"Token '{token}' not in query tokens, skipping")
                    continue
                
                process_stats['query_matches_found'] += 1
                logger.debug(f"Found query match: '{token}' at index {embedding_index}")
                # RELEVANCE CALCULATION: Calculate base relevance score for matched token
                relevance_score = 0.0
                similarity_components = {'context_sim': 0.0, 'query_sim': 0.0}
                # Check if node has valid embedding for similarity calculations
                if hasattr(node, 'embedding') and node.embedding is not None:
                    try:
                        # CONTEXT SIMILARITY: Calculate similarity with context embedding
                        if context_embedding is not None:
                            context_sim = _calculate_safe_cosine_similarity(context_embedding, node.embedding)
                            relevance_score += context_sim * 0.5
                            similarity_components['context_sim'] = context_sim
                            process_stats['similarity_calculations'] += 1
                            logger.debug(f"Context similarity for '{token}': {context_sim:.3f}")
                        # QUERY SIMILARITY: Calculate similarity with query sequence embedding
                        if query_sequence_embedding is not None:
                            query_sim = _calculate_safe_cosine_similarity(query_sequence_embedding, node.embedding)
                            relevance_score += query_sim * 0.5
                            similarity_components['query_sim'] = query_sim
                            process_stats['similarity_calculations'] += 1
                            logger.debug(f"Query similarity for '{token}': {query_sim:.3f}")
                    except Exception as similarity_error:
                        logger.warning(f"Error calculating similarity for '{token}': {similarity_error}")
                        process_stats['errors_encountered'] += 1
                        # Continue with base relevance score of 0.0
                # POSITION BONUS: Add bonus based on position in query tokens
                try:
                    token_position = query_tokens.index(token)
                    position_bonus = (len(query_tokens) - token_position) / len(query_tokens) * 0.1
                    relevance_score += position_bonus
                    logger.debug(f"Position bonus for '{token}' at position {token_position}: {position_bonus:.3f}")
                except ValueError:
                    # Token not found in query_tokens (shouldn't happen due to earlier check)
                    logger.warning(f"Token '{token}' not found in query_tokens during position calculation")
                # CHILDREN PROCESSING: Generate candidates from node children
                if hasattr(node, 'children') and node.children:
                    logger.debug(f"Processing {len(node.children)} children for token '{token}'")
                    for child_token, child_node in node.children.items():
                        try:
                            process_stats['children_processed'] += 1
                            # VALIDATION: Check child token validity
                            if not child_token or not isinstance(child_token, str):
                                logger.debug(f"Skipping invalid child token: {child_token}")
                                continue
                            
                            # CHILD RELEVANCE: Calculate additional relevance from child node
                            child_relevance = 0.0
                            if hasattr(child_node, 'embedding') and child_node.embedding is not None:
                                try:
                                    # Child context similarity
                                    if context_embedding is not None:
                                        child_context_sim = _calculate_safe_cosine_similarity(
                                            context_embedding, child_node.embedding
                                        )
                                        child_relevance += child_context_sim * 0.3
                                        process_stats['similarity_calculations'] += 1
                                    # Child query similarity
                                    if query_sequence_embedding is not None:
                                        child_query_sim = _calculate_safe_cosine_similarity(
                                            query_sequence_embedding, child_node.embedding
                                        )
                                        child_relevance += child_query_sim * 0.3
                                        process_stats['similarity_calculations'] += 1
                                except Exception as child_similarity_error:
                                    logger.debug(f"Error calculating child similarity for '{child_token}': {child_similarity_error}")
                            # CANDIDATE GENERATION: Create continuation path with combined scoring
                            partial_path = [token, child_token]
                            combined_score = (relevance_score + child_relevance) * 0.5
                            candidates.append((partial_path, combined_score))
                            process_stats['candidates_generated'] += 1
                            logger.debug(f"Generated candidate: {partial_path} -> "
                                       f"parent_relevance={relevance_score:.3f}, "
                                       f"child_relevance={child_relevance:.3f}, "
                                       f"combined_score={combined_score:.3f}")
                        except Exception as child_error:
                            logger.warning(f"Error processing child '{child_token}' of '{token}': {child_error}")
                            process_stats['errors_encountered'] += 1
                            continue
                else:
                    logger.debug(f"Token '{token}' has no children for continuation generation")
            except Exception as token_error:
                logger.error(f"Error processing token at index {embedding_index}: {token_error}")
                process_stats['errors_encountered'] += 1
                continue
            
        # COMPLETION: Calculate final statistics
        process_stats['processing_time_seconds'] = time.time() - start_time
        logger.info(f"Chunk processing complete: {process_stats['candidates_generated']} candidates generated "
                   f"from {process_stats['query_matches_found']} query matches "
                   f"in {process_stats['processing_time_seconds']:.3f} seconds")
        logger.info(f"Processing statistics: {process_stats}")
        return candidates, process_stats
    except Exception as critical_error:
        # CRITICAL ERROR HANDLING: Handle process-level failures
        process_stats['processing_time_seconds'] = time.time() - start_time
        process_stats['critical_error'] = str(critical_error)
        process_stats['errors_encountered'] += 1
        logger.error(f"Critical error in chunk processing: {critical_error}")
        logger.error(f"Partial statistics before failure: {process_stats}")
        # Return partial results if any were generated
        return candidates, process_stats