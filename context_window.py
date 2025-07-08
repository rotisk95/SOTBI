# Configure logging for execution transparency
from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import List, Optional

import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ContextWindow:
    """
    PRESERVED: Manages conversation context windows.
    """
    max_turns: int = 5
    max_tokens: int = 100
    time_window_seconds: int = 300
    
    def __init__(self, max_turns: int = 5, max_tokens: int = 100, time_window_seconds: int = 300):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.time_window_seconds = time_window_seconds
        self.conversation_history: deque = deque(maxlen=max_turns)
        self.current_context_embedding: Optional[np.ndarray] = None
        logger.info(f"Initialized ContextWindow with max_turns={max_turns}, max_tokens={max_tokens}")
    
    def add_turn(self, tokens: List[str], embedding: np.ndarray, timestamp: float = None):
        """Add a conversation turn to the context window."""
        if timestamp is None:
            timestamp = time.time()
        
        turn = {
            'tokens': tokens,
            'embedding': embedding,
            'timestamp': timestamp,
            'turn_id': len(self.conversation_history)
        }
        
        self.conversation_history.append(turn)
        self.current_context_embedding = embedding
        self._update_context_embedding()
        logger.info(f"Added conversation turn with {len(tokens)} tokens to context window")
    
    def _update_context_embedding(self):
        """Update the current context embedding from recent turns."""
        if not self.conversation_history:
            self.current_context_embedding = None
            return
        
        weighted_embeddings = []
        current_time = time.time()
        
        for i, turn in enumerate(self.conversation_history):
            recency_weight = (i + 1) / len(self.conversation_history)
            time_diff = current_time - turn['timestamp']
            time_weight = max(0.1, 1.0 - (time_diff / self.time_window_seconds))
            
            combined_weight = recency_weight * time_weight
            weighted_embeddings.append(turn['embedding'] * combined_weight)
        
        if weighted_embeddings:
            self.current_context_embedding = np.mean(weighted_embeddings, axis=0)
            norm = np.linalg.norm(self.current_context_embedding)
            if norm > 0:
                self.current_context_embedding = self.current_context_embedding / norm
        
        logger.info("Updated context embedding from conversation history")
    
    def get_context_similarity(self, token_embedding: np.ndarray) -> float:
            """Calculate similarity between token embedding and current context."""
            # Fix: Handle None token_embedding
            if token_embedding is None:
                return 0.0
            # Fix: Handle None current_context_embedding
            if self.current_context_embedding is None:
                return 0.0
            # Fix: Handle zero-norm embeddings
            if np.linalg.norm(self.current_context_embedding) == 0:
                return 0.0
            if np.linalg.norm(token_embedding) == 0:
                return 0.0

            return np.dot(self.current_context_embedding, token_embedding)
      
    def clear_context(self):
        """Clear the conversation context."""
        self.conversation_history.clear()
        self.current_context_embedding = None
        logger.info("Cleared conversation context")