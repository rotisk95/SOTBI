# OPTIMIZED: Smart caching system for aggregate calculations
# ACCOUNTABILITY: Replaces expensive always-update with threshold-based caching

import logging
import time
import numpy as np
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedAggregateManager:
    """
    ADDED: Smart caching manager for expensive aggregate calculations.
    
    JUSTIFICATION: Prevents unnecessary recalculation of global centroids
    when embeddings haven't changed significantly.
    
    FEATURES:
    - Threshold-based updates (only update after N new embeddings)
    - Time-based invalidation (force update after N seconds)
    - Incremental update tracking
    - Cache hit/miss statistics
    """
    
    def __init__(self, 
                 embedding_threshold: int = 50,     # Update after 50 new embeddings
                 time_threshold: float = 30.0,      # Force update after 30 seconds
                 activation_threshold: int = 20):   # Update activation aggregates more frequently
        
        # Configuration
        self.embedding_threshold = embedding_threshold
        self.time_threshold = time_threshold  
        self.activation_threshold = activation_threshold
        
        # Cache storage
        self.cached_global_centroid: Optional[np.ndarray] = None
        self.cached_activation_weighted: Optional[np.ndarray] = None
        
        # Update tracking
        self.embeddings_added_since_last_update = 0
        self.activations_updated_since_last_update = 0
        self.last_global_update_time = 0.0
        self.last_activation_update_time = 0.0
        self.total_embeddings_at_last_update = 0
        
        # Performance statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0, 
            'forced_updates': 0,
            'threshold_updates': 0,
            'total_time_saved': 0.0,
            'total_calculation_time': 0.0
        }
        
        logger.info(f"Initialized OptimizedAggregateManager: "
                   f"embedding_threshold={embedding_threshold}, "
                   f"time_threshold={time_threshold}s")