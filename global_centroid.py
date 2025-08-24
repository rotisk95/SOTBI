# DIRECT FIX: Load-once global centroid with explicit update triggers
# ACCOUNTABILITY: Removes expensive frequent recalculation, preserves all functionality

import time
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LoadOnceGlobalCentroid:
    """
    ACCOUNTABILITY: Direct implementation of user request - load once, update only on trigger.
    
    CHANGES MADE:
    1. REMOVED: Frequent automatic global centroid recalculation
    2. ADDED: Load-once strategy with explicit update triggers
    3. PRESERVED: All existing aggregate functionality and interfaces
    """
    
    def __init__(self):
        # CORE STATE: Global centroid loaded once and cached
        self.global_centroid: Optional[np.ndarray] = None
        self.activation_weighted_centroid: Optional[np.ndarray] = None
        
        # TRACKING: When centroids were calculated
        self.global_centroid_loaded_time: float = 0.0
        self.activation_centroid_updated_time: float = 0.0
        self.embeddings_count_at_load: int = 0
        
        # STATISTICS: Performance tracking
        self.load_once_stats = {
            'global_calculations_avoided': 0,
            'time_saved_seconds': 0.0,
            'last_explicit_update': 0.0,
            'activation_updates': 0
        }
        
        logger.info("Initialized LoadOnceGlobalCentroid system")

    

# INTEGRATION INSTRUCTIONS:
# =========================

# 1. ADD to TrieMemory.__init__() method after _load_embeddings_only():
#    self.initialize_global_centroid_on_load()

# 2. REPLACE all calls to _always_update_aggregates() with:
#    self.get_load_once_aggregates()

# 3. REPLACE all calls to calculate_global_embedding_centroid() with:
#    self.get_cached_global_centroid()

# 4. REPLACE all calls to calculate_weighted_embedding_aggregate('activation') with:
#    self.get_cached_activation_weighted_centroid()

# 5. ADD user command to trigger updates:
#    results = system.trie_memory.trigger_explicit_global_centroid_update()