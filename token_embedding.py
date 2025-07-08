import hashlib
import logging
import string
import time
from typing import List, Optional
import numpy as np
# Configure logging for execution transparency
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PRESERVED: Original TokenEmbedding class (unchanged)
class TokenEmbedding:
    """
    PRESERVED: Represents a single token with its embedding and metadata.
    """
    def __init__(self, token: str, embedding: np.ndarray, binary_values: List[int], 
                 ascii_values: List[int], index: int = 0):
        self.token = token
        self.embedding = embedding
        self.binary_values = binary_values
        self.ascii_values = ascii_values
        self.index = index
        self.predecessor_index: Optional[int] = None
        self.successor_index: Optional[int] = None
        
        # Trie-specific enhancements
        self.relevance_score: float = 0.0
        self.activation_level: float = 0.0
        self.is_complete: bool = False
        self.last_accessed: float = time.time()
        self.access_count: int = 0

# PRESERVED: Original create_token_embedding function (unchanged)
def create_token_embedding(token: str) -> TokenEmbedding:
    """
    PRESERVED: Creates a 1024-dimensional embedding for a single token.
    """
    logger.info(f"Starting embedding creation for token: '{token}'")
    
    if not isinstance(token, str):
        logger.error(f"Invalid input type: {type(token)}. Expected string.")
        raise TypeError("Token must be a string")
    
    if not token:
        logger.error("Empty token provided")
        raise ValueError("Token cannot be empty")
    
    try:
        # Extract binary and ASCII values
        logger.info("Extracting binary and ASCII values from token")
        binary_values = []
        ascii_values = []
        
        for char in token:
            ascii_val = ord(char)
            binary_val = bin(ascii_val)[2:]
            
            ascii_values.append(ascii_val)
            binary_values.extend([int(bit) for bit in binary_val])
        
        logger.info(f"Extracted {len(ascii_values)} ASCII values and {len(binary_values)} binary bits")
        
        # Create deterministic seed from token for reproducible embeddings
        token_hash = hashlib.md5(token.encode()).hexdigest()
        seed = int(token_hash[:8], 16)
        np.random.seed(seed)
        
        logger.info(f"Generated deterministic seed: {seed}")
        
        # Initialize 1024-dimensional embedding
        embedding = np.zeros(1024, dtype=np.float32)
        
        # Section 1 (0-255): ASCII value features
        logger.info("Populating ASCII value features (dimensions 0-255)")
        for i, ascii_val in enumerate(ascii_values[:256]):
            if i < 256:
                embedding[i] = ascii_val / 127.0
        
        # Section 2 (256-511): Binary pattern features
        logger.info("Populating binary pattern features (dimensions 256-511)")
        binary_chunk_size = len(binary_values) // 256 + 1
        for i in range(256):
            start_idx = i * binary_chunk_size
            end_idx = min(start_idx + binary_chunk_size, len(binary_values))
            if start_idx < len(binary_values):
                chunk = binary_values[start_idx:end_idx]
                if chunk:
                    embedding[256 + i] = sum(chunk) / len(chunk)
        
        # Section 3 (512-767): Token characteristics
        logger.info("Populating token characteristic features (dimensions 512-767)")
        embedding[512] = len(token) / 100.0
        embedding[513] = sum(c.isupper() for c in token) / len(token)
        embedding[514] = sum(c.islower() for c in token) / len(token)
        embedding[515] = sum(c.isdigit() for c in token) / len(token)
        embedding[516] = sum(c.isalpha() for c in token) / len(token)
        embedding[517] = sum(c in string.punctuation for c in token) / len(token)
        
        for i in range(518, 768):
            pos_factor = (i - 518) / 250.0
            embedding[i] = np.sin(ascii_values[0] * pos_factor) if ascii_values else 0.0
        
        # Section 4 (768-1023): Random features with token-specific patterns
        logger.info("Populating pattern-based random features (dimensions 768-1023)")
        pattern_features = np.random.normal(0, 0.1, 256)
        
        for i, feature in enumerate(pattern_features):
            char_influence = ascii_values[i % len(ascii_values)] / 127.0 if ascii_values else 0
            embedding[768 + i] = feature * (1 + 0.1 * char_influence)
        
        # Normalize the entire embedding to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            logger.info(f"Normalized embedding to unit length: {norm:.4f}")
        
        token_embedding = TokenEmbedding(
            token=token,
            embedding=embedding,
            binary_values=binary_values,
            ascii_values=ascii_values
        )
        
        logger.info(f"Successfully created embedding for token '{token}'")
        return token_embedding
        
    except Exception as e:
        logger.error(f"Error creating embedding for token '{token}': {str(e)}")
        raise



