import hashlib
from typing import List


def compare_approaches():
        """Compare different key generation approaches."""
        
        test_cases = [
            ([], None, "Root"),
            (["hello"], "world", "Simple path"),
            (["user"], "-1", "Problematic token"),
            (["some", "long", "path"], "!", "Complex case"),
            (["numbers"], "123.45", "Numeric token"),
            (["special"], "weird/chars\\here:test", "Special characters")
        ]
        
        print("🔑 EMBEDDING KEY COMPARISON:")
        print("=" * 50)
        
        for path_tokens, token, description in test_cases:
            print(f"\n{description}:")
            print(f"  Input: path={path_tokens}, token='{token}'")
            
            # Simple hash
            simple = generate_simple_hash(path_tokens, token)
            print(f"  Simple:     {simple}")
            
            # With level
            with_level = generate_with_level(path_tokens, token)
            print(f"  With level: {with_level}")
            
            # Ultra simple
            ultra = generate_ultra_simple(path_tokens, token)
            print(f"  Ultra:      {ultra}")
    
    
def generate_simple_hash(path_tokens: List[str], token: str) -> str:
    """Simple hash implementation."""
    if not path_tokens and not token:
        return "emb_root"
    
    path_str = '/'.join(path_tokens) if path_tokens else ''
    full_path = f"{path_str}:{token}" if token else path_str
    hash_value = hashlib.sha256(full_path.encode()).hexdigest()
    return f"emb_{hash_value[:12]}"


def generate_with_level(path_tokens: List[str], token: str) -> str:
    """Hash with level info."""
    level = len(path_tokens)
    if not path_tokens and not token:
        return "emb_root"
    
    path_str = '/'.join(path_tokens) if path_tokens else ''
    full_path = f"{path_str}:{token}" if token else path_str
    hash_value = hashlib.sha256(full_path.encode()).hexdigest()
    return f"emb_L{level}_{hash_value[:10]}"

    
def generate_ultra_simple(path_tokens: List[str], token: str) -> str:
    """Ultra simple implementation."""
    combined = f"{'/'.join(path_tokens)}:{token or ''}"
    return f"emb_{hashlib.md5(combined.encode()).hexdigest()[:12]}"