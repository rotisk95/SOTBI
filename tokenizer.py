from typing import List


class Tokenizer:
    def __init__(self):
        # Punctuation that should be separate tokens but attached in output
        self.sentence_endings = ['.', '!', '?']
        self.separators = [',', ';', ':', '(', ')', '[', ']', '{', '}']
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize preserving case and creating natural spacing
        """
        try:
            # DON'T lowercase - preserve original case
            original_text = text.strip()
            
            # Add spaces around punctuation for splitting
            working_text = original_text
            all_punct = self.sentence_endings + self.separators
            
            for punct in all_punct:
                working_text = working_text.replace(punct, f' {punct} ')
            
            # Split and clean
            tokens = [token.strip() for token in working_text.split() if token.strip()]
            
            return tokens
            
        except Exception as e:
            print(f"Error tokenizing: {e}")
            return []
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to natural text with proper spacing
        """
        if not tokens:
            return ""
        
        result = []
        
        for i, token in enumerate(tokens):
            if i == 0:
                # First token always gets added
                result.append(token)
            elif token in self.sentence_endings + [',', ';', ':']:
                # Punctuation attaches to previous word (no space before)
                result[-1] += token
            elif token in ['(', '[', '{']:
                # Opening brackets - space before, no space after
                result.append(token)
            elif token in [')', ']', '}']:
                # Closing brackets - no space before, space after (handled by next iteration)
                result[-1] += token
            else:
                # Regular word - add with space
                result.append(token)
        
        return ' '.join(result)