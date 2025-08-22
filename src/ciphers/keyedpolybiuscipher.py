import numpy as np

class KeyedPolybiusCipher:
    """
    A Polybius square cipher where the grid is determined by a keyword.

    Example:
        >>> cipher = KeyedPolybiusCipher("CIPHER")
        >>> encrypted = await cipher.encrypt("HELLO WORLD!")
        >>> print(encrypted)
        '23 15 31 31 34 / 35 34 41 31 14 !'
        >>> decrypted = await cipher.decrypt(encrypted)
        >>> print(decrypted)
        'HELLO WORLD!'
    """
    def __init__(self, keyword: str):
        self.keyword = ''.join(filter(str.isalpha, keyword.upper()))
        self._initialize_grid()

    def _initialize_grid(self) -> None:
        seen = set()
        keyword_unique = []
        for c in self.keyword:
            if c not in seen:
                seen.add(c)
                keyword_unique.append(c)

        # Exclude 'J' (combined with 'I')
        alphabet = [c for c in 'ABCDEFGHIKLMNOPQRSTUVWXYZ' if c not in seen]
        grid_chars = keyword_unique + alphabet

        self.grid = np.array(grid_chars).reshape(5, 5)
        self.positions = {char: (i // 5, i % 5) for i, char in enumerate(grid_chars)}
        self.positions['J'] = self.positions['I']
    
    def name(self):
        return "KeyedPolybiusCipher"

    async def encrypt(self, plaintext: str) -> str:
        result = []
        for c in plaintext.upper():
            if c in self.positions:
                row, col = self.positions[c]
                result.append(f"{row+1}{col+1}")
            else:
                result.append(c)
        return ' '.join(result)

    async def decrypt(self, ciphertext: str) -> str:
        tokens = ciphertext.split(' ')
        result = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.isdigit() and len(token) == 2:
                row = int(token[0]) - 1
                col = int(token[1]) - 1
                result.append(self.grid[row][col])
                i += 1
            else:
                # Handle combined tokens (e.g., punctuation attached)
                non_digit_sequence = [token]
                i += 1
                while i < len(tokens) and not (tokens[i].isdigit() and len(tokens[i]) == 2):
                    non_digit_sequence.append(tokens[i])
                    i += 1
                result.append(' '.join(non_digit_sequence))
        return ''.join(result)
