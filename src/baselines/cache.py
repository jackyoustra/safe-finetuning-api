# baselines/cache.py
import json
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache

class CacheSingleton:
    _instances: Dict[Path, 'CacheSingleton'] = {}
    
    def __new__(cls, cache_file: Path):
        if cache_file not in cls._instances:
            instance = super().__new__(cls)
            instance.cache_file = cache_file
            instance.cache: Dict[str, Any] = {}
            # Load existing cache from file
            if cache_file.exists():
                with cache_file.open('r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            instance.cache[entry['key']] = entry['value']
                        except json.JSONDecodeError:
                            continue
            cls._instances[cache_file] = instance
        return cls._instances[cache_file]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache and append to file."""
        self.cache[key] = value
        # Append to file in JSONL format
        with self.cache_file.open('a') as f:
            json.dump({'key': key, 'value': value}, f)
            f.write('\n')

# Global cache getter
@lru_cache(maxsize=None)
def get_cache(cache_file: Path) -> CacheSingleton:
    """Get or create cache singleton for a given file."""
    return CacheSingleton(cache_file)
