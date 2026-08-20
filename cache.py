"""
Two caching layers for production RAG. Both matter, for different reasons.

1. QUERY CACHE (exact-match): question -> final answer.
   Skips the ENTIRE pipeline on a repeat question. Biggest possible win, but only
   fires on exact repeats. Real systems normalize (lowercase/strip) to widen hits.

2. EMBEDDING CACHE: text -> vector.
   Skips re-embedding. Useful because the same query text often recurs, and at index
   time it avoids re-embedding unchanged documents on a rebuild.

3. SEMANTIC CACHE (mentioned, not implemented here): embed the query, and if it's
   within a similarity threshold of a previously-seen query, reuse that answer.
   Catches paraphrases ("Q4 revenue?" vs "what was revenue in Q4?") that exact-match
   misses. Tradeoff: a too-loose threshold serves a wrong cached answer.
"""
import hashlib, json, os

class DiskCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        self.hits = 0
        self.misses = 0

    def _key(self, s: str) -> str:
        # normalize so trivial formatting differences still hit
        norm = s.strip().lower()
        return hashlib.sha256(norm.encode()).hexdigest()[:16]

    def get(self, s: str):
        k = self._key(s)
        if k in self.data:
            self.hits += 1
            return self.data[k]
        self.misses += 1
        return None

    def set(self, s: str, value):
        self.data[self._key(s)] = value

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f)

    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / total if total else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": rate}
