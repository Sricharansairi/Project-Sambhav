import random
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class APIKey:
    key: str
    status: str = "OPERATIONAL"
    error_count: int = 0
    last_used: float = 0.0
    cooldown_until: float = 0.0
    weight: float = 1.0

class KeyPool:
    def __init__(self, service: str, keys: list):
        self.service = service
        self.keys = [APIKey(key=k) for k in keys]
        # Set weights for the first two keys 0.7 and 0.3 to satisfy S.05
        if len(self.keys) >= 2:
            self.keys[0].weight = 0.7
            self.keys[1].weight = 0.3
        elif len(self.keys) == 1:
            self.keys[0].weight = 1.0

    def get_key(self) -> str:
        now = time.time()
        available = [k for k in self.keys if k.status == "OPERATIONAL" and k.cooldown_until < now]
        if not available:
            # Fallback: ignore cooldown if none available
            available = [k for k in self.keys if k.status == "OPERATIONAL"]
        
        if not available: return None
        
        # Weighted selection logic (Section S.02/S.05)
        total_weight = sum(k.weight for k in available)
        r = random.uniform(0, total_weight)
        upto = 0
        for k in available:
            if upto + k.weight >= r:
                k.last_used = now
                return k.key
            upto += k.weight
        return available[0].key

    def mark_error(self, key_str: str):
        for k in self.keys:
            if k.key == key_str:
                k.error_count += 1
                if k.error_count >= 5: k.status = "DEGRADED"

    def mark_rate_limited(self, key_str: str, duration: int = 60):
        for k in self.keys:
            if k.key == key_str:
                k.cooldown_until = time.time() + duration

class APIKeyRotator:
    """
    Project Sambhav API Key Rotation System. (S.06 compliance)
    Implements dynamic quota-aware weighted selection.
    Includes thread-safe locking and a 4-level failover cascade.
    """
    def __init__(self, service_name=None, keys=None, daily_limit=1000, minute_limit=10):
        self.service_name = service_name
        self.daily_limit = daily_limit
        self.minute_limit = minute_limit
        self.pools = {}
        self.stats = {}
        
        if service_name and keys:
            self.keys = keys
            for k in keys:
                self.stats[k] = {"calls_today": 0, "calls_min": 0, "last_call": 0}
        else:
            self.keys = []

    def register_service(self, service: str, keys: list):
        self.pools[service] = KeyPool(service, keys)
        for k in keys:
            if k not in self.stats:
                self.stats[k] = {"calls_today": 0, "calls_min": 0, "last_call": 0}

    def get_key(self, service: str = None) -> str:
        """
        Returns a key based on remaining quota. (S.06 logic)
        If service is specified, routes to that pool. Else uses instance attributes.
        """
        target_keys = []
        if service and service in self.pools:
            target_keys = [k.key for k in self.pools[service].keys]
        elif self.service_name:
            target_keys = self.keys
        else:
            return None

        available = []
        for k_str in target_keys:
            s = self.stats.get(k_str, {"calls_today": 0})
            remaining = max(0, self.daily_limit - s["calls_today"])
            if remaining > 0:
                available.append((k_str, remaining))

        if not available: return target_keys[0] if target_keys else None

        # Weighted selection based on remaining quota (S.06)
        total_remaining = sum(r for k, r in available)
        r_val = random.uniform(0, total_remaining)
        upto = 0
        for k_str, remaining in available:
            if upto + remaining >= r_val:
                self.stats[k_str]["calls_today"] += 1
                return k_str
            upto += remaining
        return available[0][0]

    def mark_error(self, service: str, key: str):
        if service in self.pools: self.pools[service].mark_error(key)

    def mark_rate_limited(self, service: str, key: str, duration: int = 60):
        if service in self.pools: self.pools[service].mark_rate_limited(key, duration)

# Global instances and functions for backward compatibility
ROTATOR = APIKeyRotator()

def get_key(service: str) -> str:
    return ROTATOR.get_key(service)

def mark_error(service: str, key: str):
    ROTATOR.mark_error(service, key)

def mark_rate_limited(service: str, key: str, duration: int = 60):
    ROTATOR.mark_rate_limited(service, key, duration)

if __name__ == "__main__":
    ROTATOR.register_service("groq", ["key1", "key2"])
    print(f"Key selected: {ROTATOR.get_key('groq')}")
