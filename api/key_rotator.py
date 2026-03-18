import os, time, logging, random
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

def _load_keys(prefix: str, count: int = 15) -> list:
    keys = []
    for i in range(1, count+1):
        k = os.getenv(f"{prefix}_{i}")
        if k and k not in keys: keys.append(k)
    k = os.getenv(prefix)
    if k and k not in keys: keys.append(k)
    return keys

class KeyPool:
    def __init__(self, service: str, keys: list, daily_limit: int):
        self.service     = service
        self.daily_limit = daily_limit
        self.keys        = {k: {"calls":0,"errors":0,
                                "last_used":0,"healthy":True}
                            for k in keys}

    def get_key(self) -> str:
        healthy = {k:v for k,v in self.keys.items()
                   if v["healthy"] and v["calls"] < self.daily_limit}
        if not healthy:
            logger.warning(f"{self.service}: all keys exhausted, resetting...")
            for k in self.keys:
                self.keys[k]["calls"]   = 0
                self.keys[k]["healthy"] = True
            healthy = self.keys
        if not healthy:
            raise RuntimeError(f"{self.service}: no keys available")
        keys    = list(healthy.keys())
        weights = [1/(v["calls"]+1) for v in healthy.values()]
        total   = sum(weights)
        probs   = [w/total for w in weights]
        chosen  = random.choices(keys, weights=probs, k=1)[0]
        self.keys[chosen]["calls"]     += 1
        self.keys[chosen]["last_used"]  = time.time()
        return chosen

    def mark_error(self, key: str):
        if key in self.keys:
            self.keys[key]["errors"] += 1
            if self.keys[key]["errors"] >= 3:
                self.keys[key]["healthy"] = False
                logger.warning(f"{self.service}: key ...{key[-6:]} marked unhealthy")

    def mark_rate_limited(self, key: str):
        if key in self.keys:
            self.keys[key]["calls"] = self.daily_limit
            logger.warning(f"{self.service}: key ...{key[-6:]} rate limited")

    def status(self) -> dict:
        healthy_count = sum(1 for v in self.keys.values() if v["healthy"])
        total_calls   = sum(v["calls"] for v in self.keys.values())
        capacity      = self.daily_limit * len(self.keys)
        return {
            "service":       self.service,
            "total_keys":    len(self.keys),
            "healthy_keys":  healthy_count,
            "total_calls":   total_calls,
            "capacity":      capacity,
            "capacity_used": f"{total_calls}/{capacity}",
            "pct_used":      round(total_calls/max(capacity,1)*100,1),
            "status":        "OPERATIONAL" if healthy_count > 0 else "DEGRADED",
        }

# ── Initialize all pools ──────────────────────────────────────
POOLS = {
    "groq":          KeyPool("groq",          _load_keys("GROQ_API_KEY"),          14400),
    "nvidia":        KeyPool("nvidia",         _load_keys("NVIDIA_API_KEY"),        1000),
    "openrouter":    KeyPool("openrouter",     _load_keys("OPENROUTER_API_KEY"),    1000),
    "xai":           KeyPool("xai",            _load_keys("XAI_API_KEY"),           1000),
    "newsapi":       KeyPool("newsapi",         _load_keys("NEWS_API_KEY"),          100),
    "gnews":         KeyPool("gnews",           _load_keys("GNEWS_API_KEY"),         100),
    "guardian":      KeyPool("guardian",        _load_keys("GUARDIAN_API_KEY"),      500),
    "assemblyai":    KeyPool("assemblyai",      _load_keys("ASSEMBLYAI_API_KEY"),    300),
    "cohere":        KeyPool("cohere",          _load_keys("COHERE_API_KEY"),        1000),
}

def get_key(service: str) -> str:
    if service not in POOLS:
        raise ValueError(f"Unknown service: {service}. Available: {list(POOLS.keys())}")
    return POOLS[service].get_key()

def mark_error(service: str, key: str):
    if service in POOLS: POOLS[service].mark_error(key)

def mark_rate_limited(service: str, key: str):
    if service in POOLS: POOLS[service].mark_rate_limited(key)

def all_status() -> dict:
    return {svc: pool.status() for svc, pool in POOLS.items()}

def system_status() -> str:
    statuses = [p.status()["status"] for p in POOLS.values()]
    if all(s == "OPERATIONAL" for s in statuses): return "OPERATIONAL"
    elif any(s == "DEGRADED"  for s in statuses): return "CAUTION"
    return "DEGRADED"

if __name__ == "__main__":
    print("\n🔑 KEY ROTATOR STATUS\n" + "="*60)
    for svc, info in all_status().items():
        st  = "✅" if info["status"] == "OPERATIONAL" else "🔴"
        bar = "█" * info["total_keys"]
        print(f"  {st} {svc:<14} {info['total_keys']:>2} keys  {bar}")
    print(f"\n  System: {system_status()}")
    print("="*60 + "\n")
