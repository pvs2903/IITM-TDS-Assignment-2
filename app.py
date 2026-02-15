import time
import hashlib
import math
import random
from collections import OrderedDict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_COST_PER_1M = 0.60
AVG_TOKENS = 800
TTL_SECONDS = 86400  # 24 hours
MAX_CACHE_SIZE = 2000
SIMILARITY_THRESHOLD = 0.95

# ---------------------------
# APP INITIALIZATION
# ---------------------------
app = FastAPI()

# Enable CORS (Fixes OPTIONS 405 issue)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory LRU cache
cache = OrderedDict()

# Analytics tracking
analytics = {
    "totalRequests": 0,
    "cacheHits": 0,
    "cacheMisses": 0,
    "totalTokensSaved": 0
}

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def md5_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def generate_fake_embedding(text):
    random.seed(hash(text))
    return [random.random() for _ in range(20)]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0
    return dot / (mag_a * mag_b)

def cleanup_expired():
    now = time.time()
    expired_keys = [
        key for key, value in cache.items()
        if now - value["timestamp"] > TTL_SECONDS
    ]
    for key in expired_keys:
        del cache[key]

def enforce_lru():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

# ---------------------------
# REQUEST MODEL
# ---------------------------

class QueryRequest(BaseModel):
    query: str
    application: str

# ---------------------------
# MAIN QUERY ENDPOINT
# ---------------------------

@app.post("/")
def handle_query(request: QueryRequest):
    start_time = time.time()
    analytics["totalRequests"] += 1
    cleanup_expired()

    query = request.query
    key = md5_hash(query)

    # 1️⃣ Exact Match Cache
    if key in cache:
        analytics["cacheHits"] += 1
        analytics["totalTokensSaved"] += AVG_TOKENS
        cache.move_to_end(key)

        latency = max(1, int((time.time() - start_time) * 1000))

        return {
            "answer": cache[key]["response"],
            "cached": True,
            "latency": latency,
            "cacheKey": key
        }

    # 2️⃣ Semantic Match Cache
    query_embedding = generate_fake_embedding(query)

    for stored_key, value in cache.items():
        similarity = cosine_similarity(query_embedding, value["embedding"])
        if similarity > SIMILARITY_THRESHOLD:
            analytics["cacheHits"] += 1
            analytics["totalTokensSaved"] += AVG_TOKENS
            cache.move_to_end(stored_key)

            latency = max(1, int((time.time() - start_time) * 1000))

            return {
                "answer": value["response"],
                "cached": True,
                "latency": latency,
                "cacheKey": stored_key
            }

    # 3️⃣ Cache Miss → Simulate LLM Call (REAL DELAY)
    analytics["cacheMisses"] += 1

    # Simulate real API latency (~1200ms)
    time.sleep(1.2)

    response = f"Customer support answer for: {query}"

    cache[key] = {
        "response": response,
        "embedding": query_embedding,
        "timestamp": time.time()
    }

    enforce_lru()

    latency = max(1, int((time.time() - start_time) * 1000))

    return {
        "answer": response,
        "cached": False,
        "latency": latency,
        "cacheKey": key
    }

# ---------------------------
# ANALYTICS ENDPOINT
# ---------------------------

@app.get("/analytics")
def get_analytics():
    total = analytics["totalRequests"]
    hits = analytics["cacheHits"]
    misses = analytics["cacheMisses"]

    hit_rate = hits / total if total else 0
    savings_percent = hit_rate * 100

    tokens_saved = analytics["totalTokensSaved"]
    cost_savings = (tokens_saved / 1_000_000) * MODEL_COST_PER_1M

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total,
        "cacheHits": hits,
        "cacheMisses": misses,
        "cacheSize": len(cache),
        "costSavings": round(cost_savings, 2),
        "savingsPercent": round(savings_percent, 2),
        "strategies": [
            "exact match caching",
            "semantic similarity caching",
            "LRU eviction policy",
            "TTL-based expiration"
        ]
    }
