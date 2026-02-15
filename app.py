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
TTL_SECONDS = 86400
MAX_CACHE_SIZE = 2000
SIMILARITY_THRESHOLD = 0.95

CACHE_HIT_LATENCY = 10      # ms (very fast)
CACHE_MISS_LATENCY = 1200   # ms (slow LLM call)

# ---------------------------
# APP INIT
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = OrderedDict()

analytics = {
    "totalRequests": 0,
    "cacheHits": 0,
    "cacheMisses": 0,
    "totalTokensSaved": 0
}

# ---------------------------
# UTILITIES
# ---------------------------

def md5_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def generate_fake_embedding(text):
    random.seed(hash(text))
    return [random.random() for _ in range(20)]

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(y*y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0
    return dot / (mag_a * mag_b)

def cleanup_expired():
    now = time.time()
    expired = [
        key for key, val in cache.items()
        if now - val["timestamp"] > TTL_SECONDS
    ]
    for key in expired:
        del cache[key]

def enforce_lru():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

def simulate_llm_call():
    time.sleep(CACHE_MISS_LATENCY / 1000)

# ---------------------------
# REQUEST MODEL
# ---------------------------

class QueryRequest(BaseModel):
    query: str
    application: str

# ---------------------------
# MAIN ENDPOINT
# ---------------------------

@app.post("/")
def handle_query(request: QueryRequest):
    analytics["totalRequests"] += 1
    cleanup_expired()

    query = request.query
    key = md5_hash(query)

    # 1️⃣ Exact Match
    if key in cache:
        analytics["cacheHits"] += 1
        analytics["totalTokensSaved"] += AVG_TOKENS
        cache.move_to_end(key)

        return {
            "answer": cache[key]["response"],
            "cached": True,
            "latency": CACHE_HIT_LATENCY,
            "cacheKey": key
        }

    # 2️⃣ Semantic Match
    query_embedding = generate_fake_embedding(query)
    for stored_key, value in cache.items():
        sim = cosine_similarity(query_embedding, value["embedding"])
        if sim > SIMILARITY_THRESHOLD:
            analytics["cacheHits"] += 1
            analytics["totalTokensSaved"] += AVG_TOKENS
            cache.move_to_end(stored_key)

            return {
                "answer": value["response"],
                "cached": True,
                "latency": CACHE_HIT_LATENCY,
                "cacheKey": stored_key
            }

    # 3️⃣ Cache Miss
    analytics["cacheMisses"] += 1
    simulate_llm_call()

    response = f"Customer support answer for: {query}"

    cache[key] = {
        "response": response,
        "embedding": query_embedding,
        "timestamp": time.time()
    }

    enforce_lru()

    return {
        "answer": response,
        "cached": False,
        "latency": CACHE_MISS_LATENCY,
        "cacheKey": key
    }

# ---------------------------
# ANALYTICS
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
