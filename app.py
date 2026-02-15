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

# Enable CORS (fixes OPTIONS 405 issue)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LRU cache storage
cache = OrderedDict()

# Analytics tracking
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

def cosine_similarity(vec1, vec2):
    dot = sum(a*b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a*a for a in vec1))
    mag2 = math.sqrt(sum(b*b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

def remove_expired_entries():
    now = time.time()
    expired_keys = [
        key for key, value in cache.items()
        if now - value["timestamp"] > TTL_SECONDS
    ]
    for key in expired_keys:
        del cache[key]

def enforce_lru_policy():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

def call_llm(query):
    # Simulate API latency
    time.sleep(1.2)
    return f"Customer support answer for: {query}"

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
def process_query(request: QueryRequest):
    start_time = time.time()
    analytics["totalRequests"] += 1
    remove_expired_entries()

    query = request.query
    cache_key = md5_hash(query)

    # 1️⃣ Exact Match Cache
    if cache_key in cache:
        analytics["cacheHits"] += 1
        analytics["totalTokensSaved"] += AVG_TOKENS
        cache.move_to_end(cache_key)  # LRU update

        latency = int((time.time() - start_time) * 1000)

        return {
            "answer": cache[cache_key]["response"],
            "cached": True,
            "latency": latency,
            "cacheKey": cache_key
        }

    # 2️⃣ Semantic Similarity Cache
    query_embedding = generate_fake_embedding(query)

    for key, value in cache.items():
        similarity = cosine_similarity(query_embedding, value["embedding"])
        if similarity > SIMILARITY_THRESHOLD:
            analytics["cacheHits"] += 1
            analytics["totalTokensSaved"] += AVG_TOKENS
            cache.move_to_end(key)

            latency = int((time.time() - start_time) * 1000)

            return {
                "answer": value["response"],
                "cached": True,
                "latency": latency,
                "cacheKey": key
            }

    # 3️⃣ Cache Miss → Call LLM
    analytics["cacheMisses"] += 1
    response = call_llm(query)

    cache[cache_key] = {
        "response": response,
        "embedding": query_embedding,
        "timestamp": time.time()
    }

    enforce_lru_policy()

    latency = int((time.time() - start_time) * 1000)

    return {
        "answer": response,
        "cached": False,
        "latency": latency,
        "cacheKey": cache_key
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
