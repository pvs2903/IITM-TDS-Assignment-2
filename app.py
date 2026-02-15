import time
import hashlib
import math
from collections import OrderedDict
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import random

# ---------------------------
# CONFIG
# ---------------------------
MODEL_COST_PER_1M = 0.60
AVG_TOKENS = 800
TTL_SECONDS = 86400
MAX_CACHE_SIZE = 2000
SIMILARITY_THRESHOLD = 0.95

# ---------------------------
# APP INIT
# ---------------------------
app = FastAPI()

# LRU cache structure
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

def fake_embedding(text):
    random.seed(hash(text))
    return [random.random() for _ in range(10)]

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))
    return dot / (mag_a * mag_b)

def cleanup_expired():
    now = time.time()
    keys_to_delete = [
        key for key, val in cache.items()
        if now - val["timestamp"] > TTL_SECONDS
    ]
    for key in keys_to_delete:
        del cache[key]

def enforce_lru():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

def call_llm(query):
    time.sleep(1.2)  # simulate API latency
    return f"Support answer for: {query}"

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
    start = time.time()
    analytics["totalRequests"] += 1
    cleanup_expired()

    query = request.query
    key = md5_hash(query)

    # 1️⃣ Exact match
    if key in cache:
        analytics["cacheHits"] += 1
        cache.move_to_end(key)
        latency = int((time.time() - start) * 1000)
        return {
            "answer": cache[key]["response"],
            "cached": True,
            "latency": latency,
            "cacheKey": key
        }

    # 2️⃣ Semantic match
    query_embedding = fake_embedding(query)
    for stored_key, value in cache.items():
        sim = cosine_similarity(query_embedding, value["embedding"])
        if sim > SIMILARITY_THRESHOLD:
            analytics["cacheHits"] += 1
            analytics["totalTokensSaved"] += AVG_TOKENS
            latency = int((time.time() - start) * 1000)
            return {
                "answer": value["response"],
                "cached": True,
                "latency": latency,
                "cacheKey": stored_key
            }

    # 3️⃣ Cache miss
    analytics["cacheMisses"] += 1
    response = call_llm(query)

    cache[key] = {
        "response": response,
        "embedding": query_embedding,
        "timestamp": time.time()
    }

    enforce_lru()

    latency = int((time.time() - start) * 1000)

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
    tokens_saved = hits * AVG_TOKENS
    cost_savings = (tokens_saved / 1_000_000) * MODEL_COST_PER_1M
    savings_percent = hit_rate * 100

    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total,
        "cacheHits": hits,
        "cacheMisses": misses,
        "cacheSize": len(cache),
        "costSavings": round(cost_savings, 2),
        "savingsPercent": round(savings_percent, 2),
        "strategies": [
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    }
