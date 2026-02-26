import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = "models/all-MiniLM-L6-v2"
INDEX_PATH = "llama2-practice.faiss"
META_PATH = "metadata.pkl"

embedder = SentenceTransformer(MODEL_PATH)

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    chunks = pickle.load(f)


def retrieve(query, k=3, threshold=0.5):
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if dist < threshold:   # adjust based on your index
            results.append(chunks[idx])

    return results
