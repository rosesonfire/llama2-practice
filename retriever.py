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

def retrieve(query, k=3):
    query_vec = embedder.encode([query])
    query_vec = np.array(query_vec).astype("float32")

    distances, indices = index.search(query_vec, k)

    results = [chunks[i] for i in indices[0]]

    return results
