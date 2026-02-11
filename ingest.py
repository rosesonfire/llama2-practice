import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

MODEL_PATH = "models/all-MiniLM-L6-v2"
INDEX_PATH = "llama2-practice.faiss"
META_PATH = "metadata.pkl"

embedder = SentenceTransformer(MODEL_PATH)


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_index(documents):
    all_chunks = []

    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    embeddings = embedder.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(all_chunks, f) # Note: pickle is not safe

    print("Index built successfully.")


if __name__ == "__main__":
    docs = [
        "Redis is an in-memory data store used as a database and cache.",
        "FastAPI is a modern Python framework for building APIs.",
    ]

    build_index(docs)
