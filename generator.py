from llama_cpp import Llama

llm = Llama(model_path="models/llama-2-7b-chat.Q4_K_M.gguf", n_threads=24, n_ctx=2048)


def generate(context, question):
    prompt = f"""You are a helpful assistant.
Use ONLY the context below to answer the question. If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    output = llm(prompt, max_tokens=300, stop=["\n\n"])
    return output["choices"][0]["text"].strip()
