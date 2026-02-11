from llama_cpp import Llama

llm = Llama(model_path="models/llama-2-7b-chat.Q4_K_M.gguf", n_threads=24, n_ctx=2048)


def generate(context, question):
    prompt = f"""
You are a helpful assistant.

Context:
{context}

Question:
{question}

Answer using only the context above.
"""
    output = llm(prompt, max_tokens=300)

    return output["choices"][0]["text"]
