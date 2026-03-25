from llama_cpp import Llama

llm = Llama(model_path="models/llama-2-7b-chat.Q4_K_M.gguf", n_threads=24, n_ctx=2048)


def generate(context, question):
    prompt = f"""<s>[INST] <<SYS>>
You are a question-answering system.

Answer using ONLY the context provided.

Rules:
- Do not add extra words
- Do not explain
- Do not add emojis
- Return only the exact answer sentence

If the answer is not present, say "I don't know".
<</SYS>>

Context:
{context}

Question:
{question}
[/INST]
"""
    output = llm(prompt, max_tokens=100, stop=["</s>"])

    return output["choices"][0]["text"].strip()
