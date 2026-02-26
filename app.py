from fastapi import FastAPI, Request
from retriever import retrieve
from generator import generate

app = FastAPI()


@app.get("/")
async def home():
    return {"message": "LLaMA RAG API Running"}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")
    docs = retrieve(query)

    if not docs:
        return {"answer": "I don't know."}

    context = "\n".join(docs)
    answer = generate(context, query)
    return {"answer": answer}
