from retriever import retrieve
from generator import generate


def answer(question):
    retrieved_chunks = retrieve(question, k=5)
    context = "\n".join(set(retrieved_chunks))
    response = generate(context, question)

    return response


if __name__ == "__main__":
    q = "What is Redis?"

    print(answer(q))
