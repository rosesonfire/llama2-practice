from retriever import retrieve
from generator import generate


def answer(question):
    retrieved_chunks = retrieve(question)
    context = "\n".join(retrieved_chunks)
    response = generate(context, question)

    return response


if __name__ == "__main__":
    q = "What is Redis?"

    print(answer(q))
