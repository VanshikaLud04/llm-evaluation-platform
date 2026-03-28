import ollama
import time


def generate_response(model: str, prompt: str, context: str = None, mode: str = "basic") -> dict:
   
    has_context = context is not None and context.strip() != ""

    if mode == "rag" and has_context:
        final_prompt = (
            "You are a factual assistant.\n"
            "Use ONLY the context below to answer. "
            "If the answer is not in the context, say exactly: I don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n\n"
            "Answer concisely in one sentence:"
        )

    elif mode == "strict" and has_context:
        final_prompt = (
            "Rules:\n"
            "1. Answer ONLY using the given context.\n"
            "2. Do NOT add any extra information.\n"
            "3. If the answer is not present, say exactly: I don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n\n"
            "Answer:"
        )

    elif mode == "cot":
        final_prompt = (
            f"Question: {prompt}\n\n"
            "Think step by step, then give your final answer.\n"
            "Format:\nReasoning: ...\nAnswer: ..."
        )

    elif mode == "verify" and has_context:
        final_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n\n"
            "First answer the question. Then check if your answer is supported by the context.\n"
            "Format:\nAnswer: ...\nSupported: Yes / No"
        )

    else:
        final_prompt = (
            f"Answer the following question concisely in one sentence.\n\n"
            f"Question: {prompt}\n\nAnswer:"
        )

    start_time = time.time()

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": final_prompt}],
        options={"temperature": 0.1}
    )

    latency = round(time.time() - start_time, 3)

    return {
        "answer": response["message"]["content"].strip(),
        "latency": latency,
        "model": model,
        "mode": mode,
    }