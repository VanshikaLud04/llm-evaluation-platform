import ollama
import time


def generate_response(model, prompt, context=None, mode="basic"):

    start_time = time.time()

    if mode == "basic":
        final_prompt = prompt

    elif mode == "rag" and context is not None and context.strip() != "":
        final_prompt = f"""You are a factual assistant.

        Use ONLY the context below to answer.

        Context:
        {context}

        Question:
        {prompt}

        If the answer is not in the context, say "I don't know".
        """

    elif mode == "strict" and context is not None and context.strip() != "":
        final_prompt = f"""You MUST follow rules strictly.

        Rules:
        1. Answer ONLY using the given context.
        2. Do NOT add extra information.
        3. If answer is not present, say exactly: "I don't know".

        Context:
        {context}

        Question:
        {prompt}
        """

    elif mode == "cot":
        final_prompt = f"""Answer step by step.

        Question:
        {prompt}

        Explain reasoning before final answer.
        """

    elif mode == "verify" and context is not None and context.strip() != "":
        final_prompt = f"""Check whether the answer is supported by the context.

            Context:
            {context}

            Question:
            {prompt}

            First generate answer, then verify if it is supported.
            Respond in format:
            Answer: ...
            Supported: Yes/No
            """

    else:
        final_prompt = prompt

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": final_prompt}
        ]
    )

    end_time = time.time()

    return {
        "answer": response["message"]["content"],
        "latency": end_time - start_time,
        "model": model,
        "mode": mode
    }