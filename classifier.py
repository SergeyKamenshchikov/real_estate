import numpy as np

import streamlit as st
from openai import OpenAI

from math import exp
from concurrent.futures import ThreadPoolExecutor

# Параметры
model="gpt-5.4"
prob_limit = 0.95
max_workers = 18

# Инмцмация клиента
client = OpenAI(api_key=st.secrets["AITUNNEL_API_KEY"], base_url="https://api.aitunnel.ru/v1", timeout=60.0, max_retries=2,)

def extract_fact(text, problem_description, model=model, p_threshold=prob_limit,):

    messages = [
        {
            "role": "system",
            "content": (
                "Это запрос ресторана в Отдел Управления Недвижимостью головного офиса по вопросам помещения, отношений с арендатором и другим вопросам."
                "Определи, относится ли обращение к указанному типу.\n"
                f"Тип обращения: {problem_description}\n"
                "Ответь строго одной цифрой: 1 — да, 0 — нет."
            ),
        },
        {
            "role": "user",
            "content": f"Обращение:\n{text}",
        },
    ]

    r = client.chat.completions.create(model=model, messages=messages, temperature=0, logprobs=True, top_logprobs=10, max_completion_tokens=1,)

    choice = r.choices[0]

    if choice.logprobs is None or not choice.logprobs.content:

        raise ValueError(
            "Нет logprobs для ответа: "
            f"content={choice.message.content!r}, "
            f"finish_reason={choice.finish_reason!r}")

    first = choice.logprobs.content[0]
    candidates = first.top_logprobs or [first]

    p_yes = 0.0
    p_no = 0.0

    for t in candidates:

        tok = t.token.strip()

        if tok == "1":
            p_yes += exp(t.logprob)
        elif tok == "0":
            p_no += exp(t.logprob)

    total = p_yes + p_no

    if total == 0:
        raise ValueError(
            "В полученных токенах нет '1' или '0': "
            f"{[(t.token, t.logprob) for t in candidates]}")

    p_answer = p_yes / total

    return p_answer

def classify_text(text, problem_descriptions, model=model, p_threshold=prob_limit, max_workers=max_workers,):
    descriptions = list(problem_descriptions)

    if not descriptions:
        return "Others"

    probabilities = parallel_text_func(
        texts=descriptions,
        func=lambda description: extract_fact(
            text=text,
            problem_description=description,
            model=model,
        ),
        max_workers=max_workers,
    )

    best_description, best_probability = max(
        zip(descriptions, probabilities),
        key=lambda item: item[1],
    )

    return best_description.split if best_probability >= p_threshold else "Others"

def parallel_text_func(texts, func, max_workers=max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, texts))
