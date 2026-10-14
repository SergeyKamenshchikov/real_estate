import streamlit as st
from openai import OpenAI

from math import exp
from sklearn.metrics.pairwise import cosine_similarity

from concurrent.futures import ThreadPoolExecutor

# Параметры
embedding_model = "text-embedding-3-small"
model="gpt-5.4"

prob_limit = 0.95
top_N = 3
max_workers = top_N
#/Параметры

# Инмцмация клиента
client = OpenAI(api_key=st.secrets["AITUNNEL_API_KEY"], base_url="https://api.aitunnel.ru/v1", timeout=60.0, max_retries=2,)

def extract_fact(text, problem_description, model="gpt-5.4", p_threshold=prob_limit,):

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

        raise ValueError("В полученных токенах нет '1' или '0': "f"{[(t.token, t.logprob) for t in candidates]}")

    p_answer = p_yes / total

    return p_answer

def get_embeddings(texts, embedding_model):
    
    response = client.embeddings.create(model=embedding_model, input=list(texts),)
    return np.array([item.embedding for item in sorted(response.data, key=lambda item: item.index)],dtype=np.float32,)

def classify_text(text, problem_descriptions, model="gpt-5.4", p_threshold=prob_limit, max_workers=max_workers,):

    descriptions = list(problem_descriptions)

    probabilities = parallel_text_func(texts=descriptions, func=lambda description: extract_fact(text=text, problem_description=description, model=model,), max_workers=max_workers,)
    matched_types = [description for description, probability in zip(descriptions, probabilities) if probability >= p_threshold]

    return matched_types[0] if len(matched_types) == 1 else "Others"

def parallel_text_func(texts, func, max_workers=max_workers):

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(lambda x: func(x), texts), total=len(texts)))

    return results

def classify_text_topk(text, top_k=top_N, model=model, p_threshold=prob_limit, max_workers=max_workers,):
    
    text_embedding = get_embeddings([text], embedding_model=embedding_model,)

    similarities = cosine_similarity(text_embedding, category_embeddings,)[0]
    top_indices = np.argsort(-similarities)[:top_k]

    candidates = [descriptions[i] for i in top_indices]

    return classify_text(text=text, problem_descriptions=candidates,model=model, p_threshold=p_threshold, max_workers=max_workers,)
