from pathlib import Path

import pandas as pd
import numpy as np

import streamlit as st
from classifier import classify_text, model, prob_limit 

st.set_page_config(page_title="Бот по вопросам недвижимости", page_icon="📝",)
st.html("""<h2 style="font-weight: 600;">Бот по вопросам недвижимости</h2>""")
# st.write("Введите текст обращения, будет определен тип:")

@st.cache_data
def load_categories():
  
    path = Path(__file__).parent / "Контекст для LLM.xlsx"
    df = pd.read_excel(path)

    titles = df["Название обращения"].fillna("").astype(str).str.strip()
    details = df["Краткое описание"].fillna("").astype(str).str.strip()

    mask = titles.ne("")
  
    descriptions = (titles[mask] + " | " + details[mask]).tolist()

    title_by_description = dict(
        zip(descriptions, titles[mask]))

    return descriptions, title_by_description


descriptions, title_by_description = load_categories()

with st.form("classify_form"):
  
    text = st.text_area("Введите текст запроса:", height=180, max_chars=5000, placeholder="Например: пришлите договор аренды ресторана",)
    submitted = st.form_submit_button("Анализ запроса")

if submitted:
    if not text.strip():
        st.warning("Введите текст обращения.")
    else:
        try:
            with st.spinner("Анализирую ваш запрос. Это займет 5-10 секунд..."):
                path = Path(__file__).resolve().parent / "Контекст для LLM.xlsx"

                if not path.is_file():
                    raise FileNotFoundError(f"Не найден файл: {path}")

                data = pd.read_excel(path, engine="openpyxl")

                column = "Краткое описание"
                if column not in data.columns:
                    raise ValueError(
                        f"Нет столбца '{column}'. "
                        f"Доступные столбцы: {data.columns.tolist()}"
                    )

                descriptions = (
                    data[column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                descriptions = descriptions[descriptions.ne("")].tolist()

                if not descriptions:
                    raise ValueError("Список описаний в Excel пуст.")

                result = classify_text(
                    text=text.strip(),
                    problem_descriptions=descriptions,
                )

            if result == "Others":
                st.warning("Не удалось однозначно определить тип обращения.")
            else:
                title = title_by_description.get(result, result)
                st.success(f"Ваш тип запроса: '{title}'")

        except Exception as e:
            st.error("Не удалось выполнить классификацию.")
            st.exception(e)
