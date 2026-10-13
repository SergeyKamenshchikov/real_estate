from pathlib import Path

import pandas as pd
import streamlit as st

from classifier import classify_text

st.set_page_config(page_title="Классификатор обращений", page_icon="📝",)
st.subheader("Классификатор обращений")
st.write("Введите текст обращения, будет определен тип:")


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
  
    text = st.text_area("Текст обращения", height=180, max_chars=5000, placeholder="Например: пришлите договор аренды ресторана",)
    submitted = st.form_submit_button("Определить тип")


if submitted:
  
    if not text.strip():
        st.warning("Введите текст обращения.")
      
    else:
        try:
            with st.spinner("Определяем тип обращения..."):
                result = classify_text(
                    text=text.strip(),
                    problem_descriptions=descriptions,
                    p_threshold=0.95,
                    max_workers=5,)

            if result == "Others":
                st.warning(
                    "Не удалось однозначно определить тип обращения.")
            else:
                title = title_by_description.get(result, result)
                st.success(f"Тип обращения: {title}")

        except Exception:
            st.error(
                "Не удалось выполнить классификацию. "
                "Попробуйте позже.")
