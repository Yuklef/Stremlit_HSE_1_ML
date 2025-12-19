import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "linear_model.pkl"


@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model
try:
    MODEL = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

# --- Основной интерфейс ---
st.title("Предсказание стоимости автомобиля")

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)

st.header("Основная информация о данных")

st.write(df.head())
st.write(df.describe())
st.write(df.tail())

st.header("Визуализация EDA")
st.subheader("Тепловая карта")

df_train_num = df.select_dtypes(include=['number'])
df_train_corr = df.corr()
heat = sns.heatmap(df_train_corr, annot=True, vmax=1, vmin=-1, cmap="Blues")

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df_train_corr, annot=True, vmax=1, vmin=-1, cmap="Blues", ax=ax)

st.pyplot(fig)

st.subheader("Матрица диаграмм рассеивания")
st.pyplot(sns.pairplot(data=df).fig)


st.subheader("Яндекс дашборд")

profile = ProfileReport(df, title="Яндекс дашборд")

profile_html = profile.to_html()

components.html(profile_html, height=1000, scrolling=True) # Chat GPT помог с этим блоком с html

st.subheader("Гистограмма важности признаков")
coef_df = pd.DataFrame({
    'Признаки': df.columns,
    'Коэффициенты': MODEL.coef_})


fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Признаки', y='Коэффициенты', data=coef_df)
st.pyplot(fig)
