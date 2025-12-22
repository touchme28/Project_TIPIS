import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

#Настройка страницы
st.set_page_config(
    page_title="Анализ качества воды",
    layout="wide"
)

#Заголовок приложения
st.title("Прогнозирование пригодности воды для питья")

#Сайдбар для навигации
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу:", ["Описание проекта", "Предсказание модели"])

#Загрузка модели
@st.cache_resource
def load_model():
    try:
        model = joblib.load('water_quality_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Файлы модели не найдены. Пожалуйста, обучите модель и сохраните файлы.")
        return None

model = load_model()

if page == "Описание проекта":
    st.header("О проекте")
    st.markdown("""
    ### Цель проекта
    Разработка модели машинного обучения для классификации воды на пригодную и непригодную для питья.
    
    ### Используемые данные
    Датасет содержит информацию о физико-химических параметрах воды:
    - pH значение
    - Жесткости (Hardness)
    - Растворенных твердых частиц (Solids)
    - Хлораминов (Chloramines)
    - Сульфатов (Sulfate)
    - Электропроводности (Conductivity)
    - Органического углерода (Organic carbon)
    - Тригалометанов (Trihalomethanes)
    - Мутности (Turbidity)
    
    ### Выбранная модель
    После сравнения нескольких моделей (логистическая регрессия, случайный лес, XGBoost, k-ближайших соседей) 
    была выбрана модель XGBoost с оптимизированными гиперпараметрами.
    
    ### Метрики качества лучшей модели
    - F1-score: ~0.5059
    - Precision: ~0.5079
    - Recall: ~0.5039
    - ROC-AUC: ~0.6320
    """)

elif page == "Предсказание модели":
    st.header("Предсказание пригодности воды")
    
    if model is None:
        st.warning("Модель не загружена. Пожалуйста, убедитесь, что файлы модели находятся в рабочей директории.")
    else:
        #Создание формы для ввода данных
        st.subheader("Введите параметры воды:")
        
        #Делим интерфейс на две колонки для удобства
        col1, col2 = st.columns(2)
        
        with col1:
            ph = st.slider("pH value", 0.0, 14.0, 7.0, 0.1)
            hardness = st.slider("Hardness", 47.0, 324.0, 150.0, 1.0)
            solids = st.slider("Solids (ppm)", 320.0, 61230.0, 20000.0, 100.0)
            chloramines = st.slider("Chloramines (ppm)", 1.0, 13.0, 7.0, 0.1)
            sulfate = st.slider("Sulfate (mg/L)", 129.0, 481.0, 300.0, 1.0)
            
        with col2:
            conductivity = st.slider("Conductivity (μS/cm)", 181.0, 755.0, 400.0, 1.0)
            organic_carbon = st.slider("Organic carbon (ppm)", 2.0, 28.0, 12.0, 0.1)
            trihalomethanes = st.slider("Trihalomethanes (μg/L)", 5.0, 124.0, 60.0, 1.0)
            turbidity = st.slider("Turbidity (NTU)", 1.0, 7.0, 3.5, 0.1)
        
        #Кнопка для предсказания
        if st.button("Анализировать качество воды"):
            feature_order = [
                'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
            ]
            
            input_data = np.array([[
                ph, hardness, solids, chloramines, sulfate, 
                conductivity, organic_carbon, trihalomethanes, turbidity
            ]])
            
            #Предсказание
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            # Отображение результатов
            st.subheader("Результаты анализа:")
            
            if prediction[0] == 1:
                st.success(f"Вода пригодна для питья с вероятностью {probability:.2%}")
            else:
                st.error(f"Вода непригодна для питья с вероятностью {(1-probability):.2%}")
            
            # Визуализация вероятности
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Непригодна', 'Пригодна']
            probabilities = [1-probability, probability]
            colors = ['salmon', 'skyblue']
            
            bars = ax.bar(categories, probabilities, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Вероятность')
            ax.set_title('Вероятность пригодности воды для питья')
            
            # Добавление значений над столбцами
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            st.pyplot(fig)