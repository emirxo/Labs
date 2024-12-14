import streamlit as st
import pyLDAvis.gensim_models
import pandas as pd
from gensim import corpora, models
from bertopic import BERTopic
import pyLDAvis
import os

# вывод текущей рабочей директории для отладки
st.write(f"Current working directory: {os.getcwd()}")

# функция для визуализации LDA
def visualize_lda(lda_model, corpus, dictionary):
    return pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

# загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\User\Desktop\Labs\DS-Lab6\preprocessed_data.csv")

# основной интерфейс Streamlit
st.title("интерактивная визуализация тематического моделирования")
data = load_data()

# выбор модели
model_option = st.selectbox("выберите модель для анализа", ["LDA", "NMF", "BERTopic"])

# выбор количества тем
n_topics = st.slider("количество тем", 2, 20, 5)

# загрузка и обработка данных
processed_summaries = data['processed_summary']
texts = [text.split() for text in processed_summaries]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

if model_option == "LDA":
    lda_model = models.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, random_state=42)
    st.subheader("визуализация LDA")
    vis_data = visualize_lda(lda_model, corpus, dictionary)
    pyLDAvis_html = pyLDAvis.prepared_data_to_html(vis_data)
    st.components.v1.html(pyLDAvis_html, width=1300, height=800)

elif model_option == "NMF":
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['processed_summary'])
    nmf_model = NMF(n_components=n_topics, random_state=42)
    nmf_topics = nmf_model.fit_transform(X)

    st.subheader("темы (NMF)")
    for i, topic in enumerate(nmf_model.components_):
        st.write(f"тема {i + 1}:")
        st.write(", ".join([vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]))

elif model_option == "BERTopic":
    bert_model = BERTopic(nr_topics=n_topics)
    topics, probs = bert_model.fit_transform(data['processed_summary'])

    st.subheader("Темы (BERTopic)")
    st.write(bert_model.get_topic_info())

# Завершение
st.write("Выберите модель и параметры для изучения тем.")
