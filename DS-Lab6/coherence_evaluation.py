import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # отключаем предупреждения TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # отключаем OneDNN оптимизации

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # отключаем предупреждения TensorFlow

from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd


def calculate_coherence(model, texts, dictionary, corpus, coherence_type="c_v"):
    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        corpus=corpus,
        coherence=coherence_type
    )
    return coherence_model.get_coherence()


if __name__ == '__main__':
    try:
        if not os.path.exists("preprocessed_data.csv"):
            print("файл preprocessed_data.csv не найден!")
        else:
            data = pd.read_csv("preprocessed_data.csv")
            print("данные успешно загружены:")
            print(data.head())

            texts = [text.split() for text in data['processed_summary']]
            print("шаг 1: тексты подготовлены")

            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            print("шаг 2: создан словарь и корпус")

            lda_model = models.LdaModel(
                corpus,
                num_topics=5,
                id2word=dictionary,
                random_state=42,
                iterations=50,
                passes=10
            )
            print("шаг 3: LDA модель построена")

            # расчет когерентности
            print("начинается расчет когерентности LDA...")
            try:
                lda_coherence = calculate_coherence(lda_model, texts, dictionary, corpus)
                print(f"когерентность LDA: {lda_coherence}")
            except Exception as e:
                print("ошибка при расчете когерентности LDA:", e)

            # построение NMF модели
            print("начинается построение NMF модели...")
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                X = vectorizer.fit_transform(data['processed_summary'])
                nmf_model = NMF(n_components=5, random_state=42)
                nmf_model.fit_transform(X)
                print("NMF модель успешно построена")
            except Exception as e:
                print("ошибка при построении NMF модели:", e)

            # построение BERTopic модели
            print("начинается построение BERTopic модели...")
            try:
                bert_model = BERTopic(nr_topics=5)
                bert_topics, _ = bert_model.fit_transform(data['processed_summary'])
                print("BERTopic модель успешно построена")
            except Exception as e:
                print("ошибка при построении BERTopic модели:", e)

    except Exception as e:
        print("общая ошибка выполнения программы:", e)
