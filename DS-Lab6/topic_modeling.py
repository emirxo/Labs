from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd

def topic_modeling(text_data, n_topics=5):
    texts = [text.split() for text in text_data]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # LDA
    lda_model = models.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, random_state=42)
    lda_model.save("lda_model.gensim")  # сохранение модели

    # NMF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_data)
    nmf_model = NMF(n_components=n_topics, random_state=42)
    nmf_topics = nmf_model.fit_transform(X)

    return lda_model, nmf_model, vectorizer, nmf_topics, corpus, dictionary

# пример использования
if __name__ == "__main__":
    data = pd.read_csv("preprocessed_data.csv")
    processed_summaries = data['processed_summary']
    lda_model, nmf_model, vectorizer, nmf_topics, corpus, dictionary = topic_modeling(processed_summaries)
    print("темы успешно смоделированы и LDA-модель сохранена.")