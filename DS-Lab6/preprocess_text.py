import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

# загрузка данных для NLTK
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # токенизация текста
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# пример использования
if __name__ == "__main__":
    data = pd.read_csv("collected_data.csv")
    data['processed_summary'] = data['summary'].apply(preprocess_text)
    data.to_csv("preprocessed_data.csv", index=False)
    print("данные успешно предобработаны и сохранены в 'preprocessed_data.csv'")
