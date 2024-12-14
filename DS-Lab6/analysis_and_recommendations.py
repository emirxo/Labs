import pandas as pd

# оценка модели и вывод ошибок
def analyze_topics(lda_coherence, bert_model, nmf_model):
    print("анализ когерентности тем:")
    print(f"- LDA когерентность: {lda_coherence}")

    # анализ BERTopic
    print("- BERTopic анализ:")
    if bert_model is not None:
        try:
            print(bert_model.get_topic_info())
            if bert_model.get_topic_freq().shape[0] < 5:
                print("- BERTopic показывает малое количество значимых тем. рекомендуется:")
                print("  - увеличить объем данных для обучения.")
                print("  - использовать более мощную модель эмбеддингов (например, Sentence-BERT).")
        except Exception as e:
            print("ошибка при анализе BERTopic:", e)
    else:
        print("- BERTopic модель не была предоставлена.")

    # анализ NMF
    print("- NMF: визуализация и интерпретация тем.")
    if nmf_model is not None:
        try:
            components = nmf_model.components_
            for topic_idx, topic in enumerate(components):
                print(f"Тема {topic_idx + 1}:")
                print(", ".join([str(i) for i in topic[:10]]))  # Замените на реальные слова при работе с TF-IDF
        except Exception as e:
            print("ошибка при анализе NMF:", e)
    else:
        print("- NMF модель не была предоставлена.")

    # рекомендации по улучшению
    if lda_coherence < 0.5:
        print("- LDA требует улучшения:")
        print("  - очистите данные или увеличьте объем данных.")
        print("  - экспериментируйте с количеством тем.")

    print("- для NMF рекомендуется:")
    print("  - оптимизировать параметры модели (например, количество тем).")
    print("  - улучшить выбор словаря (TF-IDF) и фильтрацию стоп-слов.")


# пример данных для анализа
# замените эти примеры на реальные объекты
lda_coherence = 0.42  # примерное значение (замените на фактическое)
bert_model = None  # замените на объект BERTopic
nmf_model = None  # замените на объект NMF

# запуск анализа
analyze_topics(lda_coherence, bert_model, nmf_model)


# вывод рекомендаций
def recommendations():
    print("\nобщие рекомендации по улучшению:")
    print("- проверьте качество предобработки данных (удаление лишних символов, нормализация).")
    print("- экспериментируйте с различными параметрами моделей.")
    print("- используйте более крупные корпуса текстов.")
    print("- для BERTopic можно интегрировать SentenceTransformer для улучшения качества эмбеддингов.")
    print("- добавьте автоматизированные тесты для проверки качества предобработки и моделирования.")


# вызов функции рекомендаций
recommendations()
