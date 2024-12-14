import arxiv
import pandas as pd

def collect_data(query="data science", max_results=100):
    client = arxiv.Client(
        page_size=max_results,  # установка максимального количества результатов на странице
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = client.results(search)  # использование Client.results()
    papers = []
    for result in results:
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id
        })
    return pd.DataFrame(papers)

# пример использования
if __name__ == "__main__":
    query = "data science"
    data = collect_data(query=query, max_results=50)
    data.to_csv("collected_data.csv", index=False)
    print("данные успешно собраны и сохранены в 'collected_data.csv'")
