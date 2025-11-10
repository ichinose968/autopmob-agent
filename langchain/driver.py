# driver.py
from langchain import run_pipeline

if __name__ == "__main__":
    sample_passages = [{"text": "x + y = z", "page": 1}]
    sample_meta = {"title": "demo"}
    result = run_pipeline("doc_demo", passages=sample_passages, meta=sample_meta)
    print(result)
