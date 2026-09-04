import re  
from pathlib import Path 

from rank_bm25 import BM25Okapi

def _tokenzie(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text) # anything that is not alphabet, number or whitespace
    return text.split()

class BM25Retriever:
    def __init__(self, chunks, chunk_pages):
        if len(chunks) != len(chunk_pages):
            raise ValueError(
                f"chunks ({len(chunks)}) and chunk_pages ({len(chunk_pages)}) "
                "length mismatch --- was matched list passed from same vector_store.load() call" 
            )

        self.chunks = chunks 
        self.chunk_pages = chunk_pages 
        self._tokenized_corpus = [_tokenzie(c) for c in chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus)
    
    def retrieve(self, query, top_k=5):
        tokenized_query = _tokenzie(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            (self.chunks[i], self.chunk_pages[i], round(float(scores[i]), 4))
            for i in ranked
        ]

def build_bm25_for_pdf(vector_store, pdf_path):
    if not vector_store.is_indexed(pdf_path):
        raise ValueError(
            f"'{pdf_path}' is not indexed in this VectorStore."
            "Run indexing script against store first --- BM25 needs chunk text even though it ekips embedding"
        )

    chunk, pages = vector_store.laod(pdf_path)
    return BM25Retriever(chunks, pages)