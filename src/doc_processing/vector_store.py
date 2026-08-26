"""
ChromaDB vector store
"""
import hashlib
import chromadb
from chromadb.config import Settings
import config

class VectorStore:
    def __init__(self, persist_dir = config.PERSIST_DIR):
        """
        Args:
            persist_dir: folder where chromadb stores data ie embeddings
        """
        self.client = chromadb.PersistentClient(
            path = persist_dir,
            settings = Settings(anonymized_telemetry=False)
        )

    def _file_hash(self, pdf_path):
        """
        Hash the file's binary contents so the same file is recognized
        as already indexed regardless of its path/filename.
        """
        hasher = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def _pdf_id(self, pdf_path):
        """
        Convert pdf content into stable collection name.
        Same file bytes -> same collection, regardless of path.
        """
        content_hash = self._file_hash(pdf_path)
        return f"pdf-{content_hash}"[:63]

    def _get_or_create_collection(self, pdf_path):
        name = self._pdf_id(pdf_path)
        return self.client.get_or_create_collection(
                name = name,
                metadata = {"source_path": pdf_path, "hnsw:space": "cosine"},
                embedding_function = None 
        )

    def is_indexed(self, pdf_path):
        """Return true if the pdf has already been embedded adn stored"""
        name = self._pdf_id(pdf_path)
        existing = [c.name for c in self.client.list_collections()]
        if name not in existing:
            return False
        col = self.client.get_collection(name=name, embedding_function=None)
        return col.count() > 0
    

    def store(self, pdf_path, chunks, embeddings, chunk_pages):
        """
        Persists chunks and their embeddings for pdf 

        Args:
            pdf_path : path used to derive content-hash collection key
            chunks : chunks from LateChunking.chunks
            embeddings: embeddings from LateChunking.chunk_embeddings
        """
        col = self._get_or_create_collection(pdf_path)

        embeddings_list = [e.cpu().float().tolist() for e in embeddings]
        ids = [f"chunk-{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chunk_index": i,
                "page_start": chunk_pages[i]["page_start"] if isinstance(chunk_pages[i], dict) else (chunk_pages[i][0] if isinstance(chunk_pages[i], tuple) else chunk_pages[i]),
                "page_end":   chunk_pages[i]["page_end"]   if isinstance(chunk_pages[i], dict) else (chunk_pages[i][1] if isinstance(chunk_pages[i], tuple) else chunk_pages[i]),
            }
            for i in range(len(chunks))
        ]

        # upsert for re-indexing the same pdf is safe
        col.upsert(
            ids = ids,
            documents=chunks,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )
        print(f"[VectorStore] stored {len(chunks)} chunks for '{pdf_path}'")


    def load(self, pdf_path):
        col = self._get_or_create_collection(pdf_path)
        result = col.get(include=["documents", "metadatas", "embeddings"])

        combined = sorted(
            zip(result["metadatas"], result["documents"], result["embeddings"]),
            key = lambda x: x[0]["chunk_index"]
        )
        metadatas, chunks, embeddings = zip(*combined) if combined else ([], [], [])
        pages = [{"page_start": m["page_start"], "page_end": m["page_end"]} for m in metadatas]

        print(f"[VectorStore] Loaded {len(chunks)} chunks for '{pdf_path}'")
        return list(chunks), pages

    def query(self, pdf_path, query_embedding, top_k):
        """
        If we want chromadb to handle similarity search natively.

        Args:
            query_embedding: python list
            top_k : number of result

        Returns:
            list of matching chunk texts
        """
        col = self._get_or_create_collection(pdf_path)
        results = col.query(
                query_embeddings = [query_embedding],
                n_results = top_k,
                include=["documents", "metadatas", "distances"]
        )
        chunks = results["documents"][0]
        distances = results["distances"][0]
        pages = [
                {"page_start": m["page_start"], "page_end" : m["page_end"]}
                for m in results["metadatas"][0]
        ]
        return list(zip(chunks, pages, distances))

    def delete(self, pdf_path):
        """Remove all stored data for pdf for re-indexing"""
        name = self._pdf_id(pdf_path)
        try:
            self.client.delete_collection(name)
            print(f"[VectorStore] Deleted collection for '{pdf_path}'")
        except Exception:
            pass