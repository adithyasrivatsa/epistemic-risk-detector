"""Local vector store using SQLite and sentence-transformers."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from src.core.config import RetrievalConfig
from src.core.interfaces import EvidenceProvider
from src.core.schemas import EvidenceChunk


class LocalVectorStore(EvidenceProvider):
    """SQLite-backed vector store with local embeddings."""

    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self._encoder: Any = None
        self._db: sqlite3.Connection | None = None
        self._init_db()

    @property
    def encoder(self) -> Any:
        """Lazy load the sentence transformer model."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.config.embedding_model)
        return self._encoder

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(str(db_path))
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)
        """)
        self._db.commit()

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = int(end - self.config.chunk_size * 0.2)
                for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                    pos = text.rfind(sep, search_start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - self.config.chunk_overlap

        return [c for c in chunks if c]  # Filter empty chunks

    def _generate_chunk_id(self, source: str, chunk_index: int, text: str) -> str:
        """Generate deterministic chunk ID."""
        content = f"{source}:{chunk_index}:{text[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        return self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def _cosine_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity (embeddings are already normalized)."""
        return np.dot(doc_embs, query_emb)

    def index_document(self, path: str) -> int:
        """Index a single document."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        text = path_obj.read_text(encoding="utf-8", errors="ignore")
        chunks = self._chunk_text(text)

        if not chunks:
            return 0

        embeddings = self._embed(chunks)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = self._generate_chunk_id(str(path), i, chunk)

            self._db.execute(
                """
                INSERT OR REPLACE INTO chunks (id, text, source, chunk_index, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    chunk,
                    str(path),
                    i,
                    emb.tobytes(),
                    json.dumps({"filename": path_obj.name}),
                ),
            )

        self._db.commit()
        return len(chunks)

    def index_directory(self, path: str, extensions: list[str] | None = None) -> int:
        """Index all documents in a directory."""
        extensions = extensions or [".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml"]
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        total_chunks = 0
        for ext in extensions:
            for file_path in path_obj.rglob(f"*{ext}"):
                try:
                    total_chunks += self.index_document(str(file_path))
                except Exception as e:
                    print(f"Warning: Failed to index {file_path}: {e}")

        return total_chunks

    def retrieve(self, claim: str, top_k: int | None = None) -> list[EvidenceChunk]:
        """Retrieve evidence chunks relevant to a claim."""
        top_k = top_k or self.config.top_k

        # Get all chunks from DB
        cursor = self._db.execute(
            "SELECT id, text, source, chunk_index, embedding, metadata FROM chunks"
        )
        rows = cursor.fetchall()

        if not rows:
            return []  # No evidence is a valid signal

        # Embed the claim
        query_emb = self._embed([claim])[0]

        # Compute similarities
        results = []
        for row in rows:
            chunk_id, text, source, chunk_index, emb_bytes, metadata_str = row
            doc_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            similarity = float(np.dot(query_emb, doc_emb))

            if similarity >= self.config.similarity_threshold:
                results.append(
                    EvidenceChunk(
                        id=chunk_id,
                        text=text,
                        source=source,
                        similarity_score=similarity,
                        chunk_index=chunk_index,
                        metadata=json.loads(metadata_str),
                    )
                )

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Clear all indexed documents."""
        self._db.execute("DELETE FROM chunks")
        self._db.commit()

    def stats(self) -> dict:
        """Get index statistics."""
        cursor = self._db.execute("SELECT COUNT(*), COUNT(DISTINCT source) FROM chunks")
        total_chunks, total_docs = cursor.fetchone()
        return {"total_chunks": total_chunks, "total_documents": total_docs}
