"""Tests for evidence retrieval module."""

import pytest

from src.core.schemas import EvidenceChunk
from src.retrievers.local_vector import LocalVectorStore

# Skip tests that require sentence-transformers if not installed
try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed"
)


class TestLocalVectorStore:
    """Test suite for LocalVectorStore."""

    def test_init_creates_database(self, test_config, tmp_path):
        """Store should create database on initialization."""
        store = LocalVectorStore(test_config.retrieval)

        # Database file should exist
        db_path = tmp_path / "test_evidence.db"
        assert db_path.exists()

    @requires_sentence_transformers
    def test_index_document(self, test_config, tmp_path):
        """Store should index a document and return chunk count."""
        # Create test document
        doc_path = tmp_path / "test_doc.txt"
        doc_path.write_text("This is a test document. It has multiple sentences. Each should be indexed.")

        store = LocalVectorStore(test_config.retrieval)
        count = store.index_document(str(doc_path))

        assert count > 0

    def test_index_nonexistent_file_raises(self, test_config):
        """Store should raise FileNotFoundError for missing files."""
        store = LocalVectorStore(test_config.retrieval)

        with pytest.raises(FileNotFoundError):
            store.index_document("/nonexistent/path.txt")

    @requires_sentence_transformers
    def test_retrieve_returns_evidence_chunks(self, test_config, tmp_path):
        """Retrieve should return list of EvidenceChunk objects."""
        # Create and index test document
        doc_path = tmp_path / "test_doc.txt"
        doc_path.write_text("Python was created by Guido van Rossum in 1991. It is a popular programming language.")

        store = LocalVectorStore(test_config.retrieval)
        store.index_document(str(doc_path))

        results = store.retrieve("When was Python created?")

        assert isinstance(results, list)
        if results:  # May be empty if similarity threshold not met
            assert all(isinstance(r, EvidenceChunk) for r in results)

    def test_retrieve_empty_corpus_returns_empty(self, test_config):
        """Retrieve on empty corpus should return empty list (valid signal)."""
        store = LocalVectorStore(test_config.retrieval)
        results = store.retrieve("Any query")

        assert results == []

    @requires_sentence_transformers
    def test_retrieve_respects_top_k(self, test_config, tmp_path):
        """Retrieve should return at most top_k results."""
        # Create document with many chunks
        doc_path = tmp_path / "test_doc.txt"
        doc_path.write_text(" ".join([f"Sentence number {i}." for i in range(100)]))

        store = LocalVectorStore(test_config.retrieval)
        store.index_document(str(doc_path))

        results = store.retrieve("sentence", top_k=3)

        assert len(results) <= 3

    @requires_sentence_transformers
    def test_stats_returns_counts(self, test_config, tmp_path):
        """Stats should return chunk and document counts."""
        doc_path = tmp_path / "test_doc.txt"
        doc_path.write_text("Test document content.")

        store = LocalVectorStore(test_config.retrieval)
        store.index_document(str(doc_path))

        stats = store.stats()

        assert "total_chunks" in stats
        assert "total_documents" in stats
        assert stats["total_chunks"] > 0
        assert stats["total_documents"] == 1

    @requires_sentence_transformers
    def test_clear_removes_all_data(self, test_config, tmp_path):
        """Clear should remove all indexed data."""
        doc_path = tmp_path / "test_doc.txt"
        doc_path.write_text("Test document content.")

        store = LocalVectorStore(test_config.retrieval)
        store.index_document(str(doc_path))
        store.clear()

        stats = store.stats()
        assert stats["total_chunks"] == 0

    @requires_sentence_transformers
    def test_index_directory(self, test_config, tmp_path):
        """Store should index all matching files in directory."""
        # Create test files
        (tmp_path / "doc1.txt").write_text("First document content.")
        (tmp_path / "doc2.txt").write_text("Second document content.")
        (tmp_path / "ignored.xyz").write_text("Should be ignored.")

        store = LocalVectorStore(test_config.retrieval)
        count = store.index_directory(str(tmp_path), extensions=[".txt"])

        assert count > 0
        stats = store.stats()
        assert stats["total_documents"] == 2

    @requires_sentence_transformers
    def test_similarity_scores_in_range(self, test_config, tmp_path):
        """Similarity scores should be between 0 and 1."""
        doc_path = tmp_path / "test_doc.txt"
        doc_path.write_text("Python programming language facts and information.")

        store = LocalVectorStore(test_config.retrieval)
        store.index_document(str(doc_path))

        results = store.retrieve("Python")

        for result in results:
            assert 0 <= result.similarity_score <= 1
