from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.embeddings import Embeddings

class EmbeddingService:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"}  # ✅ Prevents meta tensor error
        )

    def get_embeddings(self):
        return self._embedding_model
