import faiss
import numpy as np
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import UnstructuredURLLoader

class VectorStoreService:
    def create_vector_store(self, url, embedding_model):
        """
        Loads content from a given website URL and creates a FAISS vector store.
        """
        try:
            loader = UnstructuredURLLoader(urls=[url])
            docs = loader.load()
            if not docs:
                return None, "No content retrieved from the website."
            vector_store = FAISS.from_documents(docs, embedding_model)
            return vector_store, None
        except Exception as e:
            return None, str(e)

    def retrieve_relevant_content(self, query, vector_store):
        """
        Performs a similarity search on the vector store based on the query,
        and returns the top documents along with their combined text.
        """
        try:
            docs = vector_store.similarity_search(query, k=5)
            if not docs:
                return [], "No relevant documents found."
            context = "\n\n".join([doc.page_content for doc in docs])
            return docs, context
        except Exception as e:
            return [], f"Error retrieving relevant content: {str(e)}"
