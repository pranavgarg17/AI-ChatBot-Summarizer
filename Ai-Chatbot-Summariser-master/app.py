import logging
import streamlit as st
from UI.sidebar import Sidebar
from UI.chat_interface import ChatInterface
from api.api_endpoint import ChatGPTClient
from data.embedding_service import EmbeddingService
from data.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

class UnifiedSummarizerApp:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store_service = VectorStoreService()
        self.chatgpt_client = ChatGPTClient()
        self.sidebar = Sidebar(self.handle_input_selection)
        self.chat_interface = ChatInterface(self.generate_response)

    def handle_input_selection(self, input_source):
        try:
            if not isinstance(input_source, dict) or "type" not in input_source:
                st.error("⚠️ Invalid input source format.")
                return False

            input_type = input_source["type"]
            input_value = input_source.get("value")
            summary_length = input_source.get("length", "Medium (300-600 words)")
            question_count = input_source.get("question_count", 5)

            from summarizer_module import (
                summarize_uploaded_pdf,
                summarize_website_url,
                summarize_youtube_url
            )

            client = self.chatgpt_client

            if input_type == "website":
                summaries = summarize_website_url(input_value)
                st.session_state.summary_data = summaries
                st.session_state.content_loaded = True
                st.session_state.summary_shown = False
                if "error" not in summaries:
                    context = summaries.get("map_reduce_summary", "")
                    questions = client.auto_generate_questions(context, count=question_count)
                    st.session_state.auto_questions = questions.splitlines()
                return True

            elif input_type == "pdf":
                if input_value is None:
                    st.error("❌ No PDF uploaded.")
                    return False
                summaries = summarize_uploaded_pdf(input_value)
                st.session_state.summary_data = summaries
                st.session_state.content_loaded = True
                st.session_state.summary_shown = False
                if "error" not in summaries:
                    context = summaries.get("map_reduce_summary", "")
                    questions = client.auto_generate_questions(context, count=question_count)
                    st.session_state.auto_questions = questions.splitlines()
                return True

            elif input_type == "youtube":
                summaries = summarize_youtube_url(input_value)
                st.session_state.summary_data = summaries
                st.session_state.content_loaded = True
                st.session_state.summary_shown = False
                if "error" not in summaries:
                    context = summaries.get("map_reduce_summary", "")
                    questions = client.auto_generate_questions(context, count=question_count)
                    st.session_state.auto_questions = questions.splitlines()
                return True

            else:
                st.error(f"❌ Unsupported input type: {input_type}")
                return False

        except Exception as e:
            logger.error(f"Error in handle_input_selection: {e}")
            st.error(f"⚠️ Error handling input: {e}")
            return False

    def generate_response(self, query):
        try:
            if "vector_store" in st.session_state:
                vector_store = st.session_state.vector_store
                docs, context = self.vector_store_service.retrieve_relevant_content(query, vector_store)
                st.write(f"🧠 Found {len(docs)} relevant sections from the website.")
                return self.chatgpt_client.generate_response(query, context)

            elif st.session_state.get("content_loaded"):
                summaries = st.session_state.summary_data
                combined_context = summaries.get("map_reduce_summary", "") + "\n\n" + summaries.get("refine_summary", "")
                return self.chatgpt_client.generate_response(query, combined_context)

            return "Please upload content or process a website first."

        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def run(self):
        st.set_page_config(page_title="Unified AI Summarizer", layout="wide")
        st.title("🧠 Unified Summarizer & Chat Assistant")
        self.sidebar.render()
        self.chat_interface.render()

if __name__ == "__main__":
    app = UnifiedSummarizerApp()
    app.run()
