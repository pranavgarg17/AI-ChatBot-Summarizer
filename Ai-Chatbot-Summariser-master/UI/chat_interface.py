import streamlit as st
import logging

logger = logging.getLogger(__name__)

class ChatInterface:

    def __init__(self, generate_response):
        self.generate_response = generate_response

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! I am your content assistant. You can ask questions about the summary of the uploaded PDF, website, or YouTube video."}
            ]

        if "summary_shown" not in st.session_state:
            st.session_state.summary_shown = False

        if "auto_questions" not in st.session_state:
            st.session_state.auto_questions = []

    def render(self):
        # Display summary if available and not already shown
        if st.session_state.get("summary_data") and not st.session_state.summary_shown:
            st.markdown("## 📄 Summary")
            summary = st.session_state.summary_data
            if isinstance(summary, dict):
                for key, value in summary.items():
                    st.markdown(f"### {key.replace('_', ' ').title()}")
                    st.success(value)
            else:
                st.success(summary)
            st.session_state.summary_shown = True

        # Show auto-generated questions
        if st.session_state.get("auto_questions"):
            st.markdown("### 🤖 Suggested Questions")
            for i, q in enumerate(st.session_state.auto_questions):
                if st.button(f"❓ {q}", key=f"auto_q_{i}"):
                    st.session_state.messages.append({"role": "user", "content": q})
                    self.process_user_input(q)

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message['content'])

        if user_input := st.chat_input("💬 Ask a question:"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            self.process_user_input(user_input)

    def process_user_input(self, query):
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = self.generate_response(query)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                logger.info("Response generated and displayed")