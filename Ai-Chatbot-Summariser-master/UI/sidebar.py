import streamlit as st

class Sidebar:
    def __init__(self, on_select_callback):
        self.on_select_callback = on_select_callback

    def render(self):
        st.sidebar.title("📂 Content Source")

        input_type = st.sidebar.radio("Choose input type", ["PDF Upload", "YouTube Video", "Website"])
        summary_length = st.sidebar.selectbox("🔠 Summary Length", [
            "Short (100-300 words)",
            "Medium (300-600 words)",
            "Long (600-1000 words)",
            "Very Long (1000+ words)"
        ])

        auto_question_count = st.sidebar.slider("🤖 Auto Questions", 0, 10, 5)

        if input_type == "PDF Upload":
            uploaded_file = st.sidebar.file_uploader("📄 Upload PDF", type=["pdf"])
            if uploaded_file and st.sidebar.button("Summarize PDF"):
                self.on_select_callback({
                    "type": "pdf",
                    "value": uploaded_file,
                    "length": summary_length,
                    "question_count": auto_question_count
                })

        elif input_type == "YouTube Video":
            yt_url = st.sidebar.text_input("📺 Paste YouTube URL")
            if yt_url and st.sidebar.button("Summarize YouTube"):
                self.on_select_callback({
                    "type": "youtube",
                    "value": yt_url,
                    "length": summary_length,
                    "question_count": auto_question_count
                })

        elif input_type == "Website":
            website_url = st.sidebar.text_input("🌐 Paste Website URL")
            if website_url and st.sidebar.button("Analyze Website"):
                self.on_select_callback({
                    "type": "website",
                    "value": website_url,
                    "length": summary_length,
                    "question_count": auto_question_count
                })
