import os
import PyPDF2
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader, YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Load API Credentials ===
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
llm = ChatGroq(groq_api_key=api_key, model=model)

# === For Speech or Raw Text Summarization ===
generic_template = """
Write a summary of the following speech:
Speech: {speech}
Translate the precise summary to {language}
"""
prompt = PromptTemplate(input_variables=['speech', 'language'], template=generic_template)
llm_chain = prompt | llm

def summarize_speech_text(speech: str, language: str = "English"):
    return llm_chain.invoke({'speech': speech, 'language': language})

# === PDF Text Extraction Function ===
def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        return text.strip()
    except Exception as e:
        return f"❌ Error reading PDF: {str(e)}"

# === LangChain Summarization Engine ===
def summarize_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)
    limited_docs = final_documents[:6]  # Limit to prevent overload (safe <12k tokens)

    map_prompt = PromptTemplate(
        input_variables=['text'],
        template="""
Please summarize the below content:
Content: `{text}`
Summary:
"""
    )
    final_prompt = PromptTemplate(
        input_variables=['text'],
        template="""
Provide the final summary of the entire content with these important points.
Add a motivational title, an intro, and numbered points.
Content: {text}
"""
    )

    map_reduce_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=final_prompt,
        verbose=False
    )

    refine_chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        verbose=False
    )

    return {
        "map_reduce_summary": map_reduce_chain.invoke(limited_docs)["output_text"],
        "refine_summary": refine_chain.invoke(limited_docs)["output_text"]
    }



# === Summarize Uploaded PDF ===
def summarize_uploaded_pdf(uploaded_file):
    if uploaded_file is None:
        return {"error": "❌ No PDF uploaded."}

    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp_uploaded.pdf")
    docs = loader.load_and_split()
    return summarize_documents(docs)

# === Summarize Website URL ===
def summarize_website_url(url):
    try:
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        return summarize_documents(docs)
    except Exception as e:
        return {"error": f"❌ Website summarization failed: {str(e)}"}

# === Summarize YouTube Video Transcript ===
def summarize_youtube_url(youtube_url):
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url)
        docs = loader.load()
        return summarize_documents(docs)
    except Exception as e:
        return {"error": f"❌ YouTube summarization failed: {str(e)}"}
