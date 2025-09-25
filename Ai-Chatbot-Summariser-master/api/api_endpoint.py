import os
import time
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

class ChatGPTClient:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.timeout = 30
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", 7000))

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
        except Exception:
            self.tokenizer = None

    def truncate_context(self, context: str) -> str:
        if not self.tokenizer:
            return context[:12000]
        tokens = self.tokenizer.encode(context)
        tokens = tokens[:self.max_context_tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _post_to_groq(self, messages: list, temperature: float = 0.7, max_retries: int = 3) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)

                if response.status_code == 413:
                    return "❌ Input too large. Please shorten the context or summary."

                if response.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"⚠️ Rate limit hit. Retrying in {wait} seconds...")
                    time.sleep(wait)
                    continue

                if response.status_code == 503:
                    return "⚠️ Groq API temporarily unavailable (503). Try again shortly."

                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return f"❌ Request to Groq failed: {str(e)}"
                time.sleep(5)

            except Exception as e:
                return f"❌ Unexpected error: {str(e)}"

        return "❌ Failed to get response after multiple attempts."

    def generate_response(self, query: str, context: str) -> str:
        safe_context = self.truncate_context(context)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer the question."
            },
            {
                "role": "user",
                "content": f"Context:\n{safe_context}\n\nQuestion: {query}"
            }
        ]
        return self._post_to_groq(messages)

    def summarize_content(self, content: str, word_limit: int = 300) -> str:
        prompt = (
            f"Please summarize the following content in approximately {word_limit} words.\n\n{self.truncate_context(content)}"
        )
        messages = [
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": prompt}
        ]
        return self._post_to_groq(messages, temperature=0.5)

    def auto_generate_questions(self, content: str, count: int = 5) -> str:
        prompt = (
            f"Based on the following content, generate {count} short and relevant questions a user might ask.\n\nContent:\n{self.truncate_context(content)}"
        )
        messages = [
            {"role": "system", "content": "You are a question-generating assistant."},
            {"role": "user", "content": prompt}
        ]
        return self._post_to_groq(messages)
