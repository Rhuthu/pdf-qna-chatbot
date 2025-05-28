import requests
import os
from dotenv import load_dotenv


load_dotenv()


LLMFOUNDRY_TOKEN = os.getenv("LLM_TOKEN")
LLMFOUNDRY_PROJECT = "my-test-project"
LLMFOUNDRY_BASE_URL = "https://llmfoundry.straive.com/openai/v1/chat/completions"

def generate_llm_response(query, context):
    headers = {
        "Authorization": f"Bearer {LLMFOUNDRY_TOKEN}:{LLMFOUNDRY_PROJECT}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
        ]
    }

    response = requests.post(LLMFOUNDRY_BASE_URL, headers=headers, json=payload)
    
    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error in LLM response: {str(e)}"
