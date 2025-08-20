import os
import requests

from dotenv import load_dotenv
load_dotenv()

# 环境变量：
#   OPENAI_API_KEY          必填
#   OPENAI_API_BASE_URL     选填，默认 https://api.openai.com/v1
#   LLM_MODEL               选填，默认 gpt-4o-mini

def _chat(messages, *, temperature: float = 0.0) -> str:
    api_key  = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model    = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置")
    url = f"{base_url}/chat/completions"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature, "n": 1, "stream": False},
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    log_path = "llm.log"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[LLM] {data['usage']['prompt_tokens']} tokens used for prompt\n")
        f.write(f"the answer is {data['choices'][0]['message']['content']}\n\n")
    return data["choices"][0]["message"]["content"]
