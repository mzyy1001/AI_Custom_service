# client_init.py
import os
from openai import OpenAI

def build_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_API_BASE_URL") or "https://api.openai.com/v1"

    if not api_key or api_key.startswith("sk-None"):
        raise RuntimeError(
            "OPENAI_API_KEY 缺失或疑似无效。请在环境中正确设置（并确认未泄露/已旋转）。"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client

if __name__ == "__main__":
    c = build_openai_client()
    # 简单连通性测试（不会消耗太多 Token）
    resp = c.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "pong"}],
        temperature=0
    )
    print(resp.choices[0].message.content.strip())
