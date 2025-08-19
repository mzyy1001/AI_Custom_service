import csv
import os
from dotenv import load_dotenv
import requests

load_dotenv()

def _chat(messages, *, temperature: float = 0.0) -> str:
    api_key  = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model    = os.getenv("LLM_MODEL", "gemini-2.5-flash-preview-05-20")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置")
    url = f"{base_url}/chat/completions"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature, "n": 1, "stream": False},
        timeout=1000,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    log_path = "llm.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[LLM] {data['usage']['prompt_tokens']} tokens used for prompt\n")
        f.write(f"the answer is {data['choices'][0]['message']['content']}\n\n")
    return data["choices"][0]["message"]["content"]

SYSTEM_PROMPT = """你是一个专业的故障诊断链生成器。  
任务：把输入的【现象+排查指引】解析成多条独立的【完整线性链条】，每条链条必须从现象开始，一直延伸到解决动作或结果，不要遗漏任何中间步骤。  

【输出要求】：
1. 每条链必须以传入的【现象】开头。  
2. 不要跳过任何分支，所有分支都要展开成独立的链条,但是不要往上跳转，所有往上跳转的链条都是不合法的。
3. 每条链必须从现象出发，直到链条终点（动作、结果或话术），保持完整。  
4. 输出为**纯文本**，每条链占一块，多行描述，每条链之间空一行。  
5. 禁止输出 JSON、编号、列表符号，仅保留纯文本。  
6. 不要返回任意的复测结果
7. 如果解决措施不是在结尾出现的，必须正常经过，但是不要输出。
8. 如果一个解决方案，使用了，但是未能解决问题，那么这个字段不需要输出。
9. 如果结尾是转人工服务的链条也不要输出。
10.如果是检查（注意不是解决方案，只是检查）+检查到的特征，合成同一句话，如果检查出来的结果是正常的，也不要输出。当且仅当是故障信息的时候才输出。
11.问题解决的字段不需要输出。
12.所有正常运行的句子不需要输出，比如排查故障发现没有故障的句子。
13.不要输出所有无故障的句子，比如未损坏，未松动等等。

【示例】  
输入：  
现象：机器人不发车  
排查指引：  
1. 确认是否能扫码？  
- 是：建议联系售后  
- 否：进入步骤2  

2. 相机是否能连接？  
- 是：进入步骤3  
- 否：更换相机  

3. 重启扫码软件  
- 否：建议联系专业的售后技术支持人员获得帮助
- 是：问题解决  

输出：  
机器人不发车  
能扫码  
建议联系售后  

机器人不发车  
无法扫码  
相机可以连接  
重启扫码软件  
  

机器人不发车  
无法扫码  
相机无法连接  
更换相机  

机器人不发车  
无法扫码  
相机可以连接  
重启扫码软件 
扫码软件正常
建议联系售后 

"""



def parse_with_llm(phenomenon: str, steps: str) -> str:
    usr = f"""现象：{phenomenon}
排查指引：
{steps}

"""

    return _chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": usr},
    ], temperature=0.0)

def process_csv(path, out_path="output.txt"):
    with open(path, newline="", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            phenomenon = row["现象"].strip()
            raw_steps = row["排查指引"]
            text = parse_with_llm(phenomenon, raw_steps)
            print(f"[LLM] 解析现象: {phenomenon}")
            print(f"[LLM] 解析结果: {text.strip()}")
            
            out.write(text.strip() + "\n\n")

if __name__ == "__main__":
    csv_path = "./data_process/data.csv"   # 输入 CSV
    process_csv(csv_path, "result.txt")  # 输出文本
