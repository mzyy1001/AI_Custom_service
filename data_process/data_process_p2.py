# -*- coding: utf-8 -*-
"""
第二步处理脚本：读取 result.txt，逐段调用 LLM 清洗链条 -> result_refined.txt
"""

import os
import sys
from dotenv import load_dotenv
import requests

load_dotenv()

# ====== 配置区 ======
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-preview-05-20")   # 你可以换成 gpt-4o, gpt-4.1, deepseek 等

if not API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY 未设置")

# ====== 工具函数 ======
def call_llm(prompt: str) -> str:
    """
    调用 LLM API，返回模型输出文本
    """
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严格遵守规则的故障诊断链清洗器。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }

    resp = requests.post(url, headers=headers, json=data, timeout=180)
    resp.raise_for_status()
    result = resp.json()
    if result["choices"][0]["message"]["content"].strip() == "null":
        return ""
    return result["choices"][0]["message"]["content"].strip()


def read_chains(file_path: str):
    """
    按空行分隔读取文件，返回链条列表
    """
    chains = []
    current = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current:
                    chains.append("\n".join(current))
                    current = []
            else:
                current.append(line)

        if current:
            chains.append("\n".join(current))

    return chains


def refine_chain_with_llm(chain: str) -> str:
    """
    用大语言模型清洗链条
    """
    prompt = f"""
你是一个“最小可用链条提炼器”。输入是一段【完整链条】（多行文本，每行一个节点；第 1 行是【现象】）。
请将其压缩为“最小可用链条”，遵守以下规则：

【目标】
- 保留能支撑最终动作的**关键异常特征**（若原文中明确出现）。
- 删除被事实否定或与成功无关的步骤。
- 结尾必须是一个**具体可执行动作**。

【硬性规则】
1) **保留第 1 行现象**（原样保留）。
2) 如果链条最后一句是“联系售后/转人工/联系客服/建议联系专业的售后技术支持人员获得帮助”等话术，则**仅输出 null 这个单词*。
3) 删除表示正常/否定/成功/恢复的行（如：正常、否、在线、成功、恢复正常、能开机/能发车 等）。
4) 若某步骤后紧跟“仍不能开机/未解决”等否定，再继续下一步，说明该步骤无效：**删除该步骤及其否定结果**。
5) **只保留最后一个有效动作**作为链条收尾（如：充电、重启、调整、替换、修复、升级、加固、统一配置 等）。结尾禁止“检查/确认/观察”。
6) **异常特征的保留策略**：
   - 仅当原文本中存在**明确的异常结果**（如“RCS 日志显示电量低/…不足/…异常/…损坏/…离线/无法…/不工作/过高/过低”等）时，**保留其中最能直接支撑最终动作的一条**，放在动作之前。
   - **禁止臆造**异常特征（例如“电池连接线异常”不能凭空添加，除非原文明确给出类似表述）。
   - 若原文没有明确异常结果，则不写特征，直接输出两行（现象 + 动作）。

【输出格式】
- 纯文本，多行；不使用编号/列表/JSON。
- 一条链即可；多条链之间以空行分隔（本次仅处理一段输入）。

【示例】
示例A（必须保留异常特征）：
原始：
机器人开不了机
查看RCS日志显示电量低
进行手动充电
机器人能开机

输出：
机器人开不了机
查看RCS日志显示电量低
进行手动充电

示例B（无明确特征 → 两行输出）：
原始：
机器人开不了机
查看RCS日志显示不是电量低
检查电池连接线
机器人能开机

输出：
机器人开不了机
重新插紧或更换电池连接线

示例C（前一步无效，后一步生效；无明确特征 → 两行输出）：
原始：
机器人开不了机
检查电池连接线
机器人不能开机
检查电池模块
机器人能开机

输出：
机器人开不了机
更换电池模块

【待清洗链条】：
{chain}

【请输出符合规则的最小可用链条】：

    """
    return call_llm(prompt)


def process_result_file(input_path: str, output_path: str):
    """
    读取文件 -> 调用 LLM 逐链清洗 -> 输出新文件
    """
    chains = read_chains(input_path)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, chain in enumerate(chains, 1):
            print(f"🚀 正在处理第 {i}/{len(chains)} 段...")
            print(f"原始链条：\n{chain}\n")
            refined = refine_chain_with_llm(chain)
            print(f"清洗后链条：\n{refined}\n")
            if refined.strip():
                out.write(refined + "\n\n")

    print(f"✅ 已处理完成，结果写入 {output_path}")


if __name__ == "__main__":
    input_file = "result.txt"
    output_file = "result_refined.txt"

    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    process_result_file(input_file, output_file)
