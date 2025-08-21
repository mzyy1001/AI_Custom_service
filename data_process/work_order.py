# -*- coding: utf-8 -*-
"""
csv_to_chains_llm.py
将带中文表头的 CSV 逐行转换为“线性问题解决链”，一条链一段纯文本。
- 第一行：现象（由 部件+标题+描述 归纳）
- 后续行：异常特征/关键事实 -> 动作（由 原因分析+解决方案，并参考描述 归纳）
- 结尾必须是动作；不输出编号/JSON；链条之间空行。
用法：
    python csv_to_chains_llm.py input.csv output.txt
环境变量（可选）：
    OPENAI_API_KEY         必填（或换成你的网关 Key）
    OPENAI_API_BASE_URL    选填，默认 https://api.openai.com/v1
    LLM_MODEL              选填，默认 gpt-4o-mini
"""

import os
import sys
import time
import math
import json
import typing as t
import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()


# ==== 基本配置（也可通过环境变量覆盖） ====
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

TIMEOUT_S = 180
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0  # 指数退避

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY 未设置。请在环境变量中配置。")

# ==== 列名别名兜底 ====
COL_PART_CANDIDATES   = ["部件", "子系统", "组件", "设备", "位置", "对象"]
COL_TITLE_CANDIDATES  = ["标题", "问题标题", "现象标题", "问题", "故障现象", "主题"]
COL_DESC_CANDIDATES   = ["描述", "问题描述", "现象描述", "补充说明", "备注"]
COL_CAUSE_CANDIDATES  = ["原因分析", "原因", "可能原因", "分析", "成因", "异常原因"]
COL_FIX_CANDIDATES    = ["解决方案", "处理措施", "处置方案", "修复措施", "解决办法", "处理方法"]

def _get(row: pd.Series, candidates: t.List[str]) -> str:
    for c in candidates:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return ""

def _compact(*parts: str, sep: str="，", max_len: int=38) -> str:
    """拼接首行现象，尽量短；去重/去空。"""
    items = [p.strip() for p in parts if p and str(p).strip()]
    if not items:
        return "故障/异常"
    s = sep.join(dict.fromkeys(items))  # 去重保序
    return s[:max_len]

# ==== 调用 LLM ====
def call_llm(prompt: str) -> str:
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是资深工业售后工程师，擅长将表格记录压缩为线性问题解决链。严格遵守用户格式要求，只输出链条文本。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 800,
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=TIMEOUT_S)
            resp.raise_for_status()
            out = resp.json()
            return out["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            # 简单指数退避
            sleep_s = (RETRY_BACKOFF ** (attempt - 1)) + (0.1 * attempt)
            print(f"⚠️  LLM 调用失败（第 {attempt}/{MAX_RETRIES} 次）：{e}  → {sleep_s:.1f}s 后重试")
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM 连续失败：{last_err}")

# ==== Prompt 模板 ====
FEWSHOT = """【示例1】
输入字段：
部件: 电源
标题: 机器人开不了机
描述: RCS 日志显示电量低
原因分析: 长时间未充电导致电量耗尽
解决方案: 手动充电30分钟以上

输出链条：
机器人开不了机
RCS 日志显示电量低
进行手动充电

【示例2】
输入字段：
部件: 扫码台
标题: 无法扫码
描述: 软件能打开但无法识别条码
原因分析: 曝光度设置不当
解决方案: 调整相机曝光度

输出链条：
无法扫码
相机曝光度设置不当
调整相机曝光度

【示例3】
输入字段：
部件: 电池系统
标题: 设备间歇性断电
描述: 触碰电缆时会掉电
原因分析: 电池连接线接触不良
解决方案: 重新插紧或更换连接线

输出链条：
设备间歇性断电
电池连接线接触不良
重新插紧或更换连接线

【示例4】
输入字段：
部件: 软件
标题: 安装驱动加密狗
描述: 平台需要安装驱动加密狗
原因分析: 3D缺少加密狗驱动
解决方案: 
1.检查加密狗是否正常插入电脑中
2.安装加密狗驱动是否能恢复
3.进入网址 http://localhost:1947/_int_/devices.html 查询加密狗驱动是否正常
4.加密狗是否在 rcs 所在的内网里

输出链条：
需要安装加密狗驱动
加密狗不正常插入电脑中
插入加密狗电脑

需要安装加密狗驱动
加密狗驱动未安装
安装加密狗驱动

需要安装加密狗驱动
进入网址http://localhost:1947/_int_/devices.html查询加密狗驱动不正常
更换新的加密狗驱动

需要安装加密狗驱动
加密狗不在rcs所在的内网里
加密狗插入内网的任意一台RCS电脑中

"""

PROMPT_TEMPLATE = """你将把一行记录转换为 1~N 段【线性问题解决链】（plain text，段与段之间空行）。
每段链条的通用结构：
- 第1行：现象/待办。若标题或部件本身是动作词（如：添加白名单/安装驱动/升级/配置…），改写为“需要<该动作>”；否则整合部件+标题+描述成一句简洁现象。
- 第2-N行：异常特征/关键事实（由“描述/原因分析/解决方案”提炼），如“未插入/未安装/未连接/参数不当/损坏/不在内网/状态异常/版本不正确”等。如果有多个特征，可以一行行依次输出。
- 最后一行：动作（维修/更换/调整/配置/重启/升级/清洁/加固/充电等），**必须以动作结尾** 输出完整的链条对应的解决方案动作，包含动作细节。但是把所有内容放在一行。
（若只有单一步骤的解决方案，且无可用特征，可输出 2 行：第1行为现象，第2/末行为动作；若能提炼出关键事实，仍优先 3 行：现象 → 事实 → 动作。）

规则：
1) 仅保留“事实→动作”，**删除**流程/问答话术（如：是/否、进入第X步、问题是否解决、联系售后等）。
2) 将“检查/确认/测试/观察/核对”类语句改写为客观事实（未插入/未安装/未连接/参数不当/损坏/不在内网/状态异常…）+ 随后的修复动作。
3) 合并等价动作；同一路径中连续操作可合并至同一末尾动作行。
4) **拆分策略**：当解决方案包含多个互斥分支（如“若…则…/否则…/是/否/不在…/在…”），为每个分支各输出一段链条；各段的第1行（现象）相同。
5) 正则相关：不要单独输出规则为一行；如需体现，可以放入放在动作行的开始。
6) 如果有多步的连续动作，合并成一行，不需要拆分成多个解决链条，解决链条只有在产生分歧的时候才需要拆分。
7) 纯文本输出：一行一句要点；不要编号/列表/JSON；段与段之间空行。

{fewshot}

输入字段：
部件: {part}
标题: {title}
描述: {desc}
原因分析: {cause}
解决方案: {fix}

请只输出符合上述规则的链条文本：
"""

# ==== 主流程 ====
def process_csv(input_csv: str, output_txt: str):
    df = pd.read_csv(input_csv, encoding="utf-8", dtype=str, keep_default_na=False)
    n = len(df)
    print(f"📄 读取 {input_csv} 共 {n} 行")

    with open(output_txt, "w", encoding="utf-8") as out:
        for idx, row in df.iterrows():
            part  = _get(row, COL_PART_CANDIDATES)
            title = _get(row, COL_TITLE_CANDIDATES)
            desc  = _get(row, COL_DESC_CANDIDATES)
            cause = _get(row, COL_CAUSE_CANDIDATES)
            fix   = _get(row, COL_FIX_CANDIDATES)

            # 第一行现象的压缩（供 LLM 参考，最终以 LLM 输出为准）
            phenomenon_hint = _compact(title, part, desc)

            prompt = PROMPT_TEMPLATE.format(
                fewshot=FEWSHOT,
                part=part or "（空）",
                title=title or phenomenon_hint or "（空）",
                desc=desc or "（空）",
                cause=cause or "（空）",
                fix=fix or "（空）",
            )

            print(f"\n🚀 处理第 {idx+1}/{n} 行 | 现象提示: {phenomenon_hint}")
            result = call_llm(prompt)
            print(f"—— 输出 ——\n{result}\n")

            # 写入：每条链之间空行
            if result.strip():
                out.write(result.strip() + "\n\n")

    print(f"✅ 完成，结果已写入：{output_txt}")

# ==== CLI ====
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法：python csv_to_chains_llm.py input.csv output.txt")
        sys.exit(1)
    process_csv(sys.argv[1], sys.argv[2])
