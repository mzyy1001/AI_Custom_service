# feature_engine/llm_engine/llm.py
from __future__ import annotations
import os, re, json
from typing import Optional, Tuple, List
import requests
from dotenv import load_dotenv
from feature_engine.llm_client.chat import _chat

def _normalize_yes_no(text: str) -> Optional[bool]:
    """把输出规整成 True/False/None（支持中英常见回答）。"""
    if text is None:
        return None
    s = str(text).strip().lower()
    s = s.strip(" \t\r\n'\"，。.!?；;：:")
    if s in {"是", "对", "正确", "yes", "y", "true", "t", "1"}:
        return True
    if s in {"不是", "否", "不对", "错误", "no", "n", "false", "f", "0"}:
        return False
    # 兜底：句子里只出现正向或负向关键词
    pos = any(w in s for w in ("是", "对", "yes", "true"))
    neg = any(w in s for w in ("不是", "否", "no", "false"))
    if pos and not neg:
        return True
    if neg and not pos:
        return False
    return None

def llm_yes_no(prompt: str, *, temperature: float = 0.0) -> Tuple[Optional[bool], str]:
    """
    向模型提问“是 / 不是”。
    返回: (结果: True/False/None, 原始文本)
    """
    sys = "你是一位判定器。只回答“是”或“不是”，不要添加其他文字或解释。"
    raw = _chat(
        [{"role": "system", "content": sys},
         {"role": "user",   "content": prompt.strip()}],
        temperature=temperature,
    )
    # 取第一行以减少冗余影响
    first = raw.splitlines()[0].strip()
    return _normalize_yes_no(first), raw

def llm_select(prompt: str, options: List[str], *, temperature: float = 0.0) -> Tuple[Optional[int], str]:
    """
    让模型在给定列表中选择一个。
    返回: (index 或 None, 原始文本)
    """
    if not options:
        raise ValueError("llm_select: options 不能为空")
    numbered = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    sys = "你是一位选择器。只输出你选择的序号（阿拉伯数字），不要输出其他任何内容。"
    usr = f"{prompt.strip()}\n\n候选项：\n{numbered}\n\n只输出一个数字序号。"
    raw = _chat(
        [{"role": "system", "content": sys},
         {"role": "user",   "content": usr}],
        temperature=temperature,
    )
    text = raw.strip()

    # 1) 直接数字
    m = re.search(r"-?\d+", text)
    if m:
        idx = int(m.group(0))
        if 0 <= idx < len(options):
            return idx, raw

    # 2) JSON 形式 {"index": 2}
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "index" in data:
            idx = int(data["index"])
            if 0 <= idx < len(options):
                return idx, raw
    except Exception:
        pass

    # 3) 直接返回了选项文本
    low = text.lower()
    for i, opt in enumerate(options):
        if str(opt).strip().lower() in low:
            return i, raw

    return None, raw


# --- 可选本地测试 ---
if __name__ == "__main__":
    ok, raw1 = llm_yes_no("电源指示灯现在亮着吗？")
    print("YES/NO =>", ok, "| raw:", raw1)

    idx, raw2 = llm_select("下一步应该做什么？", ["更换电源适配器", "检查插座", "联系维修"])
    print("CHOICE  =>", idx, "| raw:", raw2)
