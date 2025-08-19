# feature_tree/training/llm_train.py
from __future__ import annotations
import os, re, json
from typing import Any, Optional, Tuple, List
import requests
from dotenv import load_dotenv
from traitlets import Dict

# 从 .env 读取 KEY / BASE
load_dotenv()

# 环境变量：
#   OPENAI_API_KEY          必填
#   OPENAI_API_BASE_URL     选填，默认 https://api.openai.com/v1
#   LLM_MODEL               选填，默认 gpt-4o-mini

def _chat(messages, *, temperature: float = 0.0) -> str:
    api_key  = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model    = os.getenv("LLM_MODEL", "gpt-4")
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


def classify_line(text: str) -> str:
    """
    将一句话分类为：feature / problem / solution / other
    - feature: 可观测现象/特征
    - problem: 不可直接观测的现象，如问题/bug/需求，或者造成故障的不可观测的原因（比如电池坏了）
    - solution: 具体的处置/操作/方案
    - other: 其它
    """
    sys = """
    你是一个标签器。请把给定句子标成以下四类之一：

    将一句话分类为：feature / problem / solution / other
    - feature: 可观测现象/特征/或者可以直接观察到的问题（比如机器人开不了机）
    - problem: 不可直接观测的现象，如问题/bug/需求，或者造成故障的不可观测的原因（比如电池坏了）
    - solution: 具体的处置/操作/方案
    - other: 其它

    仅输出一个英文小写标签，不要解释。
    """
    usr = f"句子：{text}"
    raw = _chat(
        [{"role": "system", "content": sys},
         {"role": "user", "content": usr}],
        temperature=0.0,
    )
    tag = raw.strip().split()[0].lower()
    if tag not in {"feature", "problem", "solution", "other"}:
        return "other"
    return tag


def canonicalize_problem(text: str) -> str:
    """
    将一句可能含糊的问句/描述归纳成稳定的“问题陈述”（作为 ProblemNode.description）。
    """
    sys = (
        "把输入内容归纳成一句简洁的'问题陈述'(question)"
        "要求：可复用、可检索、面向操作；不要带多余上下文；输出中文句子即可。"
    )
    usr = f"内容：{text}"
    raw = _chat(
        [{"role": "system", "content": sys},
         {"role": "user", "content": usr}],
        temperature=0.2,
    )
    return raw.strip().splitlines()[0].strip()

def _select_index(query: str, options: List[str]) -> Optional[int]:
    """
    在候选列表中选择与 query 语义“完全等价（同义改写）”的一项。
    仅当等价才返回索引；否则返回 None。
    """
    import json, re

    if not options:
        return None

    # --- 只取“描述文本”（去掉可能的 'ID:描述' 前缀），并建立回映射 ---
    descs = []
    for opt in options:
        s = str(opt)
        # 允许 'ID:描述' 格式；只保留描述参与匹配
        if ":" in s:
            s = s.split(":", 1)[1].strip()
        descs.append(s)

    # 组合给 LLM 的编号选项
    numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(descs))

    sys = (
        "你是一个严苛的同义项匹配器。任务：从候选项中找出与“查询句子”语义完全等价的一项；"
        "如果没有等价项，返回 none。\n"
        "【等价判定】仅当两句陈述同一事实/同一现象/同一步骤，且只是改写（同义词、词序、标点、大小写、细微措辞差异）。\n"
        "【以下一律不等价】\n"
        "1) 共享领域/平台词但描述不同信息（例如：'RCS 有报错代码' vs 'RCS 上 AP 离线'）\n"
        "2) 上下位或包含关系（A 包含 B、B 是 A 的子集）\n"
        "3) 不同对象/部件/指标/状态/动作（如电量低 vs AP 离线）\n"
        "4) 现象≠原因、现象≠方案、方案≠步骤\n"
        "5) 你若不确定是否等价→ 返回 none。\n"
        "仅返回严格 JSON：{\"index\": <数字索引或 null>}。不要解释。"
    )

    # ✅ 修正过的 few-shots（第4个示例必须为 index=1）
    fewshots = [
        {"role": "user", "content": "查询：机器人开不了机\n候选：\n0. 机器人无法开机\n仅输出 JSON。"},
        {"role": "assistant", "content": "{\"index\": 0}"},
        {"role": "user", "content": "查询：RCS 上确认 AP 离线\n候选：\n0. RCS 有报错代码\n仅输出 JSON。"},
        {"role": "assistant", "content": "{\"index\": null}"},
        {"role": "user", "content": "查询：检查电池连接线是否松动\n候选：\n0. 尝试重新插拔电池连接线\n仅输出 JSON。"},
        {"role": "assistant", "content": "{\"index\": null}"},
        {"role": "user", "content": "查询：RCS 上 AP 离线\n候选：\n0. RCS 有报错代码\n1. RCS 上 AP 离线\n仅输出 JSON。"},
        {"role": "assistant", "content": "{\"index\": 1}"},
        {"role": "user", "content": "查询：RCS 小车报错闪烁且颜色为粉色\n候选：\n0. RCS 有报错代码\n仅输出 JSON。"},
        {"role": "assistant", "content": "{\"index\": null}"},
        {"role": "user", "content": "查询：RCS 上 AP 离线\n候选：\n0. RCS 有报错代码\n仅输出 JSON。"},
        {"role": "assistant", "content": "{\"index\": null}"},
    ]

    usr = (
        f"查询：{query}\n\n候选：\n{numbered}\n\n"
        "仅输出 JSON：{\"index\": <数字索引或 null>}。"
    )

    messages = [{"role": "system", "content": sys}] + fewshots + [{"role": "user", "content": usr}]

    raw = _chat(messages, temperature=0.0)

    # 先尝试严格 JSON
    try:
        j = json.loads(raw.strip())
        idx = j.get("index", None)
        if isinstance(idx, int) and 0 <= idx < len(options):
            return idx
        return None
    except Exception:
        # 简单容错：数字 or none/null
        m_num = re.search(r"\b\d+\b", raw)
        if m_num:
            idx = int(m_num.group(0))
            return idx if 0 <= idx < len(options) else None
        if re.search(r"\bnone|null\b", raw, re.I):
            return None

    # --- 兜底严格等价：文本规整后完全一致才算 ---
    def _normalize(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"[\s\u3000]+", "", s)
        s = s.replace("：", ":").replace("，", ",").replace("。", ".")
        return s

    nq = _normalize(query)
    for i, d in enumerate(descs):
        if _normalize(d) == nq:
            return i
    return None

def _fallback_best(query: str, options: List[str]) -> Optional[int]:
    """
    LLM 不可用时的兜底相似度（极简 token 交集比例）。
    """
    if not options:
        return None
    q = set(re.findall(r"\w+", query.lower()))
    best, best_idx = 0.0, None
    for i, opt in enumerate(options):
        o = set(re.findall(r"\w+", str(opt).lower()))
        sim = len(q & o) / max(1, len(q | o))
        if sim > best:
            best, best_idx = sim, i
    return best_idx if best > 0 else None


def choose_best(text: str, candidates: List[str]) -> Optional[int]:
    """
    在 candidates 文本里选择最接近 text 的一项；返回索引或 None。
    """
    # 先用 LLM
    try:
        idx = _select_index(f"请选择与“{text}”语义等价的一项。", candidates)
        if idx is not None:
            return idx
        else:
            print(f"LLM 选择失败，return none")
            return None
    except Exception:
        pass
    # 再用兜底
    return None


def solution_matches_problem(solution_text: str, problem_text: str) -> Optional[bool]:
    """
    判断“这个解决方案”是否就是为“这个问题”而设的。
    返回 True/False；无法判断返回 None。
    """
    sys = "判断给定解决方案是否直接针对给定问题。只回答“是”或“不是”。"
    usr = f"问题：{problem_text}\n解决方案：{solution_text}\n是否匹配？"
    raw = _chat(
        [{"role": "system", "content": sys},
         {"role": "user",   "content": usr}],
        temperature=0.0,
    )
    s = raw.strip().splitlines()[0].strip(" \t'\"。，.!?").lower()
    if s in {"是", "对", "yes", "y", "true", "1"}:
        return True
    if s in {"不是", "否", "no", "n", "false", "0"}:
        return False
    return None

def infer_problem_from_solution(solution_text: str) -> str:
    """
    给定一个 Solution 描述，调用 LLM 归纳出对应的 Problem 描述。
    """
    sys = (
        "你是一个问题归纳器。\n"
        "我会给你一个解决方案，请你推断这个方案要解决的问题。\n"
        "要求：\n"
        "1. 输出一句简洁的问题描述。\n"
        "2. 必须是不可观测的原因或故障现象，而不是动作方案。\n"
        "3. 只输出一句话，不要解释。"
        "4. 不要返回疑问句，而是返回陈述句"
    )
    usr = f"解决方案：{solution_text}\n\n请输出对应的问题："

    raw = _chat(
        [{"role": "system", "content": sys},
         {"role": "user",   "content": usr}],
        temperature=0.0,
    )

    # 取第一行，去掉多余空格
    problem_desc = raw.strip().splitlines()[0]
    return problem_desc


def solution_solves_problem(solution_text: str, problem_text: str) -> Optional[bool]:
    """
    判断“solution_text”是否直接解决“problem_text”。True/False/None
    """
    sys = (
        "判断给定解决方案是否直接针对给定问题。\n"
        "严格：现象≠原因；包含/上下位≠相同；不同设备/字段不可混淆。\n"
        "只输出 JSON：{\"match\": true|false|null}"
    )
    usr = f"问题：{problem_text}\n方案：{solution_text}\n请判断并只输出 JSON。"
    raw = _chat(
        [{"role": "system", "content": sys},
         {"role": "user",   "content": usr}],
        temperature=0.0,
    )
    try:
        obj: Dict[str, Any] = json.loads(raw.strip())
        v = obj.get("match", None)
        if isinstance(v, bool):
            return v
        return None
    except Exception:
        s = raw.strip().splitlines()[0].strip(" \t'\"。，.!?").lower()
        if s in {"是","对","yes","y","true","1"}: return True
        if s in {"不是","否","no","n","false","0"}: return False
        return None

def _extract_json(s: str) -> str:
    """
    尝试从模型输出里提取一段 JSON（容错）。
    """
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.search(r"\{.*\}", s, flags=re.S)
    return m.group(0) if m else s

def pick_problem_index_for_solution(solution_text: str, candidate_problems: List[str]) -> Optional[int]:
    """
    在候选问题列表中，为给定解决方案选择最匹配的问题索引（0-based）。
    - candidate_problems 支持形如 "ID:描述" 的项；判断仅依据“冒号后的描述”。
    - 若无匹配返回 None。
    """
    if not candidate_problems:
        return None

    # 构造展示项 + 仅用于语义判断的描述
    display_items = []
    only_desc = []
    for i, it in enumerate(candidate_problems):
        it = str(it)
        display_items.append(f"{i}. {it}")
        only_desc.append(it.split(":", 1)[-1].strip())

    sys = (
        "你是一个匹配器。给定一个解决方案 S 和若干问题候选 P[i]，"
        "只有当 S 直接解决 P[i] 的问题时，才算匹配。\n"
        "严格要求：\n"
        "- 现象≠原因；包含/上下位关系≠相同；不同模块/字段不可混淆；\n"
        "- 不依据扩展联想，只以语义严格一致为准；\n"
        "- 候选项可能是“ID:描述”，判断仅看冒号后的描述；\n"
        "只输出 JSON：{\"index\": <数字或 null>}。"
    )
    usr = (
        f"解决方案 S：{solution_text}\n\n"
        "候选问题（编号. 项）：\n" + "\n".join(display_items) + "\n\n"
        "请只输出 JSON，如 {\"index\": 2} 或 {\"index\": null}。"
    )

    raw = _chat(
        [{"role": "system", "content": sys},
         {"role": "user",   "content": usr}],
        temperature=0.0,
    )

    # 解析 index
    try:
        obj = json.loads(_extract_json(raw))
        idx = obj.get("index", None)
        if isinstance(idx, int) and 0 <= idx < len(candidate_problems):
            return idx
    except Exception:
        pass

    # 兜底：逐一 pairwise 判断（若唯一 True，就返回该索引；若多于一个，取第一个；若没有，None）
    truths = []
    for i, desc in enumerate(only_desc):
        ok = solution_solves_problem(solution_text, desc)
        if ok is True:
            truths.append(i)
    if len(truths) == 1:
        return truths[0]
    if len(truths) > 0:
        return truths[0]
    return None

