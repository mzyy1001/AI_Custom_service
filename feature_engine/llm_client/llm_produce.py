import re
from typing import Dict, List, Optional, Any
from feature_engine.llm_client.chat import _chat

def llm_yes_no(prompt: str, chat_log: List[Dict[str, str]]) -> Optional[bool]:
    """
    基于“当前是/否问题 prompt + 最近一次用户自由文本回答”推断 yes/no。
    - True  -> yes
    - False -> no
    - None  -> 不确定（交给上层继续追问或澄清）
    """

    sys = (
        "你是一个严格的二值判定器。只根据R判断Q的答案：yes/no/unsure。\n"
            "【硬性规则】：\n"
            "1) 判定必须基于R中的明确文本证据，不得外推或使用常识。\n"
            "2) R未提及或无法得出明确结论→ 返回 unsure（“未提及”≠“否”）。\n"
            "仅输出严格JSON：{\"answer\":\"yes\"|\"no\"|\"unsure\"}。"
    )
    usr = f"Q: {prompt}\nR: {chat_log}\n"

    try:
        raw = _chat(
            [{"role": "system", "content": sys},
             {"role": "user",   "content": usr}],
            temperature=0.0,
        ).strip()
        print(f"LLM prompt: {sys} {usr}")
        m = re.search(r'"answer"\s*:\s*"(\w+)"', raw)
        print(f"LLM raw response: {raw}")  # Debug output
        
        if m:
            ans = m.group(1).lower()
            if ans == "yes":
                return True
            if ans == "no":
                return False
            return None

        # 容错（极少数模型可能直接回 yes/no）
        s = raw.strip().lower()
        if s in {"yes", "y", "是"}:
            return True
        if s in {"no", "n", "否"}:
            return False

    except Exception:
        # 出错则交由上层继续处理/追问
        pass

    return None


def _llm_yes_no_from_user_text(prompt: str, user_texts: List[str], chat_log: List[Dict[str, str]]) -> Optional[bool]:
    """
    把“节点提问 + 用户自由回答(可多轮)”交给 LLM，要求严格输出:
    {"decision": "yes"|"no"|"ask"}
    - yes  -> True
    - no   -> False
    - ask  -> None（仍需追问）
    """
    joined = "\n".join(f"- {t}" for t in user_texts)
    sys = (
        "你是一个严格的判断器。请根据用户的自然语言回答，判断下面这个“判断性问题”是否成立。\n"
        "只能输出 JSON：{\"decision\":\"yes\"|\"no\"|\"ask\"}\n"
        "规则：\n"
        "1) 当且仅当用户的描述明确支持该判断为真时，输出 yes。\n"
        "2) 当明确为假时，输出 no。\n"
        "3) 如果信息不足或含糊，输出 ask。\n"
        "4) 不要输出解释或其它字段。\n"
    )
    usr = f"判断性问题：{prompt}\n用户回答（按时间倒序合并）:\n{joined}\n仅输出 JSON：{{\"decision\":\"yes\"|\"no\"|\"ask\"}}"

    try:
        raw = _chat(
            [{"role": "system", "content": sys},
             {"role": "user",   "content": usr}],
            temperature=0.0,
        ).strip()
    except Exception:
        raw = ""

    # 先尝试严格 JSON
    m = re.search(r'\"decision\"\s*:\s*\"(yes|no|ask)\"', raw, flags=re.I)
    if not m:
        # 容错：简单关键词
        low = raw.lower()
        if "yes" in low or "是" in raw:
            return True
        if "no" in low or "否" in raw or "不是" in raw:
            return False
        return None

    dec = m.group(1).lower()
    if dec == "yes":
        return True
    if dec == "no":
        return False
    return None  # ask


def _llm_followup_question(prompt: str, user_texts: List[str], chat_log: List[Dict[str, str]]) -> str:
    """
    请 LLM 产出“一句最小必要的澄清问题”，引导用户给出能判定 yes/no 的信息。
    """
    joined = "\n".join(f"- {t}" for t in user_texts)
    sys = (
        "你是对话澄清器。请基于判断性问题与当前用户回答，提出一条最小必要的追问，"
        "使用户能给出足以判断 yes/no 的信息。"
        "只输出一句话的追问，不要解释。"
    )
    usr = f"判断性问题：{prompt}\n当前用户回答：\n{joined}\n请输出一句精确的追问："
    try:
        q = _chat(
            [{"role": "system", "content": sys},
             {"role": "user",   "content": usr}],
            temperature=0.0,
        ).strip().splitlines()[0]
    except Exception:
        q = "为了判断，请用“是/否 + 简短原因”回答。"
    return q
# =========================================
# 交互适配（同一个回调注入所有节点）
# =========================================


def pick_child_feature_index(current_desc: str,
                             candidates: List[str],
                             chat_log: Any) -> Optional[int]:
    """
    基于当前特征描述 + 最近聊天上下文，在候选子特征中选一个。
    返回候选索引（0-based）；无法判断返回 None。
    """
    if not candidates:
        return None

    numbered = "\n".join(f"{i}. {opt}" for i, opt in enumerate(candidates))
    sys = (
        "你是一个子特征路由器。任务：从候选子特征中选择“下一个最有信息量/最可能推进诊断”的一项。\n"
        "只输出严格 JSON：{\"index\": <数字索引或 null>}，不要解释。\n"
        "若上下文不足以判断，返回 null。"
        "如果聊天上下文已经明确包含了某一个字特征的同义信息，直接返回对应的索引。"
    )
    usr = (
        f"当前特征：{current_desc}\n"
        f"聊天上下文（最近若干条）：\n{chat_log}\n\n"
        f"候选子特征（按序号）：\n{numbered}\n\n"
        "仅输出 JSON：{\"index\": <数字索引或 null>}。"
    )

    try:
        raw = _chat(
            [{"role": "system", "content": sys},
             {"role": "user",   "content": usr}],
            temperature=0.0,
        ).strip()
        # 先按 JSON 取 "index"
        m = re.search(r'"index"\s*:\s*(\d+|null)', raw)
        if m:
            tok = m.group(1)
            if tok == "null":
                return None
            idx = int(tok)
            return idx if 0 <= idx < len(candidates) else None
        # 容错：只给了数字
        m2 = re.search(r'\b(\d{1,3})\b', raw)
        if m2:
            idx = int(m2.group(1))
            return idx if 0 <= idx < len(candidates) else None
    except Exception:
        pass
    return None
