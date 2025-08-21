import re
from typing import Dict, List, Optional, Any
from feature_engine.llm_client.chat import _chat
import re, json, uuid
from typing import List, Optional, Any, Dict


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
        # print(f"LLM prompt: {sys} {usr}")
        m = re.search(r'"answer"\s*:\s*"(\w+)"', raw)
        # print(f"LLM raw response: {raw}")  # Debug output
        
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



# ---------- IO 适配：从 chat_log 拿到 out/in ----------
class _ConsoleIO:
    def out(self, text: str) -> None:
        print(text)
    def inp(self, prompt: str) -> str:
        return input(prompt)

def _get_io(chat_log: Any):
    # 1) {"io": obj} 且 obj 有 out/in
    if isinstance(chat_log, dict) and "io" in chat_log:
        io = chat_log["io"]
        if hasattr(io, "out") and hasattr(io, "inp"):
            return io
    # 2) {"out": callable, "inp": callable}
    if isinstance(chat_log, dict) and callable(chat_log.get("out")) and callable(chat_log.get("inp")):
        class _Wrap:
            def __init__(self, o, i): self._o, self._i = o, i
            def out(self, text): self._o(text)
            def inp(self, prompt): return self._i(prompt)
        return _Wrap(chat_log["out"], chat_log["inp"])
    # 3) 退化为控制台
    return _ConsoleIO()

# ---------- LLM：给出所有“可能项”的原始索引 ----------
def _llm_plausible_indices(current_desc: str, candidates: List[str], chat_log: Any) -> List[int]:
    numbered = "\n".join(f"{i}. {opt}" for i, opt in enumerate(candidates))
    sys = (
        "你是一个诊断分流器。基于当前特征与上下文，找出所有“可能推进诊断”的候选索引（0-based）。"
        "仅输出严格 JSON：{\"plausible\": [int, int, ...]}，按可能性从高到低；若无任何匹配，输出空数组。"
    )
    usr = (
        f"当前特征：{current_desc}\n"
        f"上下文：{chat_log}\n\n"
        f"候选（0-based）：\n{numbered}\n\n"
        "仅输出 JSON：{\"plausible\": [0,2,...]}（可为空数组）。"
    )
    try:
        raw = _chat(
            [{"role": "system", "content": sys},
             {"role": "user",   "content": usr}],
            temperature=0.0,
        ).strip()
        data = json.loads(raw)
        arr = data.get("plausible", [])
        seen, out = set(), []
        for i in arr:
            if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen:
                seen.add(i); out.append(i)
        return out
    except Exception:
        return []

# ---------- 解析用户选择（局部 0..k-1） ----------
def _rule_parse_local_choice(user_text: str, local_opts: List[str]) -> Optional[int]:
    s = (user_text or "").strip()
    if not s:
        return None
    # 接受 none / 无 / 不在内 等表达 => None
    if re.search(r"\bnone\b|不在|都不是|无|没有", s, flags=re.I):
        return None
    # 数字优先（先 0-based，再 1-based）
    m = re.search(r"\b(\d{1,3})\b", s)
    if m:
        n = int(m.group(1))
        if 0 <= n < len(local_opts):
            return n
        if 1 <= n <= len(local_opts):
            return n - 1
    # 关键词/子串
    sl = s.lower()
    best = (-1, -1)
    for i, opt in enumerate(local_opts):
        ol = str(opt).lower()
        if ol in sl:
            return i
        words = set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fa5]+", ol))
        hits = sum(1 for w in words if w and w in sl)
        if hits > best[0]:
            best = (hits, i)
    return best[1] if best[0] > 0 else None

def _llm_parse_local_choice(user_text: str, local_opts: List[str]) -> Optional[int]:
    numbered = "\n".join(f"{i}. {o}" for i, o in enumerate(local_opts))
    sys = (
        "你是一个严格的索引解析器。根据用户文字，在给定的局部候选中选出 0-based 索引。"
        "仅输出 JSON：{\"index\": <int 或 null>}。若无法判断或用户表达“都不是”，输出 null。"
        "若用户给的是 1-based 号码，需要换算为 0-based。"
    )
    usr = (
        f"局部候选：\n{numbered}\n\n"
        f"用户回复：{user_text}\n\n"
        "仅输出 JSON：{\"index\": 0} 或 {\"index\": null}"
    )
    try:
        raw = _chat(
            [{"role": "system", "content": sys},
             {"role": "user",   "content": usr}],
            temperature=0.0,
        ).strip()
        m = re.search(r'"index"\s*:\s*(\d+|null)', raw)
        if not m:
            return None
        tok = m.group(1)
        if tok == "null":
            return None
        idx = int(tok)
        return idx if 0 <= idx < len(local_opts) else None
    except Exception:
        return None

# ---------- 主函数（签名不变 | 同步一站式交互） ----------
def pick_child_feature_index(current_desc: str,
                             candidates: List[str],
                             chat_log: Any) -> Optional[int]:
    """
    一次性完成：
    1) 基于 chat_log + current_desc 调用 LLM 选出所有可能项；
    2) 0 个可能 → 返回 None；
       1 个可能 → 返回其原始索引；
       多个可能 → 立刻用 out/in 与用户交互，解析后返回原始索引；无法解析 → None。
    """
    if not candidates:
        return None

    plausible = _llm_plausible_indices(current_desc, candidates, chat_log)

    # 0 个可能 ⇒ 直接 None
    if not plausible:
        return None

    # 1 个可能 ⇒ 直接返回
    if len(plausible) == 1:
        return plausible[0]

    # 多个可能 ⇒ 立刻问用户（同步交互）
    io = _get_io(chat_log)
    local_texts = [candidates[i] for i in plausible]
    listing = "\n".join(f"{i}. {txt}" for i, txt in enumerate(local_texts))
    msg = "检测到多个可能的子特征，请选择其一：\n" + listing

    io.out(msg)
    user_text = io.inp("请输入序号（支持 0/1-based）或关键词；输空行/none 表示都不是： ")



    if isinstance(chat_log, list):
        chat_log.append({"role": "assistant", "content": msg})
        chat_log.append({"role": "user", "content": user_text})
    elif isinstance(chat_log, dict) and "messages" in chat_log:
        chat_log["messages"].append({"role": "assistant", "content": msg})
        chat_log["messages"].append({"role": "user", "content": user_text})


    if user_text.strip().isdigit():
        local_idx = int(user_text.strip())
        if 0 <= local_idx < len(local_texts):
            orig_idx = plausible[local_idx]
            return orig_idx
        else:
            return None

    local_idx = _rule_parse_local_choice(user_text, local_texts)
    if local_idx is None:
        local_idx = _llm_parse_local_choice(user_text, local_texts)
    if local_idx is None:
        return None

    # 映射回原始索引
    orig_idx = plausible[local_idx]
    return orig_idx