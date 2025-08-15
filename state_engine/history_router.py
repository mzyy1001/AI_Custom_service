# history_router.py
from __future__ import annotations
from typing import Dict, Any, List
from state import State

class HistoryAwareRouter:
    """把历史对话注入到 user_text 前缀里，再委托给 base_router"""
    def __init__(self, base_router):
        self.base = base_router
        self.hist: List[str] = []

    def reset(self):
        self.hist.clear()

    def push(self, utterance: str):
        self.hist.append(utterance)

    def choose_next(self, *, current_state: State, all_states: List[State],
                    user_text: str) -> Dict[str, Any]:
        print(f"History: {self.hist}")
        if self.hist:
            # 将完整历史注入，最后一句仍视为“当前输入”
            prefix = "【历史对话，按时间顺序】\n" + "\
".join(f"{i+1}. {u}" for i, u in enumerate(self.hist)) + "\n\n【当前用户输入】\n" + (user_text or "")
        else:
            prefix = user_text or ""
        return self.base.choose_next(current_state=current_state,
                                     all_states=all_states,
                                     user_text=prefix)
