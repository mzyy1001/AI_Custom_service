from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
import json

from state import State, StateCategory  # 你的 state.py

# ===== 图结构 =====
@dataclass
class Graph:
    states: Dict[str, State] = field(default_factory=dict)
    human_escalation_id: str = "human_escalation"

    def ensure_state(self, state: State) -> None:
        self.states[state.state_id] = state

    def get_state(self, state_id: str) -> Optional[State]:
        return self.states.get(state_id)

    def get_or_create_human(self) -> State:
        st = self.get_state(self.human_escalation_id)
        if st is None:
            st = State(
                state_id=self.human_escalation_id,
                category=StateCategory.ESCALATE,
                description="人工客服"
            )
            self.ensure_state(st)
        return st


# ===== 路由 LLM 接口（实现里去写 prompt/候选约束） =====
class RouterLLM(Protocol):
    def choose_next(self, *, current_state: State, all_states: List[State],
                    user_text: str) -> Dict[str, Any]:
        """
        必须返回：
          - state_id: 目标候选ID（或 None / "abstain"）
        可选返回：
          - confidence: 0~1 置信度
          - rationale: 说明
        """
        ...


# ===== 生产引擎（只做安全转移） =====
class Engine:
    def __init__(self, graph: Graph, router: RouterLLM, *, threshold: float = 0.6,
                 allow_jump_outside_candidates: bool = False):
        self.graph = graph
        self.router = router
        self.threshold = threshold
        self.allow_jump = allow_jump_outside_candidates  # 默认为 False，更安全

    def _p(self, msg: str):
        self._log(f"[ENGINE {self._trace}] {msg}")

    def transition(self, current_state_id: str, user_text: str) -> State:
        # 取当前状态
        current = self.graph.get_state(current_state_id)
        if current is None:
            raise KeyError(f"unknown state: {current_state_id}")

        # 打印当前状态 & 用户输入
        desc = (current.description or "").strip().replace("\n", " ")
        if not self._show_desc and len(desc) > 80:
            desc = desc[:77] + "..."
        self._p(f"STATE {current.state_id} ({current.category.name}) | user='{user_text}'"
                + (f" | desc='{desc}'" if self._show_desc and desc else ""))

        # 候选
        candidate_ids = [t.to_state_id for t in current.transitions]
        candidates: List[State] = [self.graph.get_state(sid) for sid in candidate_ids if self.graph.get_state(sid)]
        cand_str = ", ".join(
            f"{s.state_id}:{s.category.name}" + (f"({len((s.description or '').split())}wds)" if self._show_desc else "")
            for s in candidates
        ) or "<none>"
        self._p(f"candidates=[{cand_str}]")

        # 路由调用
        decision: Dict[str, Any] = self.router.choose_next(
            current_state=current,
            all_states=candidates,
            user_text=user_text
        ) or {}

        # 打印路由结果
        try:
            pretty = json.dumps(decision, ensure_ascii=False)
        except Exception:
            pretty = str(decision)
        self._p(f"router_decision={pretty}")

        target_id: Optional[str] = decision.get("state_id")
        try:
            conf: float = float(decision.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        # 低置信或无目标
        if not target_id or conf < self.threshold:
            self._p(f"-> ESCALATE due to abstain_or_low_confidence (target_id={target_id!r}, conf={conf:.3f}, th={self.threshold:.3f})")
            return self._escalate(current, reason="abstain_or_low_confidence")

        # 非候选跳转但禁跳
        if not self.allow_jump and target_id not in candidate_ids:
            self._p(f"-> ESCALATE due to target_not_in_candidates (target_id={target_id!r})")
            return self._escalate(current, reason="target_not_in_candidates")

        # 取目标
        target = self.graph.get_state(target_id)
        if target is None:
            self._p(f"-> ESCALATE due to target_missing (target_id={target_id!r})")
            return self._escalate(current, reason="target_missing")

        # 成功跳转
        self._p(f"-> NEXT {target.state_id} ({target.category.name})")
        return target

    def _escalate(self, current: State, reason: str) -> State:
        self._p(f"ESCALATE from {current.state_id} reason={reason}")
        human = self.graph.get_or_create_human()
        if not current.has_edge_to(human.state_id):
            current.add_transition(
                to_state_id=human.state_id,
                mark=f"生产转人工：{reason}",
                note="auto_edge_production"
            )
        return human
