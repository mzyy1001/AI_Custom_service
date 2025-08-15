from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Any


# 1) 状态类别：把你关心的“解决方法/决策层”等纳入类型集合
class StateCategory(Enum):
    # “决策层”：只负责路由/判断，不直接回答
    DECISION = auto()
    # “解决方法”：直接生成最终答复/方案（可走 LLM/RAG）
    SOLUTION = auto()
    # 收集槽位/信息
    COLLECT = auto()
    # 执行动作（调API/工具）
    ACTION = auto()
    # 展示说明性话术
    EXPLAIN = auto()
    # 转人工/异常
    ESCALATE = auto()
    # 结束
    END = auto()


# 2) 转移条件：支持“文字mark”（给 LLM 用）+ 可选的可执行谓词（本地规则）
@dataclass
class TransitionCondition:
    # 非结构化“路由标记”，交给路由LLM对齐用户输入（你要的 mark）
    mark: str

    # （可选）本地可执行规则：返回 True/False；如果你不想用规则，留空即可
    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None

    # （可选）权重/优先级；当 LLM 同时命中多条时可作为并列打分的加权项
    weight: float = 1.0


# 3) 转移：目标 + 条件（一个状态可以有多条边）
@dataclass
class Transition:
    to_state_id: str                     # 目标状态ID
    condition: TransitionCondition       # 转移条件
    note: str = ""                       # 备注（可存审计/可观测信息）


# 4) 状态：类别 + 元数据 + 多条转移
@dataclass
class State:
    state_id: str
    category: StateCategory
    description: str = ""

    # 针对不同类别可放不同元数据（你可以自定义约定）：
    # - 对于 SOLUTION：比如 {"solver": "llm|rag|template", "template": "..."}
    # - 对于 ACTION：   比如 {"tool": "get_order", "params": {"id": "{{slots.order_id}}"}}
    # - 对于 COLLECT：  比如 {"slots": [{"name":"order_id","validate":"..."}]}
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 多条可选的转移边
    transitions: List[Transition] = field(default_factory=list)

    def add_transition(self, to_state_id: str, mark: str,
                       predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
                       weight: float = 1.0, note: str = "") -> "State":
        self.transitions.append(
            Transition(
                to_state_id=to_state_id,
                condition=TransitionCondition(mark=mark, predicate=predicate, weight=weight),
                note=note
            )
        )
        return self
