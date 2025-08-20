from typing import Callable, Optional, Any, List, Tuple
from ..node import Node, NodeType
from feature_engine.llm_client.llm import llm_select
from feature_engine.llm_client.llm_produce import pick_child_feature_index, llm_yes_no

class FeatureNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        parent_node: Node,
        child_problems: Optional[List[Tuple[Node, str]]] = None,
        child_features: Optional[List['Node']] = None,
        interaction_callback: Optional[Callable[[str], Any]] = None,
        handler: Optional[Callable[['Node', Any], Any]] = None,
        output_callback: Optional[Callable[[str], None]] = None
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.FEATURE,
            description=description,
            handler=handler,
            output_callback=output_callback
        )
        self.expected_state = None
        self.parent_node = parent_node
        self.child_problems = child_problems or []               # List[Tuple[ProblemNode, "hard"/"soft"]]
        self.child_features = child_features or []               # List[FeatureNode]
        self.visited = False
        self.confirmed_positive = False
        self.interaction_callback = interaction_callback or self.default_interaction

    # --- 只允许添加 Problem 或 Feature 的连边；并校验/自动设置 link_mode 规则 ---
    def add_node(self, node: 'Node', *, link_mode: Optional[str] = None) -> None:
        """
        将子节点连到当前 Feature：
        - FEATURE -> PROBLEM：忽略外部传入的 link_mode，采用固定策略：
            * 若当前尚无任何子 Problem（即本节点添加后成为唯一的 Problem），则设置为 'hard'
            * 若已存在 >=1 个子 Problem，则设置为 'soft'
          并设置 problem.parent_feature = self、problem.mode = 该模式
        - FEATURE -> FEATURE：忽略 link_mode，并设置 child.parent_node = self
        - 其他类型一律报错（遵循文档规则：FEATURE 不可直达 SOLUTION / SUCCESS / FAILURE / ORIGIN）
        """
        if not isinstance(node, Node):
            raise TypeError("FeatureNode.add_node: 需要传入 Node 实例")

        def _exists_in_features(n: Node) -> bool:
            return any(c.node_id == n.node_id for c in self.child_features)

        def _exists_in_problems(n: Node) -> bool:
            return any(p.node_id == n.node_id for (p, _) in self.child_problems)

        if node.node_type == NodeType.PROBLEM:
            if _exists_in_problems(node):
                self.output_callback(f"⚠️ Problem {node.node_id} 已存在于子问题列表，跳过")
                return
            # 写死逻辑：本 Feature 下第一个 Problem 为 hard，其余为 soft
            effective_mode = "soft"
            self.child_problems.append((node, effective_mode))
            try:
                setattr(node, "parent_feature", self)
                setattr(node, "mode", effective_mode)
            except Exception:
                pass
            self.output_callback(f"🔗 FEATURE→PROBLEM 采用固定策略: {effective_mode}（已连接 {node.node_id}）")

        elif node.node_type == NodeType.FEATURE:
            if _exists_in_features(node):
                self.output_callback(f"⚠️ Feature {node.node_id} 已存在于子特征列表，跳过")
                return
            self.child_features.append(node)
            try:
                setattr(node, "parent_node", self)
            except Exception:
                pass
            self.output_callback(f"🔗 FEATURE→FEATURE 已连接 {node.node_id}")

        else:
            raise ValueError(
                f"规则违反：FEATURE 只能连接到 PROBLEM 或 FEATURE，"
                f"不允许连接到 {node.node_type.value}"
            )

    def default_interaction(self, prompt: str) -> Any:
        """默认交互"""
        self.output_callback(f"💬 {prompt}")
        return input("该特征是否为正？(yes/no): ").strip().lower()

    def _auto_judge_from_chatlog(self, chat_log: Any) -> Optional[bool]:
        return llm_yes_no(self.description, chat_log)

    def set_expected_state(self, state: bool) -> None:
        self.expected_state = state

    def process_next_node(self, chat_log: Any) -> Any:
        self.visited = True
        self.output_callback(f"📌 进入特征: {self.description} (期望状态: {self.expected_state})")

        # Step 1: 自动判断
        if self.expected_state is None:
            auto_result = self._auto_judge_from_chatlog(chat_log)
            self.expected_state = auto_result

        if auto_result is not None:
            if auto_result:
                self.confirmed_positive = True
                self.output_callback("✅ 自动判断为正 → 进入子节点")
                return self._next_child_node(chat_log)  # ✅ 传入 chat_log
            else:
                self.output_callback("❌ 自动判断为负 → 返回父节点")
                return {"next_node": self.parent_node}

        # Step 2: 交互确认
        reply = self.interaction_callback(f"该特征是否为正？({self.description})")
        if reply in ("yes", "y", "true", "1", True):
            self.confirmed_positive = True
            self.output_callback("✅ 用户确认特征为正 → 进入子节点")
            return self._next_child_node(chat_log)  # ✅ 传入 chat_log
        else:
            self.output_callback("❌ 用户确认特征为负 → 返回父节点")
            return {"next_node": self.parent_node}

    def _select_next_feature(self, chat_log: Any) -> Optional['Node']:
        """
        使用外部 LLM 路由器在未访问的子特征中选择一个。
        - 候选：仅未访问
        - LLM 返回 None 时回退到第一个未访问
        """
        candidates = [f for f in self.child_features if not getattr(f, "visited", False)]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        options = [f"{c.node_id}:{getattr(c, 'description', '')}" for c in candidates]
        try:
            idx = pick_child_feature_index(self.description, options, chat_log)
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                return candidates[idx]
        except Exception as e:
            self.output_callback(f"⚠️ LLM 选择失败，回退默认策略：{e}")

        return candidates[0]

    def _next_child_node(self, chat_log: Any) -> Any:
        """优先访问子问题，然后子特征"""


        target_feature = self._select_next_feature(chat_log) 
        if target_feature:
            self.output_callback(f"🔍 进入子特征: {target_feature.node_id}")
            return {"next_node": target_feature}


        # 先找未访问的问题
        for problem, link_mode in self.child_problems:
            if not getattr(problem, "visited", False):
                try:
                    problem.mode = link_mode.lower()
                except Exception:
                    pass
                self.output_callback(f"🔍 进入子问题: {problem.node_id} (模式: {getattr(problem, 'mode', link_mode)})")
                return {"next_node": problem}

        # 再找要访问的特征（由选择函数决定）
        
        # 如果都访问过
        if self.parent_node.node_type == NodeType.ORIGIN:
            self.output_callback("❌ 父节点是 Origin → 跳转到 Failure")
            return {"next_node": "FAILURE"}
        else:
            if self.expected_state == True:
                self.output_callback("✅ 特征仍为正 → 跳转到失败节点")
                return {"next_node": "FAILURE"}
            else:
                self.output_callback("❌ 特征不再为正 → 返回父节点")
                self.expected_state = False
                return {"next_node": self.parent_node}
