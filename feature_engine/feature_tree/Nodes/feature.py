from typing import Callable, Optional, Any, List, Tuple
from ..node import Node, NodeType
from feature_engine.llm_client.llm import llm_select

class FeatureNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        expected_state: bool,  # True / False
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
        self.expected_state = expected_state
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
            effective_mode = "hard" if len(self.child_problems) == 0 else "soft"
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
        """
        尝试从 chat_log 自动判断特征是否为正
        - 返回 True 表示正
        - 返回 False 表示负
        - 返回 None 表示无法判断
        """
        # TODO: 在这里实现你的自动判断逻辑
        return None

    def process_next_node(self, node: 'Node', chat_log: Any) -> Any:
        self.visited = True
        self.output_callback(f"📌 进入特征: {self.description} (期望状态: {self.expected_state})")

        # Step 1: 自动判断
        auto_result = self._auto_judge_from_chatlog(chat_log)
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
        使用 LLM 在未访问的子特征中选择一个。
        - 优先：llm_select 依据描述挑选
        - 回退：若 LLM 不可用/解析失败，选择第一个未访问的
        """
        candidates = [f for f in self.child_features if not getattr(f, "visited", False)]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        options = [f"{c.node_id}：{getattr(c, 'description', '')}" for c in candidates]

        if llm_select is None:
            self.output_callback("ℹ️ LLM 不可用，采用默认顺序选择子特征")
            return candidates[0]

        prompt = (
            f"当前位于特征《{self.description}》。请从候选子特征中选出最优先检查的一项。"
            "如果我现有的聊天记录已经可以判断该子特征是否为正，则选择这个子特征。"
            "仅输出一个数字序号（从 0 开始）。"
            f"我现有的聊天记录是：{chat_log}\n\n"
        )

        try:
            idx, raw = llm_select(prompt, options)
            self.output_callback(f"🤖 LLM 选择结果：{idx} | 原始: {raw!r}")
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                return candidates[idx]
        except Exception as e:
            self.output_callback(f"⚠️ LLM 选择失败，回退默认策略：{e}")

        return candidates[0]

    def _next_child_node(self, chat_log: Any) -> Any:
        """优先访问子问题，然后子特征"""
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
        target_feature = self._select_next_feature(chat_log)  # ✅ 传入 chat_log
        if target_feature:
            self.output_callback(f"🔍 进入子特征: {target_feature.node_id}")
            return {"next_node": target_feature}

        # 如果都访问过
        if self.parent_node.node_type == NodeType.ORIGIN:
            self.output_callback("❌ 父节点是 Origin → 跳转到 Failure")
            return {"next_node": "FAILURE"}
        else:
            self.output_callback(f"↩ 所有子节点已访问，返回父节点 {self.parent_node.node_id}")
            return {"next_node": self.parent_node}
