from typing import Callable, Optional, Any, List, Tuple
from ..Node import Node, NodeType



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
        self.child_problems = child_problems or []
        self.child_features = child_features or []
        self.visited = False
        self.confirmed_positive = False
        self.interaction_callback = interaction_callback or self.default_interaction

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
                return self._next_child_node()
            else:
                self.output_callback("❌ 自动判断为负 → 返回父节点")
                return {"next_node": self.parent_node}

        # Step 2: 交互确认
        reply = self.interaction_callback(f"该特征是否为正？({self.description})")
        if reply in ("yes", "y", "true", "1", True):
            self.confirmed_positive = True
            self.output_callback("✅ 用户确认特征为正 → 进入子节点")
            return self._next_child_node()
        else:
            self.output_callback("❌ 用户确认特征为负 → 返回父节点")
            return {"next_node": self.parent_node}

    def _select_next_feature(self) -> Optional['Node']:
        """
        选择下一个要访问的子特征。
        - 默认规则：选择第一个未访问的
        - 未来可替换成 LLM 或更复杂的策略
        """
        for feature in self.child_features:
            if not feature.visited:
                return feature
        return None

    def _next_child_node(self) -> Any:
        """优先访问子问题，然后子特征"""
        # 先找未访问的问题
        for problem, link_mode in self.child_problems:
            if not problem.visited:
                # 根据链接模式更新 problem 的模式
                problem.mode = link_mode.lower()
                self.output_callback(f"🔍 进入子问题: {problem.node_id} (模式: {problem.mode})")
                return {"next_node": problem}

        # 再找要访问的特征（由选择函数决定）
        target_feature = self._select_next_feature()
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


