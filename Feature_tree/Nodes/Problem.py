from typing import Callable, Optional, Any, List
from ..Node import Node, NodeType  # 按实际路径调整


class ProblemNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        parent_feature: 'Node',  # 改成父节点对象引用
        mode: str = "soft",
        solutions: Optional[List['Node']] = None,
        child_features: Optional[List['Node']] = None,
        interaction_callback: Optional[Callable[[str], Any]] = None,
        handler: Optional[Callable[['Node', Any], Any]] = None,
        output_callback: Optional[Callable[[str], None]] = None
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PROBLEM,
            description=description,
            handler=handler,
            output_callback=output_callback
        )
        self.parent_feature = parent_feature  # 直接存父特征对象
        self.mode = mode.lower()
        self.solutions = solutions or []      # 直接存 SolutionNode 对象
        self.child_features = child_features or []  # 直接存 FeatureNode 对象
        self.visited = False
        self.resolved = False
        self.interaction_callback = interaction_callback or self.default_interaction

    def default_interaction(self, prompt: str) -> Any:
        self.output_callback(f"💬 {prompt}")
        return input("问题是否已解决？(yes/no): ").strip().lower()

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

    def process_next_node(self) -> Any:
        self.output_callback(f"📌 进入问题: {self.description} (模式: {self.mode})")

        if self.visited:
            if self._check_problem_resolved():
                self.output_callback(f"🔙 问题已解决 → 回退到母特征 {self.parent_feature.node_id}")
                return {"next_node": self.parent_feature}
            else:
                self.output_callback("⚠ 问题未解决，继续寻找可执行节点")

        # 标记已访问
        self.visited = True

        # Step 1: 找第一个未访问的 Solution
        for sol_node in self.solutions:
            if not sol_node.visited:
                self.output_callback(f"🛠 选择解决方案: {sol_node.node_id}")
                return {"next_node": sol_node}

        # Step 2: 找第一个未访问的 Feature
        target_feature = self._select_next_feature()
        if target_feature:
            self.output_callback(f"🔍 进入子特征: {target_feature.node_id}")
            return {"next_node": target_feature}

        # Step 3: 无路可走 → 根据 hard/soft 处理
        if self.mode == "hard":
            self.output_callback("❌ 硬问题无解 → 跳转 Failure")
            return {"next_node": "FAILURE"}  # 或直接返回 FailureNode
        else:
            self.output_callback(f"ℹ 软问题无路可走 → 回退到母特征 {self.parent_feature.node_id}")
            return {"next_node": self.parent_feature}




    def _check_problem_resolved(self) -> bool:
        """
        检查父特征是否消失（纯交互版）
        """
        # 如果父节点有自己的 resolved 状态，可以直接用
        if getattr(self.parent_feature, "resolved", False):
            return True

        reply = self.interaction_callback(
            f"特征 {self.parent_feature.description} 是否已经消失？"
        )
        return reply in ("yes", "y", "true", "1", True)
