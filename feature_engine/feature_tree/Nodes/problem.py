from typing import Callable, Optional, Any, List
from ..node import Node, NodeType  # 按实际路径调整


class ProblemNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        parent_feature: 'Node',  # 父特征对象引用
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
        self.parent_feature = parent_feature
        self.mode = mode.lower()
        self.solutions = solutions or []           # List[SolutionNode]
        self.child_features = child_features or [] # List[FeatureNode]
        self.visited = False
        self.resolved = False
        
        if interaction_callback is not None:
            self.set_interaction_callback(interaction_callback)

    # === 新增：按文档规则添加连边 ===
    def add_node(self, node: 'Node') -> None:
        """
        将子节点连到当前 Problem：
        - 允许：PROBLEM -> SOLUTION（会设置 solution.parent_problem = self）
        - 允许：PROBLEM -> FEATURE（会设置 feature.parent_node = self）
        - 禁止：PROBLEM -> PROBLEM（规则禁止问题指向问题）
        - 禁止：PROBLEM -> SUCCESS（SUCCESS 只能由 SOLUTION 指向）
        - 说明：到 FAILURE 的边可随时跳转，通常不在结构中存储，这里不支持显式添加
        """
        if not isinstance(node, Node):
            raise TypeError("ProblemNode.add_node: 需要传入 Node 实例")

        # 去重函数
        def _exists_in(lst: List[Node], n: Node) -> bool:
            return any(x.node_id == n.node_id for x in lst)

        if node.node_type == NodeType.SOLUTION:
            if _exists_in(self.solutions, node):
                self.output_callback(f"⚠️ Solution {node.node_id} 已存在于 {self.node_id} 的子方案中，跳过")
                return
            self.solutions.append(node)
            # 反向父指针
            try:
                setattr(node, "parent_problem", self)
            except Exception:
                pass
            return

        if node.node_type == NodeType.FEATURE:
            if _exists_in(self.child_features, node):
                self.output_callback(f"⚠️ Feature {node.node_id} 已存在于 {self.node_id} 的子特征中，跳过")
                return
            self.child_features.append(node)
            # 反向父指针
            try:
                setattr(node, "parent_node", self)
            except Exception:
                pass
            return

        if node.node_type == NodeType.PROBLEM:
            raise ValueError("规则违反：PROBLEM 不可指向 PROBLEM")

        if node.node_type == NodeType.SUCCESS:
            raise ValueError("规则违反：SUCCESS 必须由 SOLUTION 指向，PROBLEM 不可直连 SUCCESS")

        if node.node_type == NodeType.FAILURE:
            raise ValueError("设计约定：到 FAILURE 的终止边不在结构中存储，请在运行时跳转")

        if node.node_type == NodeType.ORIGIN:
            raise ValueError("规则违反：ORIGIN 只能作为源点，不可作为 PROBLEM 的子节点")

        raise ValueError(f"未支持的连边类型：PROBLEM -> {node.node_type.value}")

    def default_interaction(self, prompt: str) -> Any:
        self.output_callback(f"💬 {prompt}")
        return input("问题是否已解决？(yes/no): ").strip().lower()

    def _select_next_feature(self) -> Optional['Node']:
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

        # Step 1: 未访问的 Solution
        for sol_node in self.solutions:
            if not getattr(sol_node, "visited", False):
                self.output_callback(f"🛠 选择解决方案: {sol_node.node_id}")
                return {"next_node": sol_node}

        # Step 2: 未访问的 Feature
        target_feature = self._select_next_feature()
        if target_feature:
            self.output_callback(f"🔍 进入子特征: {target_feature.node_id}")
            return {"next_node": target_feature}

        # Step 3: 无路可走 → hard/soft
        if self.mode == "hard":
            self.output_callback("❌ 硬问题无解 → 跳转 Failure")
            return {"next_node": "FAILURE"}
        else:
            self.output_callback(f"ℹ 软问题无路可走 → 回退到母特征 {self.parent_feature.node_id}")
            return {"next_node": self.parent_feature}

    def _check_problem_resolved(self) -> bool:
        """纯交互确认父特征是否已消失/问题可视为已解决"""
        if getattr(self.parent_feature, "resolved", False):
            return True
        reply = self.interaction_callback(f"特征《{self.parent_feature.description}》是否已经消失？")
        return reply in ("yes", "y", "true", "1", True)
