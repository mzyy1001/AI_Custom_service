from typing import Callable, Optional, Any, List
from ..node import Node, NodeType


class OriginNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        child_features: Optional[List['Node']] = None,
        interaction_callback: Optional[Callable[[str], Any]] = None,
        handler: Optional[Callable[['Node', Any], Any]] = None,
        output_callback: Optional[Callable[[str], None]] = None
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.ORIGIN,
            description=description,
            handler=handler,
            output_callback=output_callback
        )
        self.child_features = child_features or []
        self.visited = False
        if interaction_callback is not None:
            self.set_interaction_callback(interaction_callback)

    # 只允许添加 FEATURE 子节点；否则报错
    def add_node(self, node: 'Node') -> None:
        if not isinstance(node, Node):
            raise TypeError("OriginNode.add_node: 需要传入 Node 实例")
        if node.node_type != NodeType.FEATURE:
            raise ValueError("规则违反：ORIGIN 只能连接到 FEATURE 节点")

        # 去重
        if any(n.node_id == node.node_id for n in self.child_features):
            self.output_callback(f"⚠️ 节点 {node.node_id} 已存在于 ORIGIN 的子特征中，已跳过")
            return

        # 建立对象引用
        self.child_features.append(node)
        # 给子特征记录父指针（若该属性存在或允许动态设置）
        try:
            setattr(node, "parent_node", self)
        except Exception:
            pass

    def default_interaction(self, prompt: str) -> Any:
        """默认交互"""
        self.output_callback(f"💬 {prompt}")
        return input("请输入反馈(yes/no): ").strip().lower()

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

    def process_next_node(self, node: 'Node', chat_log: Any) -> Any:
        """Origin 节点处理逻辑"""
        self.visited = True
        self.output_callback(f"🚀 从 Origin 节点开始: {self.description}")

        target_feature = self._select_next_feature()
        if target_feature:
            self.output_callback(f"🔍 进入子特征: {target_feature.node_id}")
            return {"next_node": target_feature}

        # 所有子特征访问完毕 → 直接进入 Failure（可在引擎中映射为唯一的 FAILURE 节点）
        self.output_callback("❌ 所有子特征已访问，流程结束 → 跳转到 Failure")
        return {"next_node": "FAILURE"}
