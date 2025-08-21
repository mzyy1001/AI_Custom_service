from typing import Callable, Optional, Any, List

from feature_engine.feature_tree.Nodes.log import _log_dup
from ..node import Node, NodeType

from feature_engine.llm_client.llm_produce import pick_child_feature_index

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
            _log_dup(self, node, reason="ORIGIN->FEATURE")
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


    def process_next_node(self, chat_log: Any) -> Any:
        """Origin 节点处理逻辑"""
        self.visited = True
        self.output_callback(f"🚀 从 Origin 节点开始: {self.description}")

        target_feature = self._select_next_feature(chat_log)
        if target_feature:
            self.output_callback(f"🔍 进入子特征: {target_feature.node_id}")
            return {"next_node": target_feature}

        # 所有子特征访问完毕 → 直接进入 Failure（可在引擎中映射为唯一的 FAILURE 节点）
        self.output_callback("❌ 所有子特征已访问，流程结束 → 跳转到 Failure")
        return {"next_node": "FAILURE"}
