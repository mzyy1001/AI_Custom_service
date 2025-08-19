from typing import Callable, Optional, Any
from ..node import Node, NodeType  # 按你的目录结构调整路径

class FailureNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        handler: Optional[Callable[['Node', Any], Any]] = None,
        output_callback: Optional[Callable[[str], None]] = None
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.FAILURE,
            description=description,
            handler=handler,
            output_callback=output_callback
        )

    def process_next_node(self, node: 'Node', input_data: Any) -> Any:
        """失败节点默认逻辑"""
        self.output_callback(f"❌ 失败: {self.description}")
        return {"status": "failure", "node_id": self.node_id}
