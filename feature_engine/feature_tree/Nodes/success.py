from typing import Callable, Optional, Any
from ..node import Node, NodeType

class SuccessNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        handler: Optional[Callable[['Node', Any], Any]] = None,
        output_callback: Optional[Callable[[str], None]] = None
    ):
        """
        :param output_callback: 用于输出处理的回调函数，例如日志系统接口。
        """
        super().__init__(
            node_id=node_id,
            node_type=NodeType.SUCCESS,
            description=description,
            handler=handler
        )
        self.output_callback = output_callback or print  # 默认用 print

    def process_next_node(self, chat_log: Any) -> Any:
        """成功节点默认逻辑"""
        message = f"✅ 成功: {self.description}"
        self.output_callback(message)
        return {"status": "success", "node_id": self.node_id}
