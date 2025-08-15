from enum import Enum
from typing import Callable, Dict, Any, Optional
from typing import Callable, Dict, Any, Optional

class NodeType(Enum):
    FEATURE = "Feature"
    SOLUTION = "Solution"
    SUCCESS = "Success"
    FAILURE = "Failure"
    PROBLEM = "Problem"
    ORIGIN = "Origin"


class Node:
    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        description: str,
        handler: Optional[Callable[['Node', Any], Any]] = None,
        output_callback: Optional[Callable[[str], None]] = None
    ):
        """
        :param node_id: 节点唯一ID
        :param node_type: 节点类别 (Feature/Solution/Success/Failure/Problem/Hub)
        :param description: 节点描述
        :param handler: 节点的执行接口函数 (接收当前节点和输入数据)
        :param output_callback: 输出回调函数（默认用 print），用于统一处理节点输出
        """
        self.node_id = node_id
        self.node_type = node_type
        self.description = description
        self.handler = handler or self.default_handler
        self.output_callback = output_callback or print
        self.visited = False

    def process_next_node(self) -> Any:
        """默认处理逻辑"""
        self.output_callback(f"[{self.node_type.value}] {self.description}")
        return None

    def __repr__(self):
        return f"<Node id={self.node_id} type={self.node_type.value}>"




# 示例：创建不同类型节点
if __name__ == "__main__":
    # Feature 节点
    feature_node = Node(
        node_id="F1",
        node_type=NodeType.FEATURE,
        description="检测电源灯是否亮"
    )

    # Solution 节点
    solution_node = Node(
        node_id="S1",
        node_type=NodeType.SOLUTION,
        description="更换电源适配器"
    )

    # Success 节点
    success_node = Node(
        node_id="END_OK",
        node_type=NodeType.SUCCESS,
        description="问题已解决"
    )

    print(feature_node)
    print(solution_node)
    print(success_node)
