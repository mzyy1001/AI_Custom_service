# node.py
from __future__ import annotations
from enum import Enum
from typing import Callable, Dict, Any, Optional, List


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
        output_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        :param node_id: 节点唯一ID
        :param node_type: 节点类别 (Feature/Solution/Success/Failure/Problem/Origin)
        :param description: 节点描述
        :param handler: 节点的执行接口函数 (接收当前节点和输入数据)
        :param output_callback: 输出回调函数（默认用 print），用于统一处理节点输出
        """
        self.node_id = node_id
        self.node_type = node_type
        self.description = description
        self.handler = handler or self.default_handler
        self.output_callback = output_callback or print
        self.counts = 0

        self.interaction_callback: Optional[Callable[[str], Any]] = None
        # 占位：后续把“连边”放进节点内（对象引用）
        # 仅占位，不在此文件中实现具体规则/逻辑
        self.parents: List['Node'] = []     # 入边（父节点集合）
        self.children: List['Node'] = []    # 出边（子节点集合）

        self.visited = False

    def increment_counts(self) -> None:
        """增加节点计数器"""
        self.counts += 1

    def set_interaction_callback(self, fn: Optional[Callable[[str], Any]]) -> None:
        """
        绑定/更新交互回调。
        用法：
            node.set_interaction_callback(lambda prompt: input(prompt))
        """
        if fn is not None and not callable(fn):
            raise TypeError("interaction_callback 必须是可调用对象或 None")
        self.interaction_callback = fn 
        
    # ==== 占位：外部图管理器/子类将实现 ====
    def add_node(self, *args, **kwargs) -> None:
        """
        占位：新增/注册节点（不在 Node 基类里实现具体逻辑）。
        实际创建/管理通常由 Engine/Graph 管理器负责。
        """
        pass

    # =====================================

    def process_next_node(self, chat_log: List[Dict[str, str]]) -> Any:
        """默认处理逻辑（子类通常会覆写）"""
        self.output_callback(f"[{self.node_type.value}] {self.description}")
        return None

    def default_handler(self, node: 'Node', input_data: Any) -> Any:
        self.output_callback(f"[{self.node_type.value}] {self.description}")
        return None

    def __repr__(self):
        return f"<Node id={self.node_id} type={self.node_type.value}>"


# 示例：创建不同类型节点（仅演示；实际项目中通常不会直接实例化 Node 基类）
if __name__ == "__main__":
    feature_node = Node(
        node_id="F1",
        node_type=NodeType.FEATURE,
        description="检测电源灯是否亮"
    )

    solution_node = Node(
        node_id="S1",
        node_type=NodeType.SOLUTION,
        description="更换电源适配器"
    )

    success_node = Node(
        node_id="END_OK",
        node_type=NodeType.SUCCESS,
        description="问题已解决"
    )

    print(feature_node)
    print(solution_node)
    print(success_node)
