from typing import Callable, Optional, Any
from ..node import Node, NodeType  # 按实际路径调整

class SolutionNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        success_node: 'Node',           # 直接持有成功节点对象
        parent_problem: 'Node',         # 直接持有母问题节点对象
        interaction_callback: Optional[Callable[[str], Any]] = None,
        handler: Optional[Callable[['Node', Any], Any]] = None,
        output_callback: Optional[Callable[[str], None]] = None
    ):
        """
        :param success_node: 成功节点对象
        :param parent_problem: 母节点（问题节点）对象
        :param interaction_callback: 交互回调函数 (接收description，返回用户回复)
        """
        super().__init__(
            node_id=node_id,
            node_type=NodeType.SOLUTION,
            description=description,
            handler=handler,
            output_callback=output_callback
        )
        self.success_node = success_node
        self.parent_problem = parent_problem
        self.visited = False
        if interaction_callback is not None:
            self.set_interaction_callback(interaction_callback)

    def default_interaction(self, prompt: str) -> Any:
        """
        默认交互方式：
        - 输出提示（output_callback）
        - 从命令行获取用户输入
        """
        self.output_callback(f"💬 {prompt}")
        return input("请输入反馈(yes/no): ").strip().lower()

    def process_next_node(self, chat_log: Any) -> Any:
        """解决方案处理逻辑（对象引用版本）"""
        self.visited = True
        self.output_callback(f"🛠 执行解决方案: {self.description}")
        prompt = f"尝试 '{self.description}',尝试后,问题是否已解决？"
        # 调用交互接口获取用户回复
        reply = self.interaction_callback(prompt)

        # 解析用户回复
        if reply in ("yes", "y", "true", "1", True):
            self.output_callback(f"✅ 问题解决，跳转到成功节点 {self.success_node.node_id}")
            return {"next_node": self.success_node}  # 直接返回对象
        else:
            self.output_callback(f"⚠ 未解决，回溯到问题节点 {self.parent_problem.node_id}")
            return {"next_node": self.parent_problem}  # 直接返回对象
