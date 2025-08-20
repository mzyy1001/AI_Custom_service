# engine.py
from __future__ import annotations
from typing import Any, Dict, Optional, List, Callable
import json

# === 按你的项目结构调整这些导入路径 ===
from feature_engine.feature_tree.node import Node, NodeType
from feature_engine.feature_tree.Nodes.feature import FeatureNode
from feature_engine.feature_tree.Nodes.problem import ProblemNode
from feature_engine.feature_tree.Nodes.solution import SolutionNode
from feature_engine.feature_tree.Nodes.success import SuccessNode
from feature_engine.feature_tree.Nodes.failure import FailureNode
from feature_engine.feature_tree.Nodes.origin import OriginNode


class Engine:
    """
    极简 FSM 引擎（对象引用版）
    - 不再负责连边校验；连边统一交给各 Node.add_node(...) 完成
    - 仅保存/加载“节点集”（结构与连边），方便可视化
    - 保留 step() 便于联调
    """

    def __init__(
        self,
        root: OriginNode,
        success: SuccessNode,
        failure: FailureNode,
        *,
        output_callback: Optional[Callable[[str], None]] = None,
        interaction_callback: Optional[Callable[[str], Any]] = None,
    ):
        self.root: OriginNode = root
        self.current: Node = root
        self.success: SuccessNode = success
        self.failure: FailureNode = failure
        self.output_callback = output_callback or print
        self.interaction_callback = interaction_callback

        self.registry: Dict[str, Node] = {}
        self._index_graph(root)
        self.registry[self.success.node_id] = self.success
        self.registry[self.failure.node_id] = self.failure

    # ------------------------ 轻量推进（可选） ------------------------

    def step(self, new_input: Any = None) -> Dict[str, Any]:
        node = self.current
        if node is None:
            return {"done": True, "reason": "no_current_node"}

        # 注入回调（如节点端未设置）
        if getattr(node, "output_callback", None) is None:
            setattr(node, "output_callback", self.output_callback)
        if hasattr(node, "interaction_callback") and getattr(node, "interaction_callback", None) is None:
            setattr(node, "interaction_callback", self.interaction_callback)

        if hasattr(node, "process_next_node"):
            # print(f"🔄 处理节点: {node.node_id} ({new_input})")
            res = node.process_next_node(new_input)
        else:
            res = node.execute(new_input)

        if node.node_type in (NodeType.SUCCESS, NodeType.FAILURE):
            self.current = None
            return {"done": True, "terminal": node.node_type.value, "node": node}

        next_node = None
        if isinstance(res, dict):
            nxt = res.get("next_node")
            if isinstance(nxt, Node):
                next_node = nxt
            elif isinstance(nxt, str):
                next_node = self.registry.get(nxt)

        if next_node is None:
            next_node = self.failure  # 无去处统一收敛到失败，保证可预测

        self.current = next_node
        return {"done": next_node.node_type in (NodeType.SUCCESS, NodeType.FAILURE), "node": next_node}

    # ------------------------ 构图：注册 & 薄封装连边 ------------------------

    def add_node(self, node: Node) -> None:
        """仅注册节点到引擎索引（不连边）。"""
        assert isinstance(node, Node), "add_node: 只能注册 Node 实例"
        node.set_interaction_callback(self.interaction_callback)
        nid = node.node_id
        assert nid not in self.registry, f"add_node: 已存在节点 id={nid}"
        self.registry[nid] = node

    def attach(self, parent: Node, child: Node, **kwargs) -> None:
        """
        薄封装：把连边委托给节点自身的 add_node。
        用法示例：
            eng.attach(featureA, problemX, link_mode="hard")
            eng.attach(problemX, solutionY)
        """
        if not isinstance(parent, Node) or not isinstance(child, Node):
            raise TypeError("attach: 需要 (parent: Node, child: Node)")
        # 直接调用节点自带的连边方法（各 Node 实现里已做规则限制）
        parent.add_node(child, **kwargs)
        # 更新索引
        self.registry[parent.node_id] = parent
        self.registry[child.node_id] = child

    # ------------------------ 索引与序列化（仅节点集） ------------------------

    def _index_graph(self, start: Node) -> None:
        """从起点遍历建立 registry（仅收集节点）。"""
        seen = set()
        stack = [start, self.success, self.failure]
        while stack:
            n = stack.pop()
            if not isinstance(n, Node) or n.node_id in seen:
                continue
            seen.add(n.node_id)
            self.registry[n.node_id] = n

            neigh: List[Node] = []
            if n.node_type == NodeType.ORIGIN:
                neigh += getattr(n, "child_features", [])
            elif n.node_type == NodeType.FEATURE:
                cps = []
                for item in getattr(n, "child_problems", []):
                    if isinstance(item, tuple) and len(item) == 2:
                        cps.append(item[0])
                    else:
                        cps.append(item)
                neigh += cps
                neigh += getattr(n, "child_features", [])
            elif n.node_type == NodeType.PROBLEM:
                neigh += getattr(n, "solutions", [])
                neigh += getattr(n, "child_features", [])
            elif n.node_type == NodeType.SOLUTION:
                if getattr(n, "success_node", None):
                    neigh.append(n.success_node)
            stack.extend([x for x in neigh if isinstance(x, Node)])

    def save_nodes(self, path: str) -> None:
        """仅保存“节点集”（结构/连边），用于可视化。"""
        data: Dict[str, Any] = {
            "root_id": self.root.node_id,
            "success_id": self.success.node_id,
            "failure_id": self.failure.node_id,
            "nodes": {},
        }
        for nid, n in self.registry.items():
            meta: Dict[str, Any] = {
                "type": n.node_type.value,
                "description": getattr(n, "description", ""),
            }
            if n.node_type == NodeType.ORIGIN:
                meta["child_features"] = [c.node_id for c in getattr(n, "child_features", [])]
            elif n.node_type == NodeType.FEATURE:
                cps: List[List[str]] = []
                for item in getattr(n, "child_problems", []):
                    if isinstance(item, tuple) and len(item) == 2:
                        cps.append([item[0].node_id, str(item[1]).lower()])
                meta.update({
                    "parent_node": getattr(getattr(n, "parent_node", None), "node_id", None),
                    "child_problems": cps,
                    "child_features": [c.node_id for c in getattr(n, "child_features", [])],
                })
            elif n.node_type == NodeType.PROBLEM:
                meta.update({
                    "parent_feature": getattr(getattr(n, "parent_feature", None), "node_id", None),
                    "solutions": [s.node_id for s in getattr(n, "solutions", [])],
                    "child_features": [f.node_id for f in getattr(n, "child_features", [])],
                })
            elif n.node_type == NodeType.SOLUTION:
                meta.update({
                    "parent_problem": getattr(getattr(n, "parent_problem", None), "node_id", None),
                    "success_node": getattr(getattr(n, "success_node", None), "node_id", None),
                })
            data["nodes"][nid] = meta

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_nodes(
        cls,
        path: str,
        *,
        output_callback: Optional[Callable[[str], None]] = None,
        interaction_callback: Optional[Callable[[str], Any]] = None,
    ) -> "Engine":
        """从 JSON 加载“节点集”，只恢复结构与连边。（连边逻辑仍然来源于各节点类型定义）"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes_meta: Dict[str, Dict[str, Any]] = data["nodes"]
        # 第一遍：按类型构造节点（父子引用稍后再连）
        temp: Dict[str, Node] = {}
        for nid, meta in nodes_meta.items():
            ntype = NodeType(meta["type"])
            desc = meta.get("description", "")
            if ntype == NodeType.ORIGIN:
                node = OriginNode(nid, desc, child_features=[], output_callback=output_callback,
                                  interaction_callback=interaction_callback)
            elif ntype == NodeType.FEATURE:
                node = FeatureNode(nid, desc,
                                   parent_node=None, child_problems=[], child_features=[],
                                   output_callback=output_callback, interaction_callback=interaction_callback)
            elif ntype == NodeType.PROBLEM:
                node = ProblemNode(nid, desc, parent_feature=None,  # 适配新签名
                                   mode="soft", solutions=[], child_features=[],
                                   output_callback=output_callback, interaction_callback=interaction_callback)
            elif ntype == NodeType.SOLUTION:
                node = SolutionNode(nid, desc, success_node=None, parent_problem=None,
                                    output_callback=output_callback, interaction_callback=interaction_callback)
            elif ntype == NodeType.SUCCESS:
                node = SuccessNode(nid, desc, output_callback=output_callback)
            elif ntype == NodeType.FAILURE:
                node = FailureNode(nid, desc, output_callback=output_callback)
            else:
                raise ValueError(f"未知类型：{ntype}")
            temp[nid] = node

        # 创建引擎实例
        root = temp[data["root_id"]]
        success = temp[data["success_id"]]
        failure = temp[data["failure_id"]]
        eng = cls(root, success, failure, output_callback=output_callback, interaction_callback=interaction_callback)

        # 第二遍：连边/引用（使用各节点约定的数据结构）
        for nid, meta in nodes_meta.items():
            n = temp[nid]
            if n.node_type == NodeType.ORIGIN:
                n.child_features = [temp[x] for x in meta.get("child_features", [])]
                for c in n.child_features:
                    c.parent_node = n

            elif n.node_type == NodeType.FEATURE:
                pid = meta.get("parent_node")
                n.parent_node = temp.get(pid) if pid else None
                n.child_features = [temp[x] for x in meta.get("child_features", [])]
                for c in n.child_features:
                    c.parent_node = n
                cps = []
                for pid, lm in meta.get("child_problems", []):
                    p = temp[pid]
                    cps.append((p, str(lm).lower()))
                    p.parent_feature = n
                n.child_problems = cps

            elif n.node_type == NodeType.PROBLEM:
                pf = meta.get("parent_feature")
                n.parent_feature = temp.get(pf) if pf else None
                n.child_features = [temp[x] for x in meta.get("child_features", [])]
                for f in n.child_features:
                    f.parent_node = n
                n.solutions = [temp[x] for x in meta.get("solutions", [])]
                for s in n.solutions:
                    s.parent_problem = n

            elif n.node_type == NodeType.SOLUTION:
                pp = meta.get("parent_problem")
                sn = meta.get("success_node")
                n.parent_problem = temp.get(pp) if pp else None
                n.success_node = temp.get(sn) if sn else None

        # 完成注册
        eng.registry.update(temp)
        eng.current = root
        return eng
