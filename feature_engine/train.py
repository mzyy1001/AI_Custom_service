# feature_tree/training/train.py
from __future__ import annotations
from typing import List, Optional, Callable
import uuid
from pathlib import Path
import math

# 你的引擎 & 各节点
from feature_engine.engine import Engine  # 你的极简版 Engine（使用 node.add_node）
from feature_engine.feature_tree.node import Node, NodeType
from feature_engine.feature_tree.Nodes.origin import OriginNode
from feature_engine.feature_tree.Nodes.feature import FeatureNode
from feature_engine.feature_tree.Nodes.problem import ProblemNode
from feature_engine.feature_tree.Nodes.solution import SolutionNode
from feature_engine.feature_tree.Nodes.success import SuccessNode
from feature_engine.feature_tree.Nodes.failure import FailureNode

# LLM 训练辅助
from feature_engine.llm_client.llm_train import (
    classify_line,
    canonicalize_problem,
    choose_best,
    solution_matches_problem,
    infer_problem_from_solution,
    pick_problem_index_for_solution,
    is_feature_same
)

# ---------------- 工具 ----------------

_ALWAYS_NO: Callable[[str], str] = lambda prompt: "no"

def _new_id(prefix: str, existing_ids: set) -> str:
    while True:
        nid = f"{prefix}_{uuid.uuid4().hex[:8]}"
        if nid not in existing_ids:
            return nid

def _all_nodes_of_type(engine: Engine, t: NodeType):
    return [n for n in engine.registry.values() if getattr(n, "node_type", None) == t]

def _bind_training_callbacks(node: Node) -> None:
    """训练模式：把节点的交互回调固定为返回 'no'。"""
    try:
        # 基类提供 set_interaction_callback
        if hasattr(node, "set_interaction_callback"):
            node.set_interaction_callback(_ALWAYS_NO)
        else:
            # 兼容老属性
            setattr(node, "interaction_callback", _ALWAYS_NO)
    except Exception:
        pass

def _ensure_registered(engine: Engine, node: Node) -> None:
    if node.node_id not in engine.registry:
        engine.add_node(node)
    _bind_training_callbacks(node)

def _new_empty_engine() -> Engine:
    """当没有现有树时，新建一棵空树（含唯一 SUCCESS/FAILURE），并开启训练回调。"""
    root = OriginNode("ORIGIN", "训练树入口", child_features=[])
    success = SuccessNode("SUCCESS", "完成")
    failure = FailureNode("FAILURE", "失败")
    eng = Engine(root, success, failure)

    # 注册并绑定训练回调
    for n in (root, success, failure):
        _ensure_registered(eng, n)

    # 训练模式：引擎层面也保存一个全局交互回调（step 时注入）
    eng.interaction_callback = _ALWAYS_NO
    return eng

# ---------------- 主训练逻辑 ----------------

def train_on_segment(engine: Engine, segment_lines: List[str]) -> None:
    """
    对一段“问答→推导到解决方案”的文本进行训练。
    segment_lines: 已按句子拆好的列表（建议每句一行）
    """
    current = engine.root
    last_problem_under_feature: Optional[ProblemNode] = None

    # 训练模式：现有所有节点强制绑定 'no' 回调
    for n in list(engine.registry.values()):
        _bind_training_callbacks(n)

    is_first_nonempty = True
    for raw in segment_lines:
        line = str(raw).strip()
        if not line:
            continue

        if is_first_nonempty:
            tag = "feature"

            is_first_nonempty = False
        else:
            tag = classify_line(line)

        print(f"[TRAIN] line='{line}' → tag={tag} | current={current.node_id}:{current.description}")

        # ---------- FEATURE ----------
        if tag == "feature":
            print(f"[训练] 遇到 Feature: {line}")

            child_feats = []
            if current.node_type in (NodeType.ORIGIN, NodeType.FEATURE, NodeType.PROBLEM):
                child_feats = getattr(current, "child_features", [])

            if child_feats:
                idx = choose_best(line, [f"{c.node_id}:{c.description}" for c in child_feats])
                if idx is not None:
                    chosen = child_feats[idx]
                    print(f"  → 使用已有子 Feature: {chosen.node_id} {chosen.description}")
                    current = chosen
                    _bind_training_callbacks(current)
                    last_problem_under_feature = None
                    continue

            all_feats = _all_nodes_of_type(engine, NodeType.FEATURE)
            if all_feats:
                idx = choose_best(line, [f"{f.node_id}:{f.description}" for f in all_feats])
                if idx is not None:
                    chosen = all_feats[idx]
                    if is_feature_same(chosen.description, line):
                        try:
                            current.add_node(chosen)
                            print(f"  → 连接到全局已有 Feature: {chosen.node_id} {chosen.description} current: {current.node_id}:{current.description}")
                        except Exception:
                            print(f"  → 连接失败，保持原有结构")
                        current = chosen
                        _bind_training_callbacks(current)
                        last_problem_under_feature = None
                        continue

            fid = _new_id("F", set(engine.registry.keys()))
            feat = FeatureNode(
                node_id=fid,
                description=line,
                parent_node=current if current.node_type in (NodeType.ORIGIN, NodeType.FEATURE, NodeType.PROBLEM) else engine.root,
            )
            _ensure_registered(engine, feat)
            try:
                current.add_node(feat)
                print(f"  → 新建 Feature: {feat.node_id} {feat.description}")
            except Exception:
                engine.root.add_node(feat)
                print(f"  → 新建 Feature 并兜底挂到 Root: {feat.node_id} {feat.description}")
            current = feat   # ★ 保证转移到新建的末端节点
            last_problem_under_feature = None
            continue


        # ---------- PROBLEM ----------
        if tag == "problem":
            print(f"[训练] 遇到 Problem: {line}")

            if current.node_type == NodeType.PROBLEM:
                current = getattr(current, "parent_feature", current)

            if current.node_type != NodeType.FEATURE:
                feats = getattr(current, "child_features", []) if hasattr(current, "child_features") else []
                if feats:
                    current = feats[0]
                else:
                    fid = _new_id("F", set(engine.registry.keys()))
                    feat = FeatureNode(fid, "训练生成的特征（聚合）", parent_node=current)
                    _ensure_registered(engine, feat)
                    try:
                        current.add_node(feat)
                        print(f"  → 新建 Feature(聚合): {feat.node_id}")
                    except Exception:
                        engine.root.add_node(feat)
                        print(f"  → 新建 Feature(聚合) 挂到 Root: {feat.node_id}")
                    current = feat

            q_desc = canonicalize_problem(line)

            cur_probs = [p for (p, _m) in getattr(current, "child_problems", [])]
            if cur_probs:
                idx = choose_best(q_desc, [f"{p.node_id}:{p.description}" for p in cur_probs])
                if idx is not None:
                    chosen = cur_probs[idx]
                    print(f"  → 使用已有子 Problem: {chosen.node_id} {chosen.description}")
                    current = chosen
                    _bind_training_callbacks(current)
                    last_problem_under_feature = current
                    continue

            all_probs = _all_nodes_of_type(engine, NodeType.PROBLEM)
            if all_probs:
                idx = choose_best(q_desc, [f"{p.node_id}:{p.description}" for p in all_probs])
                if idx is not None:
                    chosen = all_probs[idx]
                    if is_feature_same(chosen.description, q_desc):
                        try:
                            current.add_node(chosen, link_mode="soft")
                            print(f"  → 连接到全局已有 Problem: {chosen.node_id} {chosen.description}")
                        except Exception:
                            pass
                        current = chosen
                        _bind_training_callbacks(current)
                        last_problem_under_feature = current
                        continue

            pid = _new_id("P", set(engine.registry.keys()))
            prob = ProblemNode(pid, q_desc, parent_feature=current, mode="soft")
            _ensure_registered(engine, prob)
            current.add_node(prob, link_mode="soft")
            print(f"  → 新建 Problem: {prob.node_id} {prob.description}")
            current = prob   # ★ 转移到新建 Problem
            last_problem_under_feature = prob
            continue


        # ---------- SOLUTION ----------
        if tag == "solution":
            print(f"[训练] 遇到 Solution: {line}")
            target_problem: Optional[ProblemNode] = None

            if current.node_type == NodeType.PROBLEM:
                match = solution_matches_problem(line, current.description)
                if match is True or match is None:
                    target_problem = current
                    print(f"  → 直接挂到当前 Problem: {current.node_id}")

            if target_problem is None and last_problem_under_feature is not None:
                match = solution_matches_problem(line, last_problem_under_feature.description)
                if match:
                    target_problem = last_problem_under_feature
                    print(f"  → 使用最近 Problem: {last_problem_under_feature.node_id}")

            if target_problem is None:
                feature_ctx = current
                if feature_ctx.node_type == NodeType.PROBLEM:
                    feature_ctx = getattr(feature_ctx, "parent_feature", feature_ctx)
                if feature_ctx.node_type != NodeType.FEATURE:
                    feats = getattr(engine.root, "child_features", [])
                    feature_ctx = feats[0] if feats else None
                    if feature_ctx is None:
                        fid = _new_id("F", set(engine.registry.keys()))
                        feature_ctx = FeatureNode(fid, "训练生成的特征（挂方案）", parent_node=engine.root)
                        _ensure_registered(engine, feature_ctx)
                        engine.root.add_node(feature_ctx)
                        print(f"  → 新建挂方案 Feature: {feature_ctx.node_id}")

                probs = [p for (p, _m) in getattr(feature_ctx, "child_problems", [])]
                if probs:
                    idx = pick_problem_index_for_solution(
                        line,
                        [f"{p.node_id}:{p.description}" for p in probs]
                    )
                    if idx is not None:
                        target_problem = probs[idx]
                        print(f"  → 使用已有 Problem: {target_problem.node_id}")

                if target_problem is None:
                    q_desc = infer_problem_from_solution(line)
                    pid = _new_id("P", set(engine.registry.keys()))
                    target_problem = ProblemNode(pid, q_desc, parent_feature=feature_ctx, mode="soft")
                    _ensure_registered(engine, target_problem)
                    feature_ctx.add_node(target_problem, link_mode="soft")
                    print(f"  → 新建 Problem(挂方案): {target_problem.node_id} {q_desc}")


            sols = getattr(target_problem, "solutions", [])
            if sols:
                idx = choose_best(line, [f"{s.node_id}:{s.description}" for s in sols])
                if idx is not None:
                    chosen_sol = sols[idx]
                    # 确保 success_node 绑定到引擎唯一 success（以防旧数据缺失）
                    if getattr(chosen_sol, "success_node", None) is None:
                        chosen_sol.success_node = engine.success
                    print(f"  → 复用已有 Solution: {chosen_sol.node_id} {chosen_sol.description}")
                    current = chosen_sol   # 转移到已存在的 solution 节点
                    continue
                
            sid = _new_id("S", set(engine.registry.keys()))
            sol = SolutionNode(sid, line, success_node=engine.success, parent_problem=target_problem)
            _ensure_registered(engine, sol)
            target_problem.add_node(sol)
            print(f"  → 新建 Solution: {sol.node_id} {sol.description}")
            current = sol   # ★ 转移到 Solution 节点
            continue

        # ---------- OTHER ----------
        # 非关键句子忽略
        continue


def train_from_file(tree_json_path: str, segments_path: str, save_to: Optional[str] = None) -> None:
    """
    - 若 tree_json_path 不存在：新建一棵空树（ORIGIN + SUCCESS + FAILURE）
    - 从 segments 文件读取文本，按空行或 '###' 分段，逐段训练
    - 训练模式：所有节点的 interaction_callback 固定返回 'no'
    """
    p = Path(tree_json_path)
    
    if p.exists():
        print(f"[TRAIN] 读取现有树: {p}")
        engine = Engine.load_nodes(str(p))
    else:
        engine = _new_empty_engine()

    # 训练模式：对已存在节点先统一绑定 'no'
    for n in list(engine.registry.values()):
        _bind_training_callbacks(n)

    text = Path(segments_path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in (text.split("###") if "###" in text else text.split("\n\n")) if b.strip()]
    print(f"[TRAIN] 读取分段数: {len(blocks)} from {segments_path}")
    n_blocks = len(blocks)
    checkpoint = max(1, math.floor(n_blocks * 0.05))  # 每 5% 存一次，至少每 1 个存一次
    for i, block in enumerate(blocks, 1):
        print(f"[TRAIN] 处理分段 {i}/{n_blocks}: {block[:20]}..., return {engine.current.description} → {engine.root.description}")
        engine.current = engine.root
        lines = [ln for ln in block.splitlines() if ln.strip()]
        train_on_segment(engine, lines)

        # 每 5% 存储一次
        if i % checkpoint == 0 or i == n_blocks:
            print(f"[TRAIN] 进度 {i}/{n_blocks} ({i/n_blocks:.1%}) → 临时保存")
            engine.save_nodes(str(save_to or p))

    print(f"[TRAIN] 完成全部，结果保存到 {save_to or tree_json_path}")


# 可选：命令行入口
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="现有节点集 JSON；不存在则新建")
    ap.add_argument("--segments", required=True, help="训练文本文件路径（空行或 ### 分段）")
    ap.add_argument("--out", default=None, help="输出文件（默认覆盖 --tree）")
    args = ap.parse_args()
    train_from_file(args.tree, args.segments, args.out)
