# feature_engine/run_prod.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List

# === 你的已有代码依赖 ===
from feature_engine.engine import Engine
from feature_engine.feature_tree.node import Node, NodeType
from feature_engine.feature_tree.Nodes.origin import OriginNode
from feature_engine.feature_tree.Nodes.feature import FeatureNode
from feature_engine.feature_tree.Nodes.problem import ProblemNode
from feature_engine.feature_tree.Nodes.solution import SolutionNode
from feature_engine.feature_tree.Nodes.success import SuccessNode
from feature_engine.feature_tree.Nodes.failure import FailureNode

from feature_engine.llm_client.llm_produce import (
    llm_yes_no,
    _llm_yes_no_from_user_text,
    _llm_followup_question
)

from typing import Dict, List, Optional
import re

# =========================================
# 占位：未来接 LLM 的两个函数（现在返回 None / 走人工）
# =========================================
def make_interaction_callback(chat_log: List[Dict[str, str]]):
    """
    返回一个 callback(prompt) -> 'yes'/'no' 字符串（节点普遍是这么约定的）。
    逻辑：
    1) 尝试用 LLM 判断（llm_yes_no）
    2) 否则回退到人工输入（y/n）
    """
    def _callback(prompt: str) -> str:
        print(f"🔍 需要判断：{prompt}")
        MAX_FOLLOWUPS = 3
        turn_user_texts: List[str] = []

        # 人工回退
        for turn in range(MAX_FOLLOWUPS + 1):
            # 1) 人工输入（自由文本，不限 y/n）
            raw = input("你的回答：").strip()
            chat_log.append({"role": "user", "type": "free", "prompt": prompt, "text": raw})
            turn_user_texts.append(raw)

            # 2) LLM 判定
            verdict = _llm_yes_no_from_user_text(prompt, turn_user_texts, chat_log)
            if verdict is True:
                chat_log.append({"role": "assistant", "type": "yn", "prompt": prompt, "answer": "yes"})
                print("🧠 判定：yes")
                return "yes"
            if verdict is False:
                chat_log.append({"role": "assistant", "type": "yn", "prompt": prompt, "answer": "no"})
                print("🧠 判定：no")
                return "no"

            # 3) 仍不确定 → 追问或结束
            if turn < MAX_FOLLOWUPS:
                follow_q = _llm_followup_question(prompt, turn_user_texts, chat_log)
                chat_log.append({"role": "assistant", "type": "followup", "prompt": prompt, "question": follow_q})
                print(f"🔎 追问：{follow_q}")
                # 进入下一轮输入
            else:
                print("⚠️ 信息仍不足，按‘否’处理。")
                chat_log.append({"role": "assistant", "type": "yn", "prompt": prompt, "answer": "no", "reason": "fallback_max_turns"})
                return "no"

    return _callback

def bind_callbacks_for_all_nodes(engine: Engine, interaction_cb, output_cb):
    """
    把 interaction_callback/output_callback 注入所有节点（兼容已有 set_interaction_callback / 属性赋值）
    """
    for node in engine.registry.values():
        # 输出
        try:
            node.output_callback = output_cb
        except Exception:
            pass
        # 交互
        try:
            if hasattr(node, "set_interaction_callback"):
                node.set_interaction_callback(interaction_cb)
            else:
                setattr(node, "interaction_callback", interaction_cb)
        except Exception:
            pass


def run_session(engine: Engine):
    """
    生产交互主循环：
    1) 询问“主要问题是什么？”
    2) 选择起始 Feature，并把 engine.current 设置为该节点
    3) 循环调用 engine.step(chat_log)，直到 SUCCESS / FAILURE
    """
    chat_log: List[Dict[str, str]] = []   # 你可以把后续 LLM 上下文塞这里

    # 统一输出
    def _out(s: str):
        print(s)

    # 全局交互回调（未来可接入 LLM）
    interaction_cb = make_interaction_callback(chat_log)

    # 绑定到所有节点
    bind_callbacks_for_all_nodes(engine, interaction_cb, _out)

    # 1) 主要问题
    print("🟢 欢迎使用诊断助手。")
    main_issue = input("请描述主要问题（如：机器人开不了机 / 机器人无法移动 ...）: ").strip()
    chat_log.append({"role": "user", "type": "main_issue", "text": main_issue})

    # 2) 选择起始 Feature（从 Root 的 child_features 中选择）
    engine.current = engine.root

    # 3) 迭代运行，直到终止态
    while True:
        res = engine.step(chat_log)  # Engine 会把 chat_log 透传给节点的 process_next_node
        node = engine.current
        if res.get("done"):
            terminal = res.get("terminal")
            if terminal == NodeType.SUCCESS.value:
                print("✅ 诊断完成：问题已解决。")
            elif terminal == NodeType.FAILURE.value:
                print("❌ 诊断结束：未能自动解决（请转人工）。")
            else:
                print("⚪ 结束。")
            break

        # 打印进度
        if node is not None:
            print(f"➡️ 下一节点：{node.node_id} [{node.node_type.value}] - {getattr(node, 'description', '')}")
        else:
            print("⚠️ 没有下一个节点，终止。")
            break

# =========================================
# CLI
# =========================================
def main():
    ap = argparse.ArgumentParser(description="生产环境交互入口（先不接 LLM，预留接口）")
    ap.add_argument("--tree", required=True, help="节点集 JSON（Engine.save_nodes 的产物）")
    args = ap.parse_args()

    p = Path(args.tree)
    if p.exists():
        print(f"[RUN] 加载节点集: {p}")
        engine = Engine.load_nodes(str(p))
    else:
        raise FileNotFoundError(f"树文件不存在: {p}")

    run_session(engine)

if __name__ == "__main__":
    main()
