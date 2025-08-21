import json, os
from datetime import datetime

DUP_FILE = "logs/dup_nodes.jsonl"  # 路径自己改

def _log_dup(parent, child, reason: str):
    rec = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "reason": reason,
        "parent_id": getattr(parent, "node_id", None),
        "parent_type": str(getattr(parent, "node_type", None)),
        "parent_desc": getattr(parent, "description", None),
        "child_id": getattr(child, "node_id", None),
        "child_type": str(getattr(child, "node_type", None)),
        "child_desc": getattr(child, "description", None),
    }
    with open(DUP_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")