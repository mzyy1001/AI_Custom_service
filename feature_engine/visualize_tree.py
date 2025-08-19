#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import html
import webbrowser

def _wrap(text: str, width: int = 16, max_lines: int = 4) -> str:
    """简单按字符宽度换行，超过 max_lines 的部分用 … 截断。"""
    if not text:
        return ""
    lines = [text[i:i+width] for i in range(0, len(text), width)]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines[-1] and lines[-1][-1] != "…":
            lines[-1] = lines[-1][:-1] + "…"
    return "\n".join(lines)


# ---------- 解析 JSON，构建 nodes / edges ----------
def build_nodes_edges(data: dict):
    nodes = []
    edges = []

    nodes_meta = data.get("nodes", {})

    # 样式
    shape_map = {
        "Origin": "star",
        "Feature": "box",
        "Problem": "ellipse",
        "Solution": "diamond",
        "Success": "hexagon",
        "Failure": "hexagon",
    }
    color_map = {
        "Origin": "#7b68ee",
        "Feature": "#1f77b4",
        "Problem": "#ff7f0e",
        "Solution": "#2ca02c",
        "Success": "#17becf",
        "Failure": "#d62728",
    }

    # 先把节点都放进去
    for nid, meta in nodes_meta.items():
        ntype = meta.get("type", "Unknown")
        desc = meta.get("description", "") or ""
        title_html = (
            f"<div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial;'>"
            f"<div><b>ID</b>: {html.escape(nid)}</div>"
            f"<div><b>Type</b>: {html.escape(ntype)}</div>"
            f"<div style='margin-top:6px;'><b>Description</b>:<br>{html.escape(desc)}</div>"
            f"</div>"
        )
        nodes.append({
            "id": nid,
            # ✅ 显示描述（加换行，避免太长）
            "label": _wrap(desc, width=16, max_lines=4),
            # 如果你想顺带显示类型：可以用 _wrap(f"{desc}\n({ntype})", 16, 4)
            "title": title_html,           # 悬浮时仍可看到 ID/Type/完整描述
            "shape": shape_map.get(ntype, "dot"),
            "color": color_map.get(ntype, "#999999"),
            "group": ntype,
        })

    # 再按结构加边
    for nid, meta in nodes_meta.items():
        ntype = meta.get("type")
        # ORIGIN -> FEATURE
        if ntype == "Origin":
            for fid in meta.get("child_features", []):
                if fid != nid:
                    edges.append({"from": nid, "to": fid, "etype": "origin->feature"})
        # FEATURE -> PROBLEM / FEATURE
        elif ntype == "Feature":
            for prob in meta.get("child_problems", []):
                if isinstance(prob, list) and prob:
                    pid = prob[0]
                    label = prob[1] if len(prob) > 1 else ""
                    if pid != nid:
                        edges.append({"from": nid, "to": pid, "label": label, "etype": "feature->problem"})
            for fid in meta.get("child_features", []):
                if fid != nid:
                    edges.append({"from": nid, "to": fid, "etype": "feature->feature"})
        # PROBLEM -> SOLUTION / FEATURE
        elif ntype == "Problem":
            for sid in meta.get("solutions", []):
                if sid != nid:
                    edges.append({"from": nid, "to": sid, "etype": "problem->solution"})
            for fid in meta.get("child_features", []):
                if fid != nid:
                    edges.append({"from": nid, "to": fid, "etype": "problem->feature"})
        # SOLUTION -> SUCCESS
        elif ntype == "Solution":
            succ = meta.get("success_node")
            if succ and succ != nid:
                edges.append({"from": nid, "to": succ, "etype": "solution->success"})
        # SUCCESS / FAILURE 无出边
    return nodes, edges


# ---------- 生成可交互 HTML（vis-network） ----------
HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Feature Tree</title>
<link rel="preconnect" href="https://unpkg.com">
<link rel="stylesheet" href="https://unpkg.com/vis-network@9.1.6/styles/vis-network.min.css">
<script src="https://unpkg.com/vis-network@9.1.6/dist/vis-network.min.js"></script>
<style>
  body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial; margin: 0; }
  #toolbar { padding: 10px 14px; border-bottom: 1px solid #e5e5e5; display: flex; gap: 12px; align-items:center; flex-wrap: wrap; }
  #network { height: 82vh; border-top: 0; }
  .pill { display:inline-flex; gap:6px; align-items:center; padding:6px 10px; border:1px solid #ddd; border-radius:999px; }
  .sep { height: 20px; width:1px; background:#e5e5e5; margin:0 4px; }
  #info { font-size:13px; color:#666; }
  #details { padding: 8px 14px; font-size: 13px; border-top: 1px solid #eee; height: 16vh; overflow:auto; }
  code { background:#f6f8fa; padding:2px 4px; border-radius:4px; }
</style>
</head>
<body>

<div id="toolbar">
  <div class="pill">
    <input id="query" placeholder="搜索节点 ID 或描述关键字…" style="width:260px; padding:6px 8px; border:1px solid #ccc; border-radius:8px;">
    <button id="btnSearch">搜索并定位</button>
    <button id="btnClear">清除高亮</button>
  </div>

  <div class="sep"></div>

  <div class="pill">
    <label><input type="checkbox" class="typeToggle" value="Origin" checked> Origin</label>
    <label><input type="checkbox" class="typeToggle" value="Feature" checked> Feature</label>
    <label><input type="checkbox" class="typeToggle" value="Problem" checked> Problem</label>
    <label><input type="checkbox" class="typeToggle" value="Solution" checked> Solution</label>
    <label><input type="checkbox" class="typeToggle" value="Success" checked> Success</label>
    <label><input type="checkbox" class="typeToggle" value="Failure" checked> Failure</label>
  </div>

  <div class="sep"></div>

  <div class="pill">
    <label><input type="checkbox" id="hier" {HIER_CHECKED}> 层级布局</label>
    <label><input type="checkbox" id="physics" {PHYSICS_CHECKED}> 物理引擎</label>
    <button id="btnFit">Fit</button>
  </div>

  <div id="info"></div>
</div>

<div id="network"></div>
<div id="details">👈 点击节点查看详情</div>

<script>
const RAW_NODES = {NODES_JSON};
const RAW_EDGES = {EDGES_JSON};

const container = document.getElementById('network');
const details = document.getElementById('details');
const info = document.getElementById('info');

const allNodes = new vis.DataSet(RAW_NODES);
const allEdges = new vis.DataSet(RAW_EDGES);

// 初始可见集（可被过滤）
const nodes = new vis.DataSet(allNodes.get());
const edges = new vis.DataSet(allEdges.get());

function buildOptions() {{
  return {{
    interaction: {{ hover: true, tooltipDelay: 120 }},
    physics: {{
      enabled: {PHYSICS_BOOL},
      stabilization: {{ iterations: 200 }},
      // 让层级布局更疏
      hierarchicalRepulsion: {{
        nodeDistance: 150,
        centralGravity: 0.0,
        springLength: 200,
        springConstant: 0.005,
        damping: 0.09,
        avoidOverlap: 1.0
      }}
    }},
    nodes: {{
      font: {{ size: 11, multi: 'html' }}  // ← 字体更小（原 12）
    }},
    edges: {{
      arrows: {{ to: {{enabled: true, scaleFactor: 0.7}} }}, // 箭头略小
      smooth: {{ enabled: true, type: 'dynamic' }},
      font: {{ size: 9, align: 'horizontal' }}               // ← 边文字更小（原 10）
    }},
    layout: {{
      hierarchical: {{
        enabled: {HIER_BOOL},
        direction: 'UD',
        sortMethod: 'directed',
        nodeSpacing: 150,        // ← 节点横向间距（原 180）
        levelSeparation: 150     // ← 层级纵向间距（原 180）
      }}
    }}
  }};
}}


let network = new vis.Network(container, {{nodes, edges}}, buildOptions());
network.once('stabilized', () => network.fit());

// 统计信息
function updateInfo() {{
  info.textContent = `节点: ${nodes.length}  边: ${edges.length}`;
}}
updateInfo();

// 点击节点 → 右侧详情
network.on('click', (params) => {{
  if (params.nodes.length) {{
    const nid = params.nodes[0];
    const n = allNodes.get(nid);
    details.innerHTML = n.title;
  }}
}});

// 搜索
const queryEl = document.getElementById('query');
document.getElementById('btnSearch').onclick = () => {{
  const q = (queryEl.value || '').trim().toLowerCase();
  if (!q) return;
  // 匹配 id 或描述
  const hits = allNodes.get().filter(n => n.id.toLowerCase().includes(q) || stripHtml(n.title).toLowerCase().includes(q));
  if (!hits.length) return;
  const hitIds = hits.map(h => h.id);
  // 只高亮匹配
  nodes.update(nodes.get().map(n => ({{ id: n.id, color: allNodes.get(n.id).color }})));
  hitIds.forEach(id => nodes.update({{ id, color: '#ffc107' }}));
  network.selectNodes(hitIds);
  network.focus(hitIds[0], {{ scale: 1.1, animation: true }});
}};
document.getElementById('btnClear').onclick = () => {{
  nodes.update(nodes.get().map(n => ({{ id: n.id, color: allNodes.get(n.id).color }})));
  network.unselectAll();
}};
function stripHtml(s) {{
  const div = document.createElement('div'); div.innerHTML = s; return div.textContent || '';
}}

// 类型过滤
for (const cb of document.querySelectorAll('.typeToggle')) {{
  cb.addEventListener('change', () => {{
    const enabled = new Set(Array.from(document.querySelectorAll('.typeToggle:checked')).map(x => x.value));
    // 更新 nodes 可见性（用 dataset 过滤）
    const keepIds = allNodes.get().filter(n => enabled.has(n.group)).map(n => n.id);
    nodes.clear();
    nodes.add(allNodes.get(keepIds));
    // 边：两端都在 keepIds 才保留
    edges.clear();
    edges.add(allEdges.get().filter(e => keepIds.includes(e.from) && keepIds.includes(e.to)));
    updateInfo();
  }});
}}

// 布局/物理切换
document.getElementById('hier').onchange = (e) => {{
  const hier = e.target.checked;
  const opts = buildOptions();
  opts.layout.hierarchical.enabled = hier;
  network.setOptions(opts);
  network.stabilize();
}};
document.getElementById('physics').onchange = (e) => {{
  const phys = e.target.checked;
  const opts = buildOptions();
  opts.physics.enabled = phys;
  network.setOptions(opts);
  network.stabilize();
}};

// fit
document.getElementById('btnFit').onclick = () => network.fit();

</script>
</body>
</html>
"""

def write_html(nodes, edges, out_path: Path, hierarchical: bool, physics: bool):
    html_str = (HTML_TEMPLATE
        .replace("{NODES_JSON}", json.dumps(nodes, ensure_ascii=False))
        .replace("{EDGES_JSON}", json.dumps(edges, ensure_ascii=False))
        .replace("{HIER_BOOL}", "true" if hierarchical else "false")
        .replace("{PHYSICS_BOOL}", "true" if physics else "false")
        .replace("{HIER_CHECKED}", "checked" if hierarchical else "")
        .replace("{PHYSICS_CHECKED}", "checked" if physics else ""))
    # 关键一步：把模板中用于转义的双大括号恢复成单大括号
    html_str = html_str.replace("{{", "{").replace("}}", "}")
    out_path.write_text(html_str, encoding="utf-8")
    print(out_path.name)


def main():
    ap = argparse.ArgumentParser(description="可交互可视化节点 JSON（vis-network，无第三方依赖）")
    ap.add_argument("json", help="输入 JSON（nodes.json / nodes_trained.json）")
    ap.add_argument("-o", "--out", default="tree.html", help="输出 HTML 文件")
    ap.add_argument("--no-hier", action="store_true", help="关闭层级布局（使用力导向）")
    ap.add_argument("--no-physics", action="store_true", help="关闭物理引擎")
    ap.add_argument("--no-open", action="store_true", help="生成后不自动打开浏览器")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    nodes, edges = build_nodes_edges(data)
    out_path = Path(args.out)
    write_html(nodes, edges, out_path, hierarchical=not args.no_hier, physics=not args.no_physics)

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())

if __name__ == "__main__":
    main()
