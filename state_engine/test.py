# robot_flow_llm_cases.py
# 依赖：
#   - from state_engine.production import Graph, Engine
#   - from build_graph_robot_flow import build_graph  # 就是你上面那份 build_graph()
#   - from openai_router import OpenAIRouter          # 你的 LLM 路由器
#   - from state import StateCategory

from production import Graph, Engine
from build_graph_robot_flow import build_graph
from openai_router import OpenAIRouter
from state import StateCategory
from history_router import HistoryAwareRouter
from io import StringIO
SCENARIOS = [
    {
        "name": "A_低电_充电成功_END",
        "dialogue": [
            "RCS里有低电告警，估计电量不够。",
            "RCS 确认电量不足",
            "我充了半个小时，现在一按电源就能起来了。"
        ],
        "expect_terminal": "END"
    },
    {
        "name": "B_低电_充电无效_插拔线成功_END",
        "dialogue": [
            "日志显示 Low Battery，应该就是电量问题。",
            "已经按要求充了20多分钟，但还是起不来。",
            "把电池连接线重新拔插了一下，现在能开机了。"
        ],
        "expect_terminal": "END"
    },
    {
        "name": "C_非电量_插拔线无效_换电池成功_END",
        "dialogue": [
            "RCS里没看到低电提示，不像是电量导致的。",
            "试过重新插拔电池线，依然黑屏。",
            "换了电池模块就好了，能正常启动。"
        ],
        "expect_terminal": "END"
    },
    {
        "name": "D_一路排查到换信道后成功_END",
        "dialogue": [
            "没有低电。",
            "插拔线也不行。",
            "换电池模块还是没反应。",
            "电源板和BMSP都试换过了，还是开不了。",
            "RCS上看AP是离线的。",
            "把两根天线都拧紧了，底座也避免接触金属，现在AP是在线了，但机器还打不开。",
            "网线和电源重插后，AP在线，但设备仍然无法开机。",
            "更换了AP盒子，还是不行。",
            "切换了AP信道之后，终于可以开机。"
        ],
        "expect_terminal": "END"
    },
    {
        "name": "E_到S10仍失败_ESCALATE",
        "dialogue": [
            "不是电量问题。",
            "重新插拔连接线还是不亮。",
            "电池模块换了也不行。",
            "电源板和BMSP都换过，问题依旧。",
            "RCS显示AP掉线（灰色）。",
            "天线拧紧后AP已经在线了，但开不了机。",
            "网线电源都复位，设备还是起不来。",
            "更换AP盒子后仍旧不开机。",
            "尝试换了AP信道，依旧不行。"
        ],
        "expect_terminal": "ESCALATE"
    },
    {
        "name": "F_S6_AP在线直接转人工_ESCALATE",
        "dialogue": [
            "电量正常，没有低电。",
            "插拔线也没用。",
            "电池模块更换也无效。",
            "电源板/BMSP板更换也无效。",
            "AP在RCS上是在线状态。"
        ],
        "expect_terminal": "ESCALATE"
    },
    {
        "name": "G_S7_先离线_后不离线但不开机_换盒子好_END",
        "dialogue": [
            "不是电量问题。",
            "插拔线没效果。",
            "电池模块不行。",
            "电源板和BMSP也换了还是不行。",
            "AP是掉线状态。",
            "把天线柱拧紧并挪开金属后，现在不离线了，但设备还是不开机。",
            "重插网线和电源，AP在线，机器还是起不来。",
            "把AP盒子换掉后就好了。"
        ],
        "expect_terminal": "END"
    },
    {
        "name": "H_英文混搭_最终END",
        "dialogue": [
            "No low-battery warnings in RCS.",
            "Reseated the battery cable, still won't boot.",
            "Swapped the battery module—now it boots up fine."
        ],
        "expect_terminal": "END"
    },
    {
        "name": "I_噪声+表情_低电_充电失败_插拔失败_换电池成功_END",
        "dialogue": [
            "RCS有低电⚠️，应该是电量不足吧？",
            "已经充了大概20分钟了，还是起不来 :(",
            "把电池线拔了又插，依旧没反应……",
            "换了一个电池模块，就能开机了✅"
        ],
        "expect_terminal": "END"
    },
    {
        "name": "J_否定式_不是离线(=在线)_走S7->S8->S9->END",
        "dialogue": [
            "没低电。",
            "插拔线没用。",
            "换电池还是不行。",
            "电源板和BMSP都换了，依然不行。",
            "AP不是离线（现在是在线）。",
            "网线电源都检查过，还是开不了。",
            "把AP盒子换了，解决。"
        ],
        "expect_terminal": "END"
    },
    {
        "name": "K_持续离线_多次回环后在线但仍不开机_ESCALATE",
        "dialogue": [
            "非电量问题。",
            "插拔连接线也不行。",
            "换电池也不行。",
            "电源板+BMSP更换无效。",
            "AP显示离线。",
            "天线拧紧后还是离线。",
            "又回去检查电源板BMSP也没变化（还是不行）。",
            "现在AP显示在线，但机器依旧打不开。"
        ],
        "expect_terminal": "ESCALATE"
    },
    {
        "name": "L_极简含糊_触发abstain_走ESCALATE",
        "dialogue": [
            "嗯……",  # 期望在某一步匹配失败，触发 Router 返回 abstain → Engine 升级人工
        ],
        "expect_terminal": "ESCALATE"
    },
]
SCENARIO_IDX = None

def run_all():
    ok = fail = 0
    total = len(SCENARIOS)
    if isinstance(SCENARIO_IDX, int):
        idx = SCENARIO_IDX if SCENARIO_IDX >= 0 else total + SCENARIO_IDX
        if not (0 <= idx < total):
            print(f"[WARN] SCENARIO_IDX 超界：{SCENARIO_IDX} (共有 {total} 个)。不执行。")
            return
        to_run = [idx]
    else:
        to_run = list(range(total))

    for i in to_run:
        sc = SCENARIOS[i]

        graph = build_graph()

        # 路由：历史上下文包装 → 注入完整对话历史
        base_router = OpenAIRouter()
        router = HistoryAwareRouter(base_router)

        # 日志缓冲：运行中不打印，结束后统一输出
        buf = StringIO()
        engine = Engine(graph, router,
                               threshold=0.6,
                               allow_jump_outside_candidates=False,
                               log_fn=lambda s: print(s, file=buf),
                               show_desc=False)

        cur = "S1"
        path = [cur]
        terminal = None

        # 一边推进 FSM，一边把每条用户话术注入到 router 历史
        for ut in sc["dialogue"]:
            router.push(ut)
            nxt = engine.transition(cur, ut)   # 此处传的 ut 会被 router 注入“历史+当前”
            cur = nxt.state_id
            path.append(cur)
            if nxt.category in (StateCategory.END, StateCategory.ESCALATE):
                terminal = nxt.category.name
                break

        if terminal is None:
            terminal = graph.get_state(cur).category.name

        result = "OK" if terminal == sc["expect_terminal"] else f"FAIL(expected={sc['expect_terminal']})"
        ok += (result == "OK")
        fail += (result != "OK")

        # —— 场景结束后一次性打印 —— #
        print(f"\n=== RUN {sc['name']} ===")
        print("DIALOGUE:", " | ".join(sc["dialogue"]))
        print("PATH:", " -> ".join(path))
        print("TERMINAL:", terminal, "| RESULT:", result)
        print("--- ENGINE TRACE (full) ---")
        print(buf.getvalue())

    print(f"\n=== SUMMARY ===  OK={ok}  FAIL={fail}  TOTAL={len(SCENARIOS)}")

if __name__ == "__main__":
    run_all()
