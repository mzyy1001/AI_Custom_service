from produce import Graph
from state import State, StateCategory

def build_graph() -> Graph:
    g = Graph()

    def add(st: State):
        g.ensure_state(st)
        return st

    # 成功结束
    end = add(State("end", StateCategory.END, "问题解决"))

    # 步骤 1 → 纯判断
    s1 = add(State(
        "S1", StateCategory.DECISION,
        "【步骤1】查看 RCS 日志，确认是否电量低导致无法开机；是→步骤2，否→步骤3。\n请回复：是/否"
    ))
    s1.add_transition("S2", mark="yes")
    s1.add_transition("S3", mark="no")

    # 步骤 2 → 有动作：充电
    s2 = add(State(
        "S2", StateCategory.SOLUTION,
        "【步骤2】手动充电 20 分钟后再次尝试开机；能开机→结束；否则→步骤3。\n请回复：是/否"
    ))
    s2.add_transition("end", mark="yes")
    s2.add_transition("S3", mark="no")

    # 步骤 3 → 动作：插拔电池线
    s3 = add(State(
        "S3", StateCategory.SOLUTION,
        "【步骤3】检查电池连接线是否松动，重新插拔；是否能开机？\n能开机→结束；否则→步骤4。\n请回复：是/否"
    ))
    s3.add_transition("end", mark="yes")
    s3.add_transition("S4", mark="no")

    # 步骤 4 → 动作：更换电池模块
    s4 = add(State(
        "S4", StateCategory.SOLUTION,
        "【步骤4】检查/更换电池模块；是否能开机？\n能开机→结束；否则→步骤5。\n请回复：是/否"
    ))
    s4.add_transition("end", mark="yes")
    s4.add_transition("S5", mark="no")

    # 步骤 5 → 动作：更换电源板+BMSP
    s5 = add(State(
        "S5", StateCategory.SOLUTION,
        "【步骤5】检查/更换电源板与 BMSP 板；是否能开机？\n能开机→结束；否则→步骤6。\n请回复：是/否"
    ))
    s5.add_transition("end", mark="yes")
    s5.add_transition("S6", mark="no")

    # 步骤 6 → 纯判断：AP 是否离线
    s6 = add(State(
        "S6", StateCategory.DECISION,
        "【步骤6】RCS 上确认 AP 是否离线；离线→步骤7；在线→转人工。\n请回复：离线/在线"
    ))
    s6.add_transition(g.human_escalation_id, mark="online")  # 在线→人工
    s6.add_transition("S7", mark="offline")

    # 步骤 7 → 动作：检查 AP 天线
    s7 = add(State(
        "S7", StateCategory.SOLUTION,
        "【步骤7】检查 AP 两根天线是否拧紧且底座不接触金属；AP 仍离线吗？\n离线→步骤5；若不离线则确认能否开机：能→结束；否→步骤8。\n请回复：离线/在线/是/否"
    ))
    s7.add_transition("S5", mark="offline")
    s7.add_transition("end", mark="yes")
    s7.add_transition("S8", mark="no")
    s7.add_transition("S8", mark="online")

    # 步骤 8 → 动作：检查网线/电源
    s8 = add(State(
        "S8", StateCategory.SOLUTION,
        "【步骤8】检查 AP 网线/电源并重插；AP 仍离线吗？\n离线→步骤6；若不离线则确认能否开机：能→结束；否→步骤9。\n请回复：离线/在线/是/否"
    ))
    s8.add_transition("S6", mark="offline")
    s8.add_transition("end", mark="yes")
    s8.add_transition("S9", mark="no")
    s8.add_transition("S9", mark="online")

    # 步骤 9 → 动作：更换 AP 盒子
    s9 = add(State(
        "S9", StateCategory.SOLUTION,
        "【步骤9】更换 AP 盒子；AP 仍离线吗？\n离线→转人工；若不离线则确认能否开机：能→结束；否→步骤10。\n请回复：离线/在线/是/否"
    ))
    s9.add_transition(g.human_escalation_id, mark="offline")
    s9.add_transition("end", mark="yes")
    s9.add_transition("S10", mark="no")
    s9.add_transition("S10", mark="online")

    # 步骤 10 → 动作：更换信道
    s10 = add(State(
        "S10", StateCategory.SOLUTION,
        "【步骤10】尝试更换 AP 信道；是否能开机？\n能→结束；否则→转人工。\n请回复：是/否"
    ))
    s10.add_transition("end", mark="yes")
    s10.add_transition(g.human_escalation_id, mark="no")

    # 人工节点
    g.get_or_create_human().description = "建议联系专业的售后技术支持人员获得帮助"

    return g
