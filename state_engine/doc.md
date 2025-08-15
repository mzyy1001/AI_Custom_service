# 二层结构规范：FEATURE × HUB+SOLUTION

## 1. 分层抽象
- FEATURE 层：**有向状态机**，负责“判定与分流”。部分 FEATURE 的出边指向某个 HUB（进入解决阶段）。
- SOLVE 层：以 **HUB** 为入口，调度一组 **SOLUTION**（动作）；任一步成功即终结，失败则按策略尝试下一步。

---

## 2. 统一接口（Contract）
### 调用 HUB（进入 SOLVE 层）
- 输入：`context`（证据/环境），`budget`（时间/交互/风险），`policy`（排序/Top-k）
- 输出：`status ∈ { SOLVED, UNSOLVED, ESCALATE, UNKNOWN }`
  - `evidence`：动作与观测的结构化记录
  - `next_feature`（可选）：若失败，返回建议回到哪个 FEATURE 继续判定

### SOLUTION 动作约定
- 返回：`OK | FAIL | NA`
- 必须声明：`precond`、`reversible`、`expected_cost`、`verify`（验证是否已解决）

---

## 3. FEATURE 层（有向状态机）
### 节点
- `FEATURE`：提问/判定；出边可到另一个 `FEATURE`，或 `invoke_hub: <hub_id>`
- `OBSERVE`（可选）：采集证据后再分流
- `END`：流程收敛点（仅对 FEATURE 层而言）

### 边
- `condition`（基于回答/证据） → `to: FEATURE | invoke_hub | end`

### 不变式
- 唯一入口 FEATURE
- 所有路径最终应能到达 `invoke_hub` 或 `end`

---

## 4. SOLVE 层（HUB + SOLUTION 集）
### HUB
- 角色：在一个候选 `SOLUTION[]` 集合里按策略尝试：`cheap_first / success_rate / custom`
- 失败策略：`on_all_fail ∈ { return_unsolved, escalate, redirect_feature:<id> }`

### SOLUTION
- 原子动作（例如“重启/更换/校准/清理缓存/执行命令”）
- 典型流：`precond → do → verify → (OK/FAIL/NA)`

---

## 5. YAML Schema（最小可行）

```
problem:
  name: 设备无法开机（两层范式）
  feature_graph:
    entry: F0
    nodes:
      F0:
        type: FEATURE
        prompt: "RCS 是否显示电量 < 10% ？"
        edges:
          - on: "是"
            action: { invoke_hub: hub_battery }   # 进入 SOLVE 层
          - on: "否"
            to: F1

      F1:
        type: FEATURE
        prompt: "电源指示灯是否常亮？"
        edges:
          - on: "是"
            action: { invoke_hub: hub_boot_check }
          - on: "否"
            action: { invoke_hub: hub_power_cable }

      END_OK:
        type: END

  solve_graph:
    hubs:
      hub_battery:
        strategy: cheap_first
        on_all_fail: { redirect_feature: F1 }      # 失败回到 FEATURE 继续判定
        solutions: [ S_charge20, S_replug_batt_cable, S_gauge_calib ]

      hub_power_cable:
        strategy: success_rate
        on_all_fail: { escalate: true }
        solutions: [ S_check_cable, S_replace_cable ]

      hub_boot_check:
        strategy: cheap_first
        on_all_fail: { return_unsolved: true }
        solutions: [ S_safe_boot, S_fw_reset ]

    solutions:
      S_charge20:
        precond: "battery_present == true"
        step: "静置充电 20 分钟"
        verify: "can_boot()"
        reversible: true
        expected_cost: { time_min: 20, user_steps: 1 }

      S_replug_batt_cable:
        precond: "has_access == true"
        step: "插拔电池线并固定"
        verify: "rcs_voltage() >= 10"
        reversible: true
        expected_cost: { time_min: 5 }

      S_gauge_calib:
        step: "校准电量计"
        verify: "gauge_ok()"
        reversible: false
        expected_cost: { risk: "LOW" }

      S_check_cable:
        step: "检查外部电源线接触与破损"
        verify: "led_power_on()"
        reversible: true

      S_replace_cable:
        step: "更换电源线"
        verify: "led_power_on()"
        reversible: true
        expected_cost: { money: 1 }

      S_safe_boot:
        step: "进入安全模式启动"
        verify: "system_booted()"
        reversible: true

      S_fw_reset:
        step: "执行固件复位"
        verify: "system_booted()"
        reversible: false
        expected_cost: { risk: "MID" }

---


## 6. 约束与工程要点
- **单向依赖**：FEATURE 只决定“是否进入哪个 HUB”；HUB 不改动 FEATURE 的结构，只能返回 `redirect_feature` 建议。
- **可达性检查**：构建期做状态机 reachability/无环校验（FEATURE 内允许环，但需步数上限）。
- **代价与验证**：所有 SOLUTION 必须提供 `verify`；HUB 策略可结合 `expected_cost` 与历史成功率自适应排序。
- **可替换**：FEATURE 的 `ask_or_infer` 可用规则/ML/LLM；HUB 策略可换成 MoE 门控或 RL。
