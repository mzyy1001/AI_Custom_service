# 节点类型与连接/运行规则（FSM + LLM）

## 节点类型 (NodeType)

0. **ORIGIN** — 起点节点；流程唯一入口，只能指向 **FEATURE**  
1. **FEATURE** — 特征层，用于提出/确认可观测条件；带“期望状态：成立/不成立”  
2. **PROBLEM** — 问题层，由特征触发并具体化问题；带 **hard/soft** 模式  
3. **SOLUTION** — 解决方案层，尝试修复/处置某个问题  
4. **SUCCESS** — 成功终止层，只能由 SOLUTION 进入  
5. **FAILURE** — 失败终止层，可由任意节点进入  
6. **HUB** —（可选）汇聚/编排层，用于聚合多条路径并统一分发到 FEATURE/PROBLEM（不直达 SOLUTION）

---

## 连接规则（建图约束）

1. **源点仅指向特征层**  
   - ORIGIN → FEATURE（**唯一允许**的 ORIGIN 出边）

2. **问题层必须来自于特征层**  
   - FEATURE → PROBLEM（必需）  
   - **每个 PROBLEM 必须有对应的“自特征（self-feature）”指向它**  
   - **PROBLEM 不得指向 PROBLEM**（禁止问题链式相连）

3. **解决方案层必须链接问题层**  
   - PROBLEM → SOLUTION（**唯一允许**的 SOLUTION 入边）  
   - 不允许 FEATURE/ORIGIN/HUB 直接指向 SOLUTION

4. **成功层必须来自解决方案层**  
   - SOLUTION → SUCCESS（**唯一允许**的 SUCCESS 入边）

5. **失败层可以来自任何地方**  
   - 任意节点（ORIGIN/FEATURE/PROBLEM/SOLUTION/HUB）→ FAILURE 合法

6. **HUB 的连接（如使用）**  
   - 入边：任意节点可进入 HUB  
   - 出边：仅可到 FEATURE / PROBLEM / FAILURE；**不得**到 SOLUTION / SUCCESS

---

## 运行期语义（对象引用 & 交互式）

### ORIGIN
- 仅管理子 FEATURE；进入时选择一个子 FEATURE（默认选第一个未访问；可替换为 LLM 选择器）  
- 当所有子 FEATURE 均访问完毕 → **转 FAILURE**（ORIGIN 无父节点可回退）w

### FEATURE
- 属性：`expected_state: bool`（True=期望“成立”，False=期望“不成立”）  
- 流程：  
  1) 先尝试从 **chat_log** 自动判定是否成立（不可判定则为 `None`）  
  2) 若不可判定 → 交互询问“该特征是否成立？”  
  3) 若判定 **与期望一致** → 进入子节点（**先 PROBLEM，后 FEATURE**）  
  4) 若判定 **与期望不一致** → 回退到父节点  
  5) 若无子节点可走：父为 **ORIGIN** → **FAILURE**；否则回父节点
- **链接模式（FEATURE → PROBLEM）**：每条边带 `link_mode ∈ {hard, soft}`；**进入 PROBLEM 前将 `problem.mode = link_mode`**

### PROBLEM
- 属性：`mode ∈ {hard, soft}`（**由父 FEATURE 的链接模式动态赋值**）、`solutions: List[SolutionNode]`、`child_features: List[FeatureNode]`、`parent_feature: FeatureNode`  
- 流程：  
  1) **若已访问过**：先交互确认“父特征是否已消失/该问题是否可视为已解决？”  
     - 已解决 → **回退到 `parent_feature`**  
     - 未解决 → 继续第 2 步  
  2) 选择**未访问**的子节点：**先 SOLUTION，后 FEATURE**（可替换为 LLM 选择器）  
  3) 若无子节点可走：`mode == hard` → **FAILURE**；`mode == soft` → **回 `parent_feature`**

### SOLUTION
- 执行方案 → 交互确认“是否解决？”  
  - 解决 → **SUCCESS**  
  - 未解决 → 回母 **PROBLEM**（问题切换下一个未访问子节点）

### SUCCESS
- 仅由 SOLUTION 进入；终止态

### FAILURE
- 任意节点可进入；终止态

### HUB（可选）
- 汇聚编排用；**不得**直连 SOLUTION / SUCCESS（保持“问题→方案→成功”的严格路径）

---


## 回退与终止一览（速查）

- FEATURE 判定与期望不一致 → 回父节点（父为 ORIGIN 则 **FAILURE**）  
- PROBLEM 二次进入先询问；未解决则选未访问子节点；无路可走：**hard→FAILURE / soft→回 `parent_feature`**  
- SOLUTION 未解决 → 回 `parent_problem`；解决 → **SUCCESS**  
- ORIGIN 子特征耗尽 → **FAILURE**

---

## 实现建议

- **对象互指**：`parent`/`children` 保存对象引用，返回下一节点也直接返回对象  
- **状态内聚**：`visited/confirmed_positive/resolved` 存在节点对象内部  
- **统一接口**：`interaction_callback`、`output_callback` 必须可注入  
- **可插拔选择器**：`_select_next_feature/_select_next_solution` 作为 LLM/规则引擎的挂载点
