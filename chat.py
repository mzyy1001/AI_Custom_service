#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat + produce 编排机器人（单文件版）
- 自由聊天 -> 识别“主要问题” -> 唤醒 feature_engine.produce
- 截获并改写 produce 的提问，向用户友好地问
- 若上下文可直接回答，则不再追问
- 多选项：少量集中问；选项很多时逐项确认
- 全量 I/O 统一日志
"""

import os, re, sys, json, time, datetime, threading, queue, signal
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import dotenv

dotenv.load_dotenv()

# === LLM 客户端（OpenAI 兼容） ============================================
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# === 交互子进程（用 pexpect 驱动 produce） ===============================
try:
    import pexpect
except Exception as e:
    print("缺少依赖 pexpect，请安装：pip install pexpect", file=sys.stderr)
    raise

# === 日志工具 =============================================================
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

class Logger:
    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self.path = path
        self._lock = threading.Lock()

    def log(self, who: str, text: str):
        line = f"[{now()}] [{who}] {text.rstrip()}\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

# === 配置 ==================================================================
@dataclass
class Config:
    # LLM
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    # 兼容两种写法：OPENAI_BASE_URL 优先；没有则用 OPENAI_API_BASE_URL
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE_URL")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # 单独给自由聊天一个模型名（可与上面一致）
    openai_chat_model: str = os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")

    # produce 命令
    produce_cmd: List[str] = field(default_factory=lambda: [
        sys.executable, "-m", "feature_engine.produce",
        "--tree", "feature_engine/nodes_trained_v2.json",
    ])

    # 提示语匹配（按你的输出保留不变或微调）
    pat_main_issue: re.Pattern = re.compile(r"请描述主要问题（[^）]*）\s*:\s*$")
    pat_many_features: re.Pattern = re.compile(r"检测到多个可能的子特征，请选择其一：")
    pat_enter_index: re.Pattern = re.compile(r"请输入序号（[^）]*）[^：]*：\s*$")
    pat_need_judge: re.Pattern = re.compile(r"需要判断：(.+?)\s*你的回答：\s*$")
    pat_next_node: re.Pattern = re.compile(r"^➡️ 下一节点：")
    pat_into_feature: re.Pattern = re.compile(r"^📌 进入特征:")
    pat_into_problem: re.Pattern = re.compile(r"^📌 进入问题:")
    pat_into_solution: re.Pattern = re.compile(r"^🛠 执行解决方案:")
    pat_need_judge_head: re.Pattern = re.compile(r"^🔍?\s*需要判断：(.+)")
    pat_need_judge_answer: re.Pattern = re.compile(r"^你的回答：\s*$")
    many_option_threshold: int = 5
    polite_prefix: str = "请排查以下特征，让我可以帮您定位问题"
    log_dir: str = "logs"

# === LLM 层 ===============================================================
class LLM:
    def __init__(self, cfg: Config, logger: Logger):
        self.cfg = cfg
        self.logger = logger
        if not cfg.openai_api_key and OpenAI is not None:
            self.logger.log("SYS", "未设置 OPENAI_API_KEY，后续 LLM 调用将失败。")
        if OpenAI is not None:
            self.client = OpenAI(
                api_key=cfg.openai_api_key or None,
                base_url=cfg.openai_base_url or None
            )
        else:
            self.client = None

    def _chat(self, messages: List[Dict[str, str]], response_format: Optional[dict]=None, model: Optional[str]=None) -> str:
        self.logger.log("DEBUG", f"请求 LLM: {json.dumps(messages, ensure_ascii=False)}")
        if self.client is None:
            raise RuntimeError("OpenAI 客户端不可用，请安装 openai>=1.0 并设置 OPENAI_API_KEY。")
        try:
            resp = self.client.chat.completions.create(
                model=model or self.cfg.openai_model,
                messages=messages,
                temperature=0.5,
                response_format=response_format,
                timeout=30
            )
            text = resp.choices[0].message.content or ""
            self.logger.log("LLM", text)
            return text
        except Exception as e:
            self.logger.log("ERR", f"LLM 调用失败: {e}")
            # 兜底：返回空串，避免程序崩
            return ""

    def ask_natural_yesno(self, raw: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        将技术化提示改写成自然中文的一句“是/否”问句。
        例：raw='该特征是否为正？(PDA未连接正确的网络)'
            -> 'PDA现在是不是没有连到正确的网络？（是/否）'
        要求：一句话、口语化、结尾加（是/否），不要“该特征是否为正”这种生硬措辞。
        """
        ctx = (chat_history or [])[-6:]
        sys_prompt = (
            "把技术化、系统化的判断提示改写成自然中文的一句是/否问句。"
            "避免使用“该特征是否为正”等生硬表述；保留关键名词；口语化；句末加（是/否）。"
            "只输出最终问句，不要多余说明。"
            "请注意，你是一台寻找故障的机器人"
        )
        payload = {"raw": raw, "history": ctx}
        q = self._chat(
            [{"role":"system","content":sys_prompt},
            {"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
            model=self.cfg.openai_chat_model
        )
        q = (q or "").strip()
        # 兜底
        if not q:
            q = f"{raw}？（是/否）"
        return q


    def choose_or_ask(self, chat_history: List[Dict[str, str]], options: List[str]) -> Dict[str, Any]:
        """
        让 LLM 决定下一步怎么问：
        返回结构：
        - {"action":"decide","idx":int}                         直接选出一个最可能的
        - {"action":"ask_yn","option_idx":int,"question":str}   先问一个信息增益最高的是/否
        - {"action":"ask_open","question":str}                  让用户再具体描述
        - {"action":"ask_select","question":str}                一次性列短清单请用户选
        """
        sys_prompt = (
            "你是交互规划器。根据对话历史与候选项（<=20条），决定下一步如何提问以最快三步内确定答案。\n"
            "优先级：若把握足够→直接决定(decide)；否则选一条最区分的信息做是/否确认(ask_yn)；"
            "若用户描述过少→ask_open；若选项很少(≤4)→ask_select。\n"
            "输出JSON：{"
            "\"action\":\"decide|ask_yn|ask_open|ask_select\","
            "\"idx\":int可选,"
            "\"option_idx\":int可选,"
            "\"question\":\"自然中文，一句\"}"
        )
        payload = {
            "history": chat_history[-16:],
            "options": options
        }
        out = self._chat(
            [{"role":"system","content":sys_prompt},
            {"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
            response_format={"type":"json_object"}
        )
        try:
            data = json.loads(out) if out else {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def interpret_yes_no(self, text: str) -> Optional[bool]:
        """把用户自然语言解析成是/否；返回 True/False/None"""
        t = text.strip().lower()
        yes = {"是","yes","y","好","好的","对","对的","嗯","ok","可以","已解决","解决","好了"}
        no  = {"否","no","n","不是","不对","还没","没有","未解决","不行","没好"}
        if t in yes: return True
        if t in no:  return False
        # 交给LLM兜底
        sys_prompt = '将用户回复判定为"yes"或"no"或"unsure"，只输出其中一个单词。'
        out = self._chat(
            [{"role":"system","content":sys_prompt},
            {"role":"user","content":text}],
        )
        out = (out or "").strip().lower()
        if "yes" in out or "是" in out: return True
        if "no" in out or "否" in out:  return False
        return None

    # 是否已描述主要问题
    def detect_main_issue_from_history(self, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        sys_prompt = (
            "你是一个严格的信息抽取器。根据最近的对话历史，判断用户是否已经明确说出了主要问题。"
            "输出 JSON：{\"has_issue\": true|false, \"issue_text\": \"简短问题短语或空\"}。"
            "主要问题例：机器人开不了机/无法移动/无法充电/RCS显示电量低/所有小车无法开机 等。"
            "若表述分散请归纳为最接近的一条。"
        )
        msgs = [{"role":"system","content":sys_prompt}]
        msgs += chat_history[-20:]  # 取近 20 条
        out = self._chat(msgs, response_format={"type":"json_object"})
        try:
            return json.loads(out) if out else {"has_issue": False, "issue_text": ""}
        except Exception:
            return {"has_issue": False, "issue_text": ""}

    # —— 改良自由聊天：自然对话 + 引导挖掘关键信息 ——
    def smart_reply(self, chat_history: List[Dict[str, str]]) -> str:
        sys_prompt = (
            "你是一个专业而自然的中文技术客服对话助手。"
            "目标：用简洁自然的语气对话，每次最多2句，同时提出1个具体追问，帮助尽快锁定主要问题。"
            "追问建议优先级：现象细节→设备型号/数量→是否批量/单台→是否有报错提示/灯光→最近改动→能否稳定复现。"
            "避免重复同一句话；避免机械口吻。"
        )
        msgs = [{"role":"system","content":sys_prompt}]
        msgs += chat_history[-12:]  # 引入最近上下文
        return self._chat(msgs, model=self.cfg.openai_chat_model)

    # —— 原有 detect_main_issue（保留，用于单句场景） ——
    def detect_main_issue(self, user_utterance: str) -> Dict[str, Any]:
        sys_prompt = (
            "你是一个严格的抽取器。输入为用户最新一句话，输出 JSON："
            '{"has_issue": true|false, "issue_text": "若已表述主要问题则抽取简短短语，否则空"}'
        )
        user_prompt = f"用户：{user_utterance}\n仅输出 JSON："
        out = self._chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_prompt}],
            response_format={"type":"json_object"}
        )
        try:
            return json.loads(out) if out else {"has_issue": False, "issue_text": ""}
        except Exception:
            return {"has_issue": False, "issue_text": ""}

    def prettify_question(self, raw: str, options: Optional[List[str]]=None) -> Dict[str, Any]:
        sys_prompt = (
            "你是中文产品助理。将技术性提示改写为用户友好的单问句；若有选项，"
            "请给出简洁的选择说明。输出 JSON："
            '{"ask":"问题句","mode":"single|multi|yn","options":["..",".."]}'
        )
        payload = {"raw": raw, "options": options or []}
        out = self._chat(
            [{"role":"system","content":sys_prompt},
             {"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
            response_format={"type":"json_object"}
        )
        try:
            obj = json.loads(out)
            if options:
                obj["options"] = options
            return obj
        except Exception:
            # 兜底
            return {"ask": raw, "mode":"single", "options": options or []}

    # —— 新增：从历史抽取“已知事实”，供后续自动回填 produce ——
    def extract_facts(self, chat_history: List[Dict[str,str]]) -> Dict[str, Any]:
        sys_prompt = (
            "抽取与设备故障相关的事实，输出 JSON："
            '{"features": ["提到的关键现象或特征(如RCS显示电量低, AP离线, 防火墙未开等)"],'
            ' "resolved": "yes|no|unsure"}'
        )
        msgs = [{"role":"system","content":sys_prompt}]
        msgs += chat_history[-30:]
        out = self._chat(msgs, response_format={"type":"json_object"})
        try:
            data = json.loads(out) if out else {}
            data.setdefault("features", [])
            data.setdefault("resolved", "unsure")
            return data
        except Exception:
            return {"features": [], "resolved": "unsure"}
        
    def infer_answer_from_context(self, chat_history: List[Dict[str,str]], raw_prompt: str, options: Optional[List[str]] = None) -> Dict[str, Any]:
        sys_prompt = (
            """
            你是一个严格的判断器。
            - 仅当用户历史文本里出现某个选项里的关键词或同义词时，才输出 {"can_answer": true, "answer": "<该选项文本或序号>"}。
            - 如果没有出现，必须输出 {"can_answer": false, "answer": ""}。
            - 不要推理，不要联想，不要解释。
            - 必须返回json 格式
            """
        )
        payload = {
            "raw_prompt": raw_prompt,
            "options": options or [],
            "history": chat_history[-20:]  # 取近 20 轮
        }
        out = self._chat(
            [{"role":"system","content":sys_prompt},
             {"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
            response_format={"type":"json_object"}
        )
        self.logger.log("DEBUG", f"infer_answer_from_context: {out}")
        try:
            return json.loads(out)
        except Exception:
            return {"can_answer": False, "answer": ""}
        
# === produce 交互桥 =======================================================
class ProduceBridge:
    def __init__(self, cfg: Config, logger: Logger):
        self.cfg = cfg
        self.logger = logger
        self.child: Optional[pexpect.spawn] = None
        self.alive = False

    def start(self):
        cmd = " ".join(self.cfg.produce_cmd)
        self.logger.log("SYS", f"启动 produce: {cmd}")
        self.child = pexpect.spawn(
            command=self.cfg.produce_cmd[0],
            args=self.cfg.produce_cmd[1:],
            encoding="utf-8",
            timeout=None
        )
        self.alive = True

    def stop(self):
        if self.child and self.alive:
            try:
                self.child.sendcontrol('c')
            except Exception:
                pass
            try:
                self.child.terminate(force=True)
            except Exception:
                pass
        self.alive = False

    def expect_and_read_until_prompt(self, linger: float = 0.8, max_wait: float = 5.0):
        """
        连续拉取子进程输出，直到：
        - 发现输入提示（以“：”收尾或命中已知提示正则），或者
        - 在 linger 秒内没有任何新输出，或者
        - 超过 max_wait 总等待时间
        返回: (lines, saw_prompt)
        """
        lines = []
        if not self.alive or not self.child:
            return lines, False

        def is_prompt_line(ln: str) -> bool:
            ln = ln.rstrip()
            if ln.endswith("："):
                return True
            if self.cfg.pat_enter_index.search(ln):
                return True
            if self.cfg.pat_need_judge.search(ln):
                return True
            if self.cfg.pat_many_features.search(ln):
                return True
            return False

        start = time.time()
        last_activity = time.time()
        saw_prompt = False

        while True:
            try:
                s = self.child.read_nonblocking(size=4096, timeout=0.1)
                if s:
                    for ln in s.splitlines():
                        self.logger.log("PRODUCE", ln)
                        lines.append(ln)
                        if is_prompt_line(ln):
                            saw_prompt = True
                    last_activity = time.time()
                    if saw_prompt:
                        break
            except pexpect.exceptions.TIMEOUT:
                pass
            except pexpect.exceptions.EOF:
                self.alive = False
                break

            if time.time() - last_activity >= linger:
                break
            if time.time() - start >= max_wait:
                break

        return lines, saw_prompt


    def sendline(self, text: str):
        if not self.alive or not self.child:
            return
        self.logger.log("TO_PRODUCE", text)
        self.child.sendline(text)

# === Orchestrator =========================================================
class Orchestrator:
    def __init__(self, cfg: Config):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = Logger(os.path.join(cfg.log_dir, f"session_{stamp}.log"))
        self.cfg = cfg
        self.llm = LLM(cfg, self.logger)
        self.bridge = None  # type: Optional[ProduceBridge]
        self.chat_history: List[Dict[str,str]] = []  # {"role":"user/assistant/system", "content":...}
        self.mode = "free_chat"  # or "produce"
        self.pending_question: Optional[Dict[str, Any]] = None  # 当前等待用户回答的问题上下文
        self.known_facts: Dict[str, Any] = {"features": [], "resolved": "unsure"}  # ← 新增


    def _map_answer_to_index(self, answer: str, options: List[str]) -> Optional[int]:
        """
        把 LLM/用户给的答案映射为选项的索引：
        - 若本身是纯数字且在范围内：直接用
        - 否则做关键词匹配（包含/相等），命中第一个就返回
        - 都不命中则返回 None
        """
        if answer is None:
            return None
        s = str(answer).strip()
        if s.isdigit():
            i = int(s)
            return i if 0 <= i < len(options) else None

        # 规范化做包含/相等匹配（中英文都可）
        def norm(x: str) -> str:
            return re.sub(r"\s+", "", x.lower())

        s_norm = norm(s)
        # 完全相等优先
        for i, opt in enumerate(options):
            if s_norm == norm(opt):
                return i
        # 子串包含次之（“rcs显示电量低” 命中 “RCS显示电量低”）
        for i, opt in enumerate(options):
            on = norm(opt)
            if s_norm and (s_norm in on or on in s_norm):
                return i
        return None

    # ---- 主回路（CLI 示例） ----
    def run_cli(self):
        print("🟢 诊断聊天助手已启动。输入 Ctrl+C 退出。")
        try:
            while True:
                user = input("> 你：").strip()
                if user == "":
                    continue
                self.logger.log("USER", user)
                self.chat_history.append({"role":"user", "content":user})

                # —— 新增：每次用户说话后抽取事实，便于自动回填 produce ——
                facts = self.llm.extract_facts(self.chat_history)
                # 合并去重
                feats = set(self.known_facts.get("features", [])) | set(facts.get("features", []))
                self.known_facts["features"] = list(feats)
                self.known_facts["resolved"] = facts.get("resolved", self.known_facts["resolved"])

                if self.pending_question:
                    answered = self.handle_user_answer_to_pending(user)
                    if answered:
                        continue

                if self.mode == "free_chat":
                    self.free_chat_step(user)
                else:
                    # 进入 produce：先推进一轮
                    self.produce_step()
                    # 若没有挂起问题，继续短暂快照轮询，避免“明明还能跑却等人”的假挂起
                    if self.mode == "produce" and self.pending_question is None and self.bridge and self.bridge.alive:
                        deadline = time.time() + 100  # 快速抽水窗口
                        while time.time() < deadline and self.pending_question is None and self.bridge.alive:
                            time.sleep(0.3)
                            self.produce_step()

        except KeyboardInterrupt:
            print("\n👋 已退出。")
        finally:
            if self.bridge:
                self.bridge.stop()

    # ---- 自由聊天：检测是否具备“主要问题” ----
    def free_chat_step(self, last_user_text: str):
        # 先从对话历史综合判断是否已有主要问题
        det_hist = self.llm.detect_main_issue_from_history(self.chat_history)
        if det_hist.get("has_issue"):
            issue = det_hist.get("issue_text", "").strip() or last_user_text
            print(f"🤖 明白了，你的主要问题是：{issue}。我来启动诊断引擎。")
            self.logger.log("ASSIST", f"识别主要问题：{issue}")
            self.bridge = ProduceBridge(self.cfg, self.logger)
            self.bridge.start()
            lines, _ = self.bridge.expect_and_read_until_prompt()
            if lines and any(self.cfg.pat_main_issue.search(ln) for ln in lines):
                self.bridge.sendline(issue)
            else:
                self.bridge.sendline(issue)
            self.mode = "produce"
            self.produce_step()
            return

        # 否则进入自然聊天，继续引导
        reply = self.llm.smart_reply(self.chat_history) or "我在听～可以再多描述一点具体现象吗？"
        print(f"🤖 {reply}")
        self.logger.log("ASSIST", reply)
        self.chat_history.append({"role":"assistant","content":reply})

    # ---- 简单闲聊兜底 ----
    def simple_smalltalk(self, text: str) -> str:
        # 你可改为真正 LLM 闲聊；这里保持简洁
        return "收到，我会继续记录你的描述。有更多细节也可以继续补充。"

    # ---- produce 推进一步 ----
    def produce_step(self):
        if not self.bridge or not self.bridge.alive:
            print("⚠️ 诊断引擎已结束或异常退出。若问题仍在，我们可以再启动一次。")
            self.logger.log("SYS", "produce not alive on produce_step()")
            self.mode = "free_chat"
            return

        self.logger.log("DEBUG", "进入 produce_step 循环")
        while True:
            lines, saw_prompt = self.bridge.expect_and_read_until_prompt()
            self.logger.log("DEBUG", f"produce_step: 读取 {len(lines)} 行, saw_prompt={saw_prompt}")


            # 1) 需要判断（是/否）
            need_judge_line = next((ln for ln in lines if self.cfg.pat_need_judge.search(ln)), None)
            if need_judge_line:
                m = self.cfg.pat_need_judge.search(need_judge_line)
                raw = m.group(0)
                infer = self.llm.infer_answer_from_context(self.chat_history, raw, None)
                if infer.get("can_answer"):
                    ans = (infer.get("answer","") or "").strip()
                    self.logger.log("DEBUG", f"need_judge 自动回填: {ans}")
                    self.bridge.sendline(ans)
                    continue
                # 需要用户回答
                pretty = self.llm.prettify_question(raw, None)
                ask = self.llm.ask_natural_yesno(raw, self.chat_history)
                self.pending_question = {"type": "yn_or_free", "raw": raw}
                print("🤖 " + ask)
                self.logger.log("ASSIST", f"提问（判断）：{ask}")
                return

            judge_head_line = next((ln for ln in lines if self.cfg.pat_need_judge_head.search(ln)), None)
            has_judge_answer = any(self.cfg.pat_need_judge_answer.search(ln) for ln in lines)

            if judge_head_line and has_judge_answer:
                raw = self.cfg.pat_need_judge_head.search(judge_head_line).group(1).strip()
                # 先尝试从上下文自动给出“是/否”
                infer = self.llm.infer_answer_from_context(self.chat_history, f"需要判断：{raw}", None)
                ans = (infer.get("answer","") or "").strip()
                yn = self.llm.interpret_yes_no(ans)
                if yn is True:
                    self.logger.log("DEBUG", f"need_judge(分行) 自动回填: 是  [{raw}]")
                    self.bridge.sendline("是")
                    # 继续泵下一轮
                    continue
                if yn is False:
                    self.logger.log("DEBUG", f"need_judge(分行) 自动回填: 否  [{raw}]")
                    self.bridge.sendline("否")
                    continue

                # 自动判不出来 → 向用户发“是/否”确认（自然语气）
                ask = self.llm.ask_natural_yesno(raw, self.chat_history)
                self.pending_question = {"type":"yn_or_free","raw":raw}
                print("🤖 " + ask)
                self.logger.log("ASSIST", f"提问（判断-分行）：{ask}")
                return


            # 2) 多个子特征
            many = any(self.cfg.pat_many_features.search(ln) for ln in lines)
            need_index = any(self.cfg.pat_enter_index.search(ln) for ln in lines)
            if many or need_index:
                options = []
                for ln in lines:
                    m = re.match(r"^\s*(\d+)\.\s*[A-Za-z0-9_]+:(.+)$", ln.strip())
                    if m:
                        _i, desc = m.group(1), m.group(2).strip()
                        options.append(desc)
                if options:
                    # a) 已知事实命中→直接回序号
                    facts = self.known_facts.get("features", [])
                    hit_idx = next((i for i, opt in enumerate(options) if any(k in opt for k in facts)), None)
                    if hit_idx is not None:
                        self.logger.log("DEBUG", f"known_facts 命中，回序号 {hit_idx}")
                        self.bridge.sendline(str(hit_idx))
                        continue

                    # b) 上下文推断→映射为索引
                    infer = self.llm.infer_answer_from_context(self.chat_history, "如果你能选择子特征，请选择, 不要推测，回答必须严谨", options)
                    if infer.get("can_answer"):
                        ans = (infer.get("answer","") or "").strip()
                        idx = self._map_answer_to_index(ans, options)
                        if idx is not None:
                            self.logger.log("DEBUG", f"infer 命中: answer={ans} -> idx={idx}")
                            self.bridge.sendline(str(idx))
                            continue

                    # c) 交给 LLM 决定问法
                    plan = self.llm.choose_or_ask(self.chat_history, options) or {}
                    act = plan.get("action")
                    self.logger.log("DEBUG", f"probe plan: {plan}")

                    if act == "decide" and isinstance(plan.get("idx"), int) and 0 <= plan["idx"] < len(options):
                        self.bridge.sendline(str(plan["idx"]))
                        continue

                    if act == "ask_yn" and isinstance(plan.get("option_idx"), int) and 0 <= plan["option_idx"] < len(options):
                        q = plan.get("question") or f"是否存在以下现象：{options[plan['option_idx']]}？（是/否）"
                        self.pending_question = {
                            "type": "yn_probe",
                            "options": options,
                            "target_idx": plan["option_idx"],
                            "ruled_out": set()
                        }
                        print("🤖 " + q)
                        self.logger.log("ASSIST", f"提问（YN确认）：{q}")
                        return

                    if act == "ask_open":
                        q = plan.get("question") or "能再具体描述一下故障现象/时间点/报错提示吗？"
                        self.pending_question = {
                            "type": "open_probe",
                            "options": options
                        }
                        print("🤖 " + q)
                        self.logger.log("ASSIST", f"提问（开放追问）：{q}")
                        return

                    # 默认退回单选清单（≤4项才列）
                    short = options[:4]
                    msg = f"{self.cfg.polite_prefix}以下哪一项最符合？\n" + \
                        "\n".join([f"{i}. {opt}" for i, opt in enumerate(short)])
                    msg += "\n（可回复编号或关键字；若都不符合直接说出来）"
                    self.pending_question = {"type":"select", "options": short}
                    print("🤖 " + msg)
                    self.logger.log("ASSIST", "提问（单选/短清单）：\n" + msg)
                    return

            # 3) 其他以“：”收尾的通用输入
            prompt_line = next((ln for ln in lines if ln.rstrip().endswith("：")), None)
            if prompt_line:
                infer = self.llm.infer_answer_from_context(self.chat_history, prompt_line, None)
                if infer.get("can_answer"):
                    ans = (infer.get("answer","") or "").strip()
                    self.logger.log("DEBUG", f"通用提示自动回填: {ans}")
                    self.bridge.sendline(ans)
                    continue
                pretty = self.llm.prettify_question(prompt_line, None)
                ask = pretty.get("ask") or prompt_line
                self.pending_question = {"type":"free", "raw": prompt_line}
                print(f"🤖 {self.cfg.polite_prefix}{ask}")
                self.logger.log("ASSIST", f"提问（自由输入）：{ask}")
                return

            # 4) 无交互提示
            if not saw_prompt:
                # 等一丢丢再拉一次
                time.sleep(0.3)

            # self.logger.log(f"line: {lines[-1] if lines else '无输出'}")

            # self.logger.log("DEBUG", "本轮无输入提示，返回")
            # return

            # 若没有新的输入提示，则静默等待下一轮
            # 可在此打印 produce 的阶段性信息（如进入问题/解决方案等），这里保持安静

    def ask_next_iter_option(self):
        ctx = self.pending_question
        options = ctx["options"]
        cur = ctx["cursor"]
        if cur >= len(options):
            if ctx["accepted"]:
                # accepted 里是文本，把它映射为索引
                first_text = ctx["accepted"][0]
                try:
                    idx = options.index(first_text)
                    self.logger.log("DEBUG", f"iter 结束，用 accepted[0] -> idx={idx}")
                    self.bridge.sendline(str(idx))
                except ValueError:
                    self.logger.log("DEBUG", f"iter 结束，但 accepted[0] 不在 options 中，回传空行")
                    self.bridge.sendline("")  # fallback
            else:
                self.bridge.sendline("")  # 表示都不是
            return
        opt = options[cur]
        q = f"请确认：是否存在此现象——“{opt}”？（是/否）"
        print("🤖 " + q)
        self.logger.log("ASSIST", f"提问（迭代确认）：{opt}")

    # ---- 将用户输入映射到 pending_question 并回填给 produce ----
    def handle_user_answer_to_pending(self, user_text: str) -> bool:
        if not self.pending_question:
            return False
        kind = self.pending_question["type"]

        def match_option(user_text: str, options: List[str]) -> Optional[str]:
            # 编号匹配
            m = re.match(r"^\s*(\d+)\s*$", user_text)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < len(options):
                    return str(idx)
            # 关键字模糊匹配（最长子串优先）
            txt = user_text.strip()
            ranked = sorted(options, key=lambda o: (txt in o, len(os.path.commonprefix([txt,o]))), reverse=True)
            if txt and (txt in ranked[0]):
                # 直接用关键字回传
                return txt
            return None

        if kind == "yn_or_free":
            # 规范化是/否/yes/no
            norm = user_text.strip().lower()
            if norm in ["是","yes","y","已解决","解决","ok","好了"]:
                self.bridge.sendline("是")
            elif norm in ["否","no","n","未解决","没有","还没"]:
                self.bridge.sendline("否")
            else:
                # 回传原文（有些 produce 允许自由文本）
                self.bridge.sendline(user_text)
            self.pending_question = None
            # 推进
            self.produce_step()
            return True

        if kind == "select":
            opts = self.pending_question["options"]

            # ① 输入是数字 → 直接回序号
            if user_text.strip().isdigit():
                idx = int(user_text.strip())
                if 0 <= idx < len(opts):
                    self.logger.log("DEBUG", f"user 直输数字 idx={idx}")
                    self.bridge.sendline(str(idx))
                    self.pending_question = None
                    self.produce_step()
                    return True
                else:
                    print("🤖 输入的数字超出范围，请重新输入。")
                    return True

            # ② 尝试把关键字映射为序号
            idx = self._map_answer_to_index(user_text, opts)
            if idx is not None:
                self.logger.log("DEBUG", f"user 关键字匹配 -> idx={idx}")
                self.bridge.sendline(str(idx))
                self.pending_question = None
                self.produce_step()
                return True

            # ③ 空行代表“都不是”
            self.bridge.sendline("")
            self.pending_question = None
            self.produce_step()
            return True

            # print("🤖 我没能对应到选项，请回复编号或关键字。")
            # return True

        if kind == "select_many_iter":
            ctx = self.pending_question
            opts = ctx["options"]
            cur = ctx["cursor"]
            norm = user_text.strip().lower()
            if norm in ["是","yes","y"]:
                ctx["accepted"].append(opts[cur])
            # 否/其他均视作“不存在”
            ctx["cursor"] += 1
            self.ask_next_iter_option()
            # 当迭代结束时，ask_next_iter_option 会回填
            if ctx["cursor"] >= len(opts):
                self.pending_question = None
                # 下一步输出将由 produce_step 捕获
                time.sleep(0.05)
                self.produce_step()
            return True

        if kind == "free":
            self.bridge.sendline(user_text)
            self.pending_question = None
            self.produce_step()
            return True

        if kind == "yn_probe":
            opts = self.pending_question["options"]
            t = self.pending_question["target_idx"]
            yn = self.llm.interpret_yes_no(user_text)
            if yn is True:
                # 确认命中 → 直接回序号并结束 probe
                self.bridge.sendline(str(t))
                self.pending_question = None
                self.produce_step()
                return True
            if yn is False:
                # 否认 → 从候选中排除，重新规划
                ruled = self.pending_question.get("ruled_out", set())
                ruled.add(t)
                remain = [o for i,o in enumerate(opts) if i not in ruled]
                if not remain:
                    # 一个都不符合 → 回空
                    self.bridge.sendline("")
                    self.pending_question = None
                    self.produce_step()
                    return True
                plan = self.llm.choose_or_ask(self.chat_history, remain) or {}
                act = plan.get("action")
                if act == "decide" and isinstance(plan.get("idx"), int) and 0 <= plan["idx"] < len(remain):
                    # 映射回原索引
                    decided_text = remain[plan["idx"]]
                    idx_global = opts.index(decided_text)
                    self.bridge.sendline(str(idx_global))
                    self.pending_question = None
                    self.produce_step()
                    return True
                if act == "ask_yn" and isinstance(plan.get("option_idx"), int):
                    target_text = remain[plan["option_idx"]]
                    idx_global = opts.index(target_text)
                    q = plan.get("question") or f"是否存在以下现象：{target_text}？（是/否）"
                    self.pending_question["target_idx"] = idx_global
                    self.pending_question["ruled_out"] = ruled
                    print("🤖 " + q)
                    self.logger.log("ASSIST", f"提问（YN确认）：{q}")
                    return True
                if act == "ask_open":
                    q = plan.get("question") or "能再具体描述一下故障细节吗？"
                    self.pending_question = {"type":"open_probe", "options": remain}
                    print("🤖 " + q)
                    self.logger.log("ASSIST", f"提问（开放追问）：{q}")
                    return True
                # 默认：给一个短清单
                short = remain[:4]
                msg = f"{self.cfg.polite_prefix}以下哪一项更接近？\n" + \
                    "\n".join([f"{i}. {opt}" for i, opt in enumerate(short)])
                msg += "\n（可回复编号或关键字）"
                self.pending_question = {"type":"select", "options": short}
                print("🤖 " + msg)
                self.logger.log("ASSIST", "提问（单选/短清单）：\n" + msg)
                return True

        if kind == "open_probe":
            # 把用户新描述并入上下文，再自动推进
            self.pending_question = None
            # 新描述可能让 LLM 直接能决定
            self.produce_step()
            return True


        return False

# === main ================================================================
def main():
    cfg = Config()
    orch = Orchestrator(cfg)
    orch.run_cli()

if __name__ == "__main__":
    main()
