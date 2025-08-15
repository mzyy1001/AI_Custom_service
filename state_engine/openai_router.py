# openai_router.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import json, re
from state import State
from client_init import build_openai_client

class OpenAIRouter:
    def __init__(self, client=None, model: str = "gpt-4o", use_response_format: bool = True):
        self.client = client or build_openai_client()
        self.model = model
        self.use_response_format = use_response_format

    @staticmethod
    def _clip_conf(x: Optional[float]) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, v))

    @staticmethod
    def _extract_json(s: str) -> Dict[str, Any]:
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            raise ValueError("no JSON object found")
        return json.loads(m.group(0))

    def choose_next(self, *, current_state: State, all_states: List[State],
                    user_text: str) -> Dict[str, Any]:
        candidate_ids = [s.state_id for s in all_states]
        allowed = set(candidate_ids + ["abstain"])

        def _fmt(s: State) -> str:
            return f"{s.state_id} | {s.category.name} | {s.description}"

        candidates_str = "\n".join(f"- { _fmt(s) }" for s in all_states)

        system_msg = (
            "你是一个严格的状态机路由器；只能从候选状态中选择一个 state_id，"
            "或在不确定时返回 'abstain'。"
            "FEATURE=提问/判断；SOLUTION=执行动作后仅判断是否解决。"
        )

        user_prompt = f"""
当前用户输入：{user_text}

当前状态：
- id: {current_state.state_id}
- category: {current_state.category.name}
- description: {current_state.description}

候选状态（只能选四一个，否则就 'abstain'）：
{candidates_str}

只返回 JSON（不要任何额外文字）：
{{
  "state_id": "<候选 id 或 'abstain'>",
  "confidence": <0 到 1>,
  "rationale": "<简要理由>"
}}
""".strip()

        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0
        )
        if self.use_response_format:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content

        try:
            data = self._extract_json(content)
        except Exception:
            return {"state_id": "abstain", "confidence": 0.0, "rationale": "JSON parse failed"}

        state_id = str(data.get("state_id", "")).strip()
        conf = self._clip_conf(data.get("confidence"))
        rationale = str(data.get("rationale", "")).strip()

        if state_id not in allowed:
            return {"state_id": "abstain", "confidence": 0.0, "rationale": f"'{state_id}' not in candidates"}

        return {"state_id": state_id, "confidence": conf, "rationale": rationale}
