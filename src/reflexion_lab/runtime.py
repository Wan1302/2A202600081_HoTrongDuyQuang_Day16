from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv() -> bool:  # type: ignore[override]
        return False


@dataclass
class ModelCall:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0


def _format_context(example: QAExample) -> str:
    lines = []
    for idx, chunk in enumerate(example.context, start=1):
        title = chunk.title.strip()
        text = chunk.text.strip()
        if title:
            lines.append(f"{idx}. {title}: {text}")
        else:
            lines.append(f"{idx}. {text}")
    return "\n".join(lines)


def _extract_json_payload(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty model response")
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        payload = json.loads(candidate[start : end + 1])
        if isinstance(payload, dict):
            return payload
    raise ValueError("Model response is not valid JSON")


def _usage_total(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
    return prompt_tokens, completion_tokens, total_tokens


class OpenAIRuntime:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        load_dotenv()
        self.model = model
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The openai package is not installed") from exc
        self.client = OpenAI(api_key=key)

    def _chat(self, *, messages: list[dict[str, str]], response_format: dict[str, Any] | None = None, max_tokens: int = 256) -> ModelCall:
        start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)
        content = response.choices[0].message.content or ""
        prompt_tokens, completion_tokens, total_tokens = _usage_total(response)
        return ModelCall(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

    def answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> ModelCall:
        user_prompt = [
            "<input>",
            f"<q>{example.question}</q>",
            "<ctx>",
            _format_context(example),
            "</ctx>",
            f"<attempt>{attempt_id}</attempt>",
            f"<mode>{agent_type}</mode>",
        ]
        if reflection_memory:
            user_prompt.extend(["<mem>", *[f"{idx}. {item}" for idx, item in enumerate(reflection_memory, start=1)], "</mem>"])
        user_prompt.extend(["</input>", "Return only the answer."])
        return self._chat(
            messages=[
                {"role": "system", "content": ACTOR_SYSTEM.strip()},
                {"role": "user", "content": "\n".join(user_prompt)},
            ],
            max_tokens=128,
        )

    def judge(self, example: QAExample, answer: str) -> tuple[JudgeResult, ModelCall]:
        user_prompt = "\n".join(
            [
                "<input>",
                f"<q>{example.question}</q>",
                f"<gold>{example.gold_answer}</gold>",
                f"<pred>{answer}</pred>",
                "</input>",
                "Return JSON only.",
            ]
        )
        call = self._chat(
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM.strip()},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=256,
        )
        try:
            data = _extract_json_payload(call.content)
            judge = JudgeResult.model_validate(data)
        except Exception:
            is_correct = normalize_answer(example.gold_answer) == normalize_answer(answer)
            judge = JudgeResult(
                score=1 if is_correct else 0,
                reason="Fallback exact-match judge.",
                missing_evidence=[],
                spurious_claims=[],
                correct_answer=example.gold_answer,
            )
        return judge, call

    def reflect(self, example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, ModelCall]:
        user_prompt = "\n".join(
            [
                "<input>",
                f"<q>{example.question}</q>",
                f"<gold>{example.gold_answer}</gold>",
                f"<attempt>{attempt_id}</attempt>",
                f"<judge>{judge.reason}</judge>",
                f"<missing>{json.dumps(judge.missing_evidence, ensure_ascii=False)}</missing>",
                f"<spurious>{json.dumps(judge.spurious_claims, ensure_ascii=False)}</spurious>",
                "</input>",
                "Return JSON only.",
            ]
        )
        call = self._chat(
            messages=[
                {"role": "system", "content": REFLECTOR_SYSTEM.strip()},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=192,
        )
        try:
            data = _extract_json_payload(call.content)
            reflection = ReflectionEntry.model_validate(data)
        except Exception:
            reflection = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Review the second hop and avoid stopping at a partial entity.",
                next_strategy="Re-read the supporting context and answer only after verifying the final entity.",
            )
        return reflection, call


class MockRuntime:
    def answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> ModelCall:
        from .mock_runtime import actor_answer

        start = time.perf_counter()
        content = actor_answer(example, attempt_id, agent_type, reflection_memory)
        latency_ms = int((time.perf_counter() - start) * 1000) + 160 + (attempt_id * 40) + (90 if agent_type == "reflexion" else 0)
        total_tokens = 320 + (attempt_id * 65) + (120 if agent_type == "reflexion" else 0)
        return ModelCall(content=content, total_tokens=total_tokens, prompt_tokens=total_tokens // 2, completion_tokens=total_tokens - (total_tokens // 2), latency_ms=latency_ms)

    def judge(self, example: QAExample, answer: str) -> tuple[JudgeResult, ModelCall]:
        from .mock_runtime import evaluator

        start = time.perf_counter()
        judge = evaluator(example, answer)
        latency_ms = int((time.perf_counter() - start) * 1000) + 35
        return judge, ModelCall(content=judge.model_dump_json(), total_tokens=40, prompt_tokens=20, completion_tokens=20, latency_ms=latency_ms)

    def reflect(self, example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, ModelCall]:
        from .mock_runtime import reflector

        start = time.perf_counter()
        reflection = reflector(example, attempt_id, judge)
        latency_ms = int((time.perf_counter() - start) * 1000) + 45
        return reflection, ModelCall(content=reflection.model_dump_json(), total_tokens=50, prompt_tokens=25, completion_tokens=25, latency_ms=latency_ms)


def build_runtime(model: str = "gpt-4o-mini", use_mock: bool = False, api_key: str | None = None) -> OpenAIRuntime | MockRuntime:
    if use_mock:
        return MockRuntime()
    try:
        return OpenAIRuntime(model=model, api_key=api_key)
    except Exception:
        return MockRuntime()


def combine_usage(*calls: ModelCall) -> tuple[int, int]:
    total_tokens = sum(call.total_tokens for call in calls)
    latency_ms = sum(call.latency_ms for call in calls)
    return total_tokens, latency_ms
