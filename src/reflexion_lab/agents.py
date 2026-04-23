from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID
from .runtime import MockRuntime, build_runtime, combine_usage
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord
from .utils import normalize_answer


def _infer_failure_mode(example: QAExample, answer: str, final_score: int, attempt_id: int, max_attempts: int, reflection_memory: list[str]) -> str:
    if final_score == 1:
        return "none"
    if example.qid in FAILURE_MODE_BY_QID:
        return FAILURE_MODE_BY_QID[example.qid]
    normalized = normalize_answer(answer)
    if "london" in normalized:
        return "incomplete_multi_hop"
    if attempt_id >= max_attempts and reflection_memory:
        return "looping"
    if len(reflection_memory) > 1:
        return "reflection_overfit"
    return "wrong_final_answer"


def _adaptive_attempt_budget(example: QAExample, ceiling: int) -> int:
    budget_by_difficulty = {
        "easy": 2,
        "medium": 3,
        "hard": ceiling,
    }
    budget = budget_by_difficulty.get(example.difficulty, ceiling)
    if len(example.context) >= 12:
        budget = min(ceiling, budget + 1)
    elif len(example.context) <= 4:
        budget = min(budget, 2)
    return max(1, min(ceiling, budget))

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    model: str = "gpt-4o-mini"
    use_mock: bool = False
    runtime_mode: str = "unknown"
    def run(self, example: QAExample) -> RunRecord:
        runtime = build_runtime(model=self.model, use_mock=self.use_mock)
        self.runtime_mode = "mock" if isinstance(runtime, MockRuntime) else "openai"
        attempt_budget = self.max_attempts
        if self.agent_type == "reflexion":
            attempt_budget = _adaptive_attempt_budget(example, self.max_attempts)
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, attempt_budget + 1):
            answer_call = runtime.answer(example, attempt_id, self.agent_type, reflection_memory)
            answer = answer_call.content.strip()
            judge, judge_call = runtime.judge(example, answer)
            trace_calls = [answer_call, judge_call]
            trace_reflection: ReflectionEntry | None = None
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                token_estimate, latency_ms = combine_usage(*trace_calls)
                trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
                traces.append(trace)
                break
            if self.agent_type == "reflexion" and attempt_id < attempt_budget:
                trace_reflection, reflection_call = runtime.reflect(example, attempt_id, judge)
                reflections.append(trace_reflection)
                reflection_memory.append(trace_reflection.next_strategy)
                trace_calls.append(reflection_call)
            token_estimate, latency_ms = combine_usage(*trace_calls)
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=trace_reflection,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = _infer_failure_mode(example, final_answer, final_score, len(traces), self.max_attempts, reflection_memory)
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, attempt_budget=attempt_budget, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini", use_mock: bool = False) -> None:
        super().__init__(agent_type="react", max_attempts=1, model=model, use_mock=use_mock)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, model: str = "gpt-4o-mini", use_mock: bool = False) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, model=model, use_mock=use_mock)
