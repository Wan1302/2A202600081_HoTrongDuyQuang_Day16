from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable
from .schemas import QAExample, RunRecord

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def _coerce_context(raw_context: object) -> list[dict[str, str]]:
    if raw_context is None:
        return []
    chunks: list[dict[str, str]] = []
    if isinstance(raw_context, list):
        for item in raw_context:
            if isinstance(item, dict):
                title = str(item.get("title") or item.get("name") or item.get("heading") or "")
                text = item.get("text")
                if text is None and isinstance(item.get("sentences"), list):
                    text = " ".join(str(sentence) for sentence in item["sentences"])
                chunks.append({"title": title, "text": "" if text is None else str(text)})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                payload = item[1]
                if isinstance(payload, list):
                    text = " ".join(str(sentence) for sentence in payload)
                else:
                    text = str(payload)
                chunks.append({"title": title, "text": text})
            else:
                chunks.append({"title": "", "text": str(item)})
    elif isinstance(raw_context, dict):
        for title, value in raw_context.items():
            if isinstance(value, list):
                text = " ".join(str(sentence) for sentence in value)
            else:
                text = str(value)
            chunks.append({"title": str(title), "text": text})
    else:
        chunks.append({"title": "", "text": str(raw_context)})
    return chunks

def _coerce_record(item: dict) -> dict:
    qid = item.get("qid") or item.get("id") or item.get("_id") or item.get("question_id")
    answer = item.get("gold_answer")
    if answer is None:
        answer = item.get("answer")
    if answer is None and isinstance(item.get("answers"), list) and item["answers"]:
        answer = item["answers"][0]
    if answer is None:
        answer = ""
    context = item.get("context")
    if context is None:
        context = item.get("paragraphs") or item.get("supporting_context") or item.get("documents")
    difficulty = item.get("difficulty") or item.get("level") or "medium"
    return {
        "qid": str(qid or ""),
        "difficulty": difficulty,
        "question": str(item.get("question") or ""),
        "gold_answer": str(answer),
        "context": _coerce_context(context),
    }

def _validate_record(item: object) -> dict:
    if not isinstance(item, dict):
        raise TypeError(f"Expected dataset item to be a dict, got {type(item).__name__}")
    if "question" in item and "context" in item and "gold_answer" in item:
        return item
    return _coerce_record(item)

def load_dataset(path: str | Path) -> list[QAExample]:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Reading parquet files requires pandas") from exc
        raw_rows = pd.read_parquet(path).to_dict(orient="records")
        return [QAExample.model_validate(_coerce_record(item)) for item in raw_rows]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("examples", raw.get("records", [])))
    return [QAExample.model_validate(_validate_record(item)) for item in raw]

def save_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
