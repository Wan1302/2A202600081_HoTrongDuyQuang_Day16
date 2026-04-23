# Lab 16 - Reflexion Agent Scaffold

This repository provides a scaffold for building and evaluating a Reflexion Agent on HotpotQA-style multi-hop questions.

## 1. Repository Goal
- Start from a mock runtime and replace it with a real LLM-backed workflow.
- Understand the ReAct / Reflexion loop, reflection memory, structured evaluation, and benchmark reporting.
- Generate `report.json` and `report.md` in a format that can be autograded.

## 2. What You Need to Do
1. Build a real agent by replacing the mock runtime with a real LLM call.
2. Run a benchmark on at least 100 real HotpotQA samples.
3. Save benchmark outputs as `report.json` and `report.md`.
4. Track real token usage instead of estimated tokens.

## 3. Current Setup in This Workspace
- Python dependencies installed or used by the project:
  - `pydantic`
  - `rich`
  - `typer`
  - `pandas`
  - `python-dotenv`
  - `openai`
  - `pyarrow`
- Model used in this lab:
  - `gpt-4o-mini`
- Main dataset:
  - `data/hotpot.json`

## 4. Dataset Details
- The dataset is based on **HotpotQA**, downloaded from Hugging Face and then trimmed into a smaller working set for this lab.
- Current file used for runs: `data/hotpot.json`
- Total samples: **100**
- Difficulty distribution:
  - `easy`: **31**
  - `medium`: **43**
  - `hard`: **26**
- The dataset is a mix of multi-hop questions, so the harder samples are useful for testing whether Reflexion actually improves over a one-shot ReAct baseline.

## 5. Run Commands for Your Current Model and Data

### 5.1 Set up the environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 5.2 Make sure your OpenAI API key is available
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

Or put it into `.env`:
```env
OPENAI_API_KEY="your-api-key-here"
```

### 5.3 Run the benchmark on your current dataset
```powershell
python run_benchmark.py --dataset data/hotpot.json --out-dir outputs\sample_run --model gpt-4o-mini
```

### 5.4 Run the autograder
```powershell
python autograde.py --report-path outputs\sample_run\report.json
```

### 5.5 Optional smoke test with mock mode
```powershell
python run_benchmark.py --dataset data/hotpot.json --out-dir outputs\sample_run --model gpt-4o-mini --use-mock
python autograde.py --report-path outputs\sample_run\report.json
```

## 6. Important Note
- The current codebase is designed around an OpenAI-compatible runtime.
- For this lab, the intended model is `gpt-4o-mini`.

## 7. Bonus Features Implemented

### 7.1 `structured_evaluator`
- The evaluator returns a structured JSON object instead of free-form text.
- This makes grading and debugging more reliable because the score, reason, missing evidence, and spurious claims are explicitly separated.
- Implemented in `src/reflexion_lab/runtime.py` and supported by `src/reflexion_lab/schemas.py`.

### 7.2 `reflection_memory`
- Reflexion stores the previous lesson/strategy in memory and feeds it back into the next attempt.
- This helps the agent avoid repeating the same mistake and is the core idea behind the Reflexion loop.
- Implemented in `src/reflexion_lab/agents.py` and passed into the actor prompt in `src/reflexion_lab/runtime.py`.

### 7.3 `adaptive_max_attempts`
- The Reflexion budget is no longer fixed for every sample.
- Easy questions usually get fewer attempts, while harder or longer-context questions can use more attempts up to the ceiling.
- This reduces wasted tokens on easy samples and looks more principled during code review.
- Implemented in `src/reflexion_lab/agents.py` through `_adaptive_attempt_budget(...)`.

### 7.4 `benchmark_report_json`
- The benchmark generates a machine-readable `report.json` and a human-readable `report.md`.
- The JSON report includes metadata, summary metrics, failure breakdown, examples, extensions, and discussion.
- Implemented in `src/reflexion_lab/reporting.py`.

### 7.5 Why these bonuses are useful
- They are not only for autograding.
- They also make the code easier to explain, easier to review, and easier to extend later.
- In particular, `adaptive_max_attempts` and `reflection_memory` show that the agent is doing more than a one-shot answer loop.

## 8. Source Files
- `src/reflexion_lab/schemas.py`: data schemas for traces, records, and reports.
- `src/reflexion_lab/prompts.py`: system prompts for Actor, Evaluator, and Reflector.
- `src/reflexion_lab/mock_runtime.py`: deterministic mock LLM behavior.
- `src/reflexion_lab/agents.py`: ReAct and Reflexion agent loops.
- `src/reflexion_lab/reporting.py`: benchmark report generation.
- `run_benchmark.py`: benchmark entry point.
- `autograde.py`: report-based auto-grading helper.