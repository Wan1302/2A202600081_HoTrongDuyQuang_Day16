# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot.json
- Mode: openai
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.72 | 0.82 | 0.1 |
| Avg attempts | 1 | 1.4 | 0.4 |
| Avg token estimate | 1650.99 | 2464.11 | 813.12 |
| Avg latency (ms) | 2470.31 | 3952.15 | 1481.84 |

## Failure modes
```json
{
  "overall": {
    "wrong_final_answer": 36,
    "none": 154,
    "looping": 10
  },
  "by_agent": {
    "react": {
      "wrong_final_answer": 28,
      "none": 72
    },
    "reflexion": {
      "wrong_final_answer": 8,
      "none": 82,
      "looping": 10
    }
  },
  "by_difficulty": {
    "hard": {
      "wrong_final_answer": 29,
      "none": 97,
      "looping": 8
    },
    "easy": {
      "none": 41,
      "wrong_final_answer": 3
    },
    "medium": {
      "none": 16,
      "wrong_final_answer": 4,
      "looping": 2
    }
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- adaptive_max_attempts
- benchmark_report_json

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains or caused the agent to overcorrect on later attempts.
