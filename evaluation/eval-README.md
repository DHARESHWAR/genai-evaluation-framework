# Evaluation: LLM-as-a-Judge

`evaluate.py` measures how close the agent's outputs are to human-verified ground truth. It does this by asking a *
*separate LLM (the judge)** to compare each agent response against the ideal answer and produce a score. The average of
those scores across all 100 rows becomes the **baseline** — the single number every future change is measured against.

## What `evaluate.py` does

The script runs in three sequential phases:

### Phase 1 — Collect agent responses

The script reads `ground_truth_dataset.csv` (100 rows) and sends them to the agent endpoint in batches of 10, all under
a single `thread_id` for consistent scoring.

For each row, the agent returns `{agent_sentiment, agent_score}`. Now every row has both the ground truth and the
agent's prediction side by side:

| # | Movie    | Review                                                            | gt_sentiment<br/>(ground_truth) | gt_score<br/> (ground_truth) | agent_sentiment | agent_score |
|---|----------|-------------------------------------------------------------------|---------------------------------|------------------------------|-----------------|-------------|
| 1 | Lagaan   | "A masterpiece that blends cricket, patriotism..."                | positive                        | 95                           | positive        | 88          |
| 2 | Race 3   | "Absolute disaster. The dialogue is laughably bad..."             | negative                        | 8                            | negative        | 12          |
| 3 | Tamasha  | "Misunderstood on release, this is Imtiaz Ali's most personal..." | positive                        | 86                           | positive        | 74          |
| 4 | Drishyam | "Ajay Devgn is superb as a father protecting his family..."       | positive                        | 90                           | positive        | 91          |
| 5 | Sadak 2  | "The most disliked trailer on YouTube for a reason..."            | negative                        | 6                            | positive        | 30          |

Row 5 is the interesting one — the agent got the sentiment **wrong** (predicted positive when ground truth says
negative). The judge will penalize this heavily in Phase 2.

### Phase 2 — Judge each row

A **judge LLM** (a different model from the agent) evaluates every row independently. It receives three things:

1. The original review
2. The ground truth (human-verified sentiment + score)
3. The agent's response (predicted sentiment + score)

The judge returns a structured verdict:

| Field         | Type        | Description                                                                     |
|---------------|-------------|---------------------------------------------------------------------------------|
| `review`      | string      | The original review text, passed through for traceability                       |
| `eval_score`  | int (0–100) | How close the agent got to the ideal. 100 = perfect match, 0 = completely wrong |
| `eval_reason` | string      | One-line justification citing the numeric gap and sentiment match               |

#### How the judge scores

The judge follows explicit scoring rules baked into its system prompt:

- **Sentiment mismatch** is treated as a major failure — if the agent says "positive" when ground truth says "negative",
  `eval_score` is capped at 40 no matter how close the numeric score is. Getting the direction wrong is worse than being
  off by a few points.
- **Score proximity** determines the rest:
    - Within **±5** of ground truth → near-perfect
    - Within **±15** → good
    - Beyond **±30** → weak

These two dimensions — sentiment correctness and score proximity — are combined into a single `eval_score`.

#### Example judge verdicts

Using the same 5 rows from Phase 1:

| # | Review                                                            | gt           | agent        | Sentiment match? | Score gap | eval_score | eval_reason                                                        |
|---|-------------------------------------------------------------------|--------------|--------------|------------------|-----------|------------|--------------------------------------------------------------------|
| 1 | "A masterpiece that blends cricket, patriotism..."                | positive, 95 | positive, 88 | Yes              | 7         | 85         | Sentiment matched. Score gap of 7 — within good range.             |
| 2 | "Absolute disaster. The dialogue is laughably bad..."             | negative, 8  | negative, 12 | Yes              | 4         | 92         | Sentiment matched. Score gap of 4 — near-perfect.                  |
| 3 | "Misunderstood on release, this is Imtiaz Ali's most personal..." | positive, 86 | positive, 74 | Yes              | 12        | 72         | Sentiment matched. Score gap of 12 — acceptable but notable drift. |
| 4 | "Ajay Devgn is superb as a father protecting his family..."       | positive, 90 | positive, 91 | Yes              | 1         | 97         | Sentiment matched. Score gap of 1 — near-perfect alignment.        |
| 5 | "The most disliked trailer on YouTube for a reason..."            | negative, 6  | positive, 30 | **No**           | 24        | **12**     | Sentiment mismatch (capped at 40). Score gap of 24. Combined: 12.  |

Row 5 scores 12 — the sentiment mismatch caps the score at 40, and the 24-point gap drags it further down. This is the
judge working as intended: getting the direction wrong is the most expensive mistake.

### Phase 3 — Compute the baseline

After all 100 rows are judged, the script computes three metrics:

```
baseline_score = avg(eval_score across all 100 rows)
```

This is the **baseline** — the single number that represents the agent's current quality level.
> The baseline is **not** an absolute measure of quality. It's a **reference point**. Its value depends on the dataset,
> the judge model, and the scoring rubric — change any of those and the number shifts.
---
## Running

```bash
# Terminal 1: start the agent
uvicorn agent.server:app --reload --port 8000

# Terminal 2: run evaluation
python evaluation/evaluate.py
```

## Files

| File          | Purpose                                                        |
|---------------|----------------------------------------------------------------|
| `evaluate.py` | Main script — agent calls, judge scoring, baseline computation |
| `results/`    | Auto-created directory where each run's JSON output is saved   |
