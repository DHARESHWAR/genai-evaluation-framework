# GenAI Agents Evaluation Framework

> A step-by-step framework to test and score any LLM-powered agent — so you can measure exactly how well it performs before and after every change.

![Agent Evaluation Framework](resources/evaluation-framework.svg)

---

## Why You Need a Structured Evaluation

Every GenAI agent takes a user instruction and produces an output. But there is no built-in mechanism to measure how accurate that output is. Unlike traditional software where you can check `expected == actual`, LLM outputs vary every time — and there is no automatic way to tell if the output is correct, partially correct, or wrong.

On top of that, new LLM models are released frequently. Each model behaves differently, and there is no guarantee that a model that works well for one use case will work well for another. Blindly switching models — or changing a prompt, or updating a retrieval config — can silently degrade your agent's performance with no way to detect it.

You need a structured evaluation framework that can measure overall agent performance, and show you exactly what changed — before and after every update.

This framework works for **any type of agent** — RAG assistants, summarizers, classifiers, code generators, or multi-step tool-using agents. To explain how it works, I'll use a running example: a **movie-review sentiment analyzer**. This agent takes a movie review as input and produces two outputs — a sentiment label (`positive` or `negative`) and a positivity score (0–100).  The evaluation framework measures how accurately the agent produces these outputs across 100 such reviews.

---

## The Three-Step Framework

No matter what your agent does, evaluation comes down to three steps:

1. **Build a Ground Truth Dataset** — a small set of inputs with human-verified correct outputs, matching the exact format your agent produces.
2. **Build (or Wrap) the Agent** — the agent you want to test, running behind an API endpoint so the evaluation tests it the same way real users would use it.
3. **Evaluate with LLM-as-a-Judge** — use a separate LLM to score each agent response against the ground truth, and compute a single baseline score you can track over time.

Every agent takes inputs and produces outputs. This framework needs three things:
- (a) A fixed set of test inputs with known correct answers,
- (b) The agent's actual outputs for those same inputs, and
- (c) A scoring method that compares the two and produces a number.

Everything below explains how to do each of these well.

---

## Step 1 — Build the Ground Truth Dataset

Each row in the ground truth dataset has three parts: the **input** (what you send to the agent), the **expected output** (the correct answer, verified by a human), and optional **metadata** (like difficulty level or category) for analyzing results later.

```python
{
    "input":    <what the agent receives>,
    "expected": <the correct output, in the same format the agent produces>,
    "meta":     {...},
}
```

For our running example, the agent is a **movie-review sentiment analyzer**. It receives a movie review as plain text and produces two outputs: a **sentiment label** (`positive` or `negative`) and a **positivity score** (an integer from 0 to 100, where 0 means the review is entirely negative and 100 means it is overwhelmingly positive). Think of the sentiment as the classification and the score as the intensity.

Because the agent produces structured output, the ground truth dataset must use the exact same format — same fields, same value ranges — so we can compare them directly during evaluation.

### Example schema (sentiment agent)

```python
{
    "gt_title":      str,         # movie title (for identification)
    "review":        str,         # the review text — this is the input to the agent
    "gt_sentiment":  "positive" | "negative",  # human-verified sentiment label
    "gt_score":      int,         # 0–100 positivity score, assigned by a human
}
```

The `gt_` prefix stands for "ground truth." These are the human-verified correct answers. During evaluation, they will be compared against the agent's predictions (`agent_sentiment`, `agent_score`) to measure how accurate the agent is.

### Where does the data come from?

The general rule: **start with real data, then have a human verify it.** Raw data alone gives you a dataset. Human verification makes it *ground truth*.

For our sentiment example, we curated 100 Bollywood movie reviews — 50 positive, 50 negative — covering classics (Sholay, Andaz Apna Apna) and recent releases (Kill, 12th Fail). Each row was verified by a human: the sentiment label was confirmed and the 0–100 positivity score was assigned by reading the actual review, not copied from a star rating.

```
gt_title,review,gt_sentiment,gt_score
Lagaan,"A masterpiece that blends cricket, patriotism, and drama...",positive,95
Race 3,"Absolute disaster. The dialogue is laughably bad...",negative,8
```

The full dataset is in [`ground_truth_dataset.csv`](evaluation/ground_truth_dataset.csv) — 100 rows, balanced sentiment, score range 4–98.

Where your ground truth comes from depends on the type of agent:

| Agent type | Typical source |
|---|---|
| **Classification / sentiment** | Public datasets (IMDB, SST) + human review pass |
| **RAG** | Real user questions + expert-written reference answers |
| **Summarization** | Documents + human-written summaries |
| **Code generation** | Task specs + verified reference implementations |
| **Tool-using agent** | Real task logs + human-verified action sequences |

Whatever the source, the human verification step is what makes it ground truth instead of just data.

---

## Step 2 — Build (or Wrap) the Agent

The agent is whatever system your team has built or is building. The framework only requires three things:

- **Consistent output format.** The same type of input always produces the same type of output — same fields, same structure.
- **Run it like production.** Wrap it in the same API endpoint you would use in production. The evaluation should test the full system, not a shortcut version of it.
- **Structured output.** If your agent returns free text, add a parsing step so the outputs can be compared against the ground truth numerically.

For the running example: the agent receives a review and returns a sentiment label + score. It uses **LangChain** with **Groq** (Llama 3.1 70B) — chosen because it is fast and inexpensive for iteration. You can swap in any LLM provider; the framework works the same way.

The full agent implementation — including the LangChain setup, system prompt with scoring guidelines, session management, and the FastAPI server — is documented in [`agent/agent-README.md`](agent/agent-README.md).

### The consistency problem: batching + thread_id

If you send 100 reviews as 100 separate API calls, the LLM treats each one independently. Review #1 and review #47 might get scored using different internal standards — a "pretty good" review could score 72 in one call and 65 in another. Even with `temperature=0`, the scores drift because each call has no context about how previous reviews were scored.

Two techniques solve this:

**Batching** — send 10 reviews per request instead of one. Within a single batch, the LLM sees all 10 reviews together and scores them relative to each other. A glowing review next to a harsh one forces the model to separate their scores clearly.

**Thread ID (conversation history)** — across batches, pass a `thread_id` that carries the conversation history forward. When batch 2 arrives, the LLM can see batch 1's reviews *and the scores it gave them*. This gives the model a reference point: "I scored a similar review 78 in the last batch, so this one should be around 75."

```python
thread_id = str(uuid.uuid4())        # one ID for the entire evaluation run
for i in range(0, 100, BATCH_SIZE):   # 10 batches of 10
    batch = ground_truth[i : i + BATCH_SIZE]
    results = requests.post(ENDPOINT, json={
        "reviews": batch,
        "thread_id": thread_id,       # same ID → LLM sees all prior batches
    })
```

The LLM now scores all 100 reviews using a consistent standard because every batch builds on the scores from previous batches.

For a detailed explanation of how `thread_id` works — including the session history implementation, what changes with vs. without it, and practical notes on batch sizing — see [`agent/agent-README.md` → Why thread_id matters](agent/agent-README.md#why-thread_id-matters).

---

## Step 3 — Evaluate with LLM-as-a-Judge

For each row in the ground truth dataset, send the input to the agent, then ask a **judge LLM** to score how close the agent's output is to the correct answer. The judge is a separate LLM whose only job is to compare two outputs and produce a structured score. This approach works for any type of agent output — summaries, answers, plans, code, or classifications.

### How it works

The evaluation script ([`evaluation/evaluate.py`](evaluation/evaluate.py)) runs in three phases:

**Phase 1 — Collect agent responses.** The script reads the ground truth CSV and sends reviews to the agent in batches of 10, using a shared `thread_id`. After this phase, every row has both the correct answer and the agent's prediction side by side:

| # | Review | gt_sentiment | gt_score | agent_sentiment | agent_score |
|---|---|---|---|---|---|
| 1 | "A masterpiece that blends cricket, patriotism..." | positive | 95 | positive | 88 |
| 2 | "Absolute disaster. The dialogue is laughably bad..." | negative | 8 | negative | 12 |
| 3 | "Misunderstood on release, this is Imtiaz Ali's most personal..." | positive | 86 | positive | 74 |
| 4 | "Ajay Devgn is superb as a father protecting his family..." | positive | 90 | positive | 91 |
| 5 | "The most disliked trailer on YouTube for a reason..." | negative | 6 | positive | 30 |

**Phase 2 — Judge each row.** A judge LLM (a *different model* from the agent, to avoid the model favoring its own style of output) evaluates every row independently. It receives the review, the correct answer, and the agent's response, then returns:

| Field | Type | Description |
|---|---|---|
| `review` | string | The original review text (included for traceability) |
| `eval_score` | int (0–100) | How close the agent's output is to the correct answer |
| `eval_reason` | string | A one-line explanation of why it scored this way |

The judge follows explicit scoring rules: if the agent got the sentiment wrong (e.g., said "positive" when the correct answer is "negative"), the `eval_score` is capped at 40 — because getting the direction wrong is the biggest possible mistake. The rest of the score depends on how close the numeric score is to the correct one (within ±5 = near-perfect, within ±15 = good, beyond ±30 = poor).

| # | Review | gt | agent | Sentiment match? | Score gap | eval_score | eval_reason |
|---|---|---|---|---|---|---|---|
| 1 | "A masterpiece that blends cricket..." | positive, 95 | positive, 88 | Yes | 7 | 85 | Sentiment matched. Score gap of 7 — within good range. |
| 2 | "Absolute disaster..." | negative, 8 | negative, 12 | Yes | 4 | 92 | Sentiment matched. Score gap of 4 — near-perfect. |
| 3 | "Misunderstood on release..." | positive, 86 | positive, 74 | Yes | 12 | 72 | Sentiment matched. Score gap of 12 — acceptable but noticeable drift. |
| 4 | "Ajay Devgn is superb..." | positive, 90 | positive, 91 | Yes | 1 | 97 | Sentiment matched. Score gap of 1 — near-perfect. |
| 5 | "The most disliked trailer..." | negative, 6 | positive, 30 | **No** | 24 | **12** | Sentiment mismatch (capped at 40). Score gap of 24. Combined: 12. |

**Phase 3 — Compute the baseline.** The average of all 100 `eval_score` values becomes the **baseline** — the single number that represents how well your agent performs right now. Two additional metrics are computed alongside it:

```
baseline_score       = avg(eval_score across all 100 rows)
sentiment_accuracy   = % of rows where agent_sentiment == gt_sentiment
score_mae            = avg(|gt_score - agent_score|) across all rows
```

For the full evaluation walkthrough — including the judge prompt, scoring rules, and how the three metrics work together — see [`evaluation/eval-README.md`](evaluation/eval-README.md).

---

## Applying the Framework to Other Agents

The sentiment example was just for illustration. The same pattern works for any type of agent:

| Agent | Ground-truth row | Measurable metric | What the judge evaluates |
|---|---|---|---|
| **RAG assistant** | question → reference answer + citations | citation precision/recall | factual accuracy, completeness |
| **Summarizer** | document → human summary + key points | ROUGE / BERTScore | coverage, conciseness, faithfulness |
| **Classifier** | input → label | accuracy, F1 | quality of reasoning on wrong answers |
| **Code agent** | task → reference code + tests | tests pass, lint, type-check | readability, scope |
| **Tool-using agent** | task → expected actions + final output | action sequence match | result quality, efficiency |
| **Chat / assistant** | user message → reference response | — | helpfulness, safety, tone |

> The three steps — ground truth, agent under test, judge — stay the same for every agent. Only the data format and the scoring rules change.

---

## Project Structure

```
genai-evaluation-framework/
├── README.md                       ← this document
├── requirements.txt
├── .env.example
├── agent/
│   ├── agent-README.md             ← detailed agent documentation
│   ├── agent.py                    ← core agent: LangChain + Groq, session history
│   └── server.py                   ← FastAPI endpoint
└── evaluation/
    ├── eval-README.md              ← detailed evaluation documentation
    ├── evaluate.py                 ← evaluation script: batch → judge → metrics
    ├── ground_truth_dataset.csv    ← 100 human-verified Bollywood reviews
    └── results/                    ← JSON output per run
```

- **[`agent/agent-README.md`](agent/agent-README.md)** — how the agent works, why `thread_id` matters for consistent scoring, the session history implementation, and the API reference.
- **[`evaluation/eval-README.md`](evaluation/eval-README.md)** — how the LLM-as-a-Judge evaluation works end to end: the three phases, scoring rules, how the baseline is computed, and how the three metrics work together.
