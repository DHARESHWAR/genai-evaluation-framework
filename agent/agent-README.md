# Sentiment Analysis Agent

A movie-review sentiment agent built with LangChain and Groq. It takes movie reviews as input and returns a structured response: a sentiment label (`positive` / `negative`) and a positivity score (0–100).

This agent exists as the **system under test** in the evaluation framework. The evaluation harness calls its endpoint, compares the output against a human-verified ground truth dataset, and produces a baseline score.

## What it does

The agent receives a batch of movie reviews and, for each one, produces:

| Field | Type | Description |
|---|---|---|
| `title` | string | Movie title (echoed back from input) |
| `sentiment` | `"positive"` or `"negative"` | Binary classification |
| `score` | int (0–100) | Positivity intensity. 0 = fully negative, 100 = fully positive |

It uses Groq's Llama 3.1 70B model via the OpenAI-compatible SDK, with `temperature=0` and LangChain's structured output to guarantee a valid JSON response every time.

## Why `thread_id` matters

### The problem

LLMs are stateless by default. Each API call is independent — the model has no memory of what it said before. When you send 100 reviews as 100 separate calls, the model applies a fresh, uncalibrated rubric each time.

In practice this means:
- Two similarly-worded reviews might receive scores 15 points apart.
- The model's internal threshold for "positive vs. negative" can shift between calls.
- `temperature=0` helps but does not fully solve this — different input contexts produce different token probabilities, so identical intent can still yield different scores.

The result: **noisy, inconsistent outputs** that make evaluation unreliable. If the agent scores inconsistently, you can't tell whether a baseline change came from a real improvement or from scoring drift.

### The solution: conversation history as calibration

The agent accepts a `thread_id` parameter that enables **cross-batch memory**. Here's how it works:

1. **Batching** — reviews are sent 10 at a time instead of one by one. Within a batch, the model sees all 10 reviews in a single prompt and scores them relative to each other. A glowing review sitting next to a harsh one forces clearer separation.

2. **Thread-scoped history** — after the model scores a batch, both the input (reviews) and the output (scores) are appended to a conversation history keyed by `thread_id`. When the next batch arrives with the same `thread_id`, the model sees all previous batches and their scores in its context window.

```
Batch 1 (reviews 1–10)
  → Model scores them, history stores: [reviews + scores]

Batch 2 (reviews 11–20), same thread_id
  → Model sees: [batch 1 reviews + scores] + [batch 2 reviews]
  → Scores batch 2 calibrated against batch 1's decisions

Batch 3 (reviews 21–30), same thread_id
  → Model sees: [batch 1] + [batch 2] + [batch 3 reviews]
  → Three batches of calibration context
  ...
```

By batch 10, the model has seen 90 prior reviews and its own scores for all of them. It has effectively built an internal scoring rubric through its own conversation history — without any explicit fine-tuning or few-shot examples.

### What changes with vs. without `thread_id`

| | Without `thread_id` | With `thread_id` |
|---|---|---|
| **Context per call** | Single review (or single batch) in isolation | All prior batches + current batch |
| **Calibration** | None — model invents a fresh rubric each time | Self-calibrating — model references its own prior scores |
| **Score consistency** | High variance across similar reviews | Tight clustering for semantically similar reviews |
| **Evaluation reliability** | Baseline fluctuates between runs | Baseline is stable and reproducible |

### How it's implemented

The agent maintains an in-memory dictionary (`sessions`) keyed by `thread_id`:

```python
# After each batch, store the exchange
history.append(HumanMessage(content=user_content))   # the reviews
history.append(AIMessage(content=summary))            # the scores
self.sessions[thread_id] = history
```

On the next call with the same `thread_id`, this history is prepended to the prompt:

```python
messages = [SystemMessage(...)] + history + [HumanMessage(content=new_batch)]
```

The system prompt explicitly instructs the model to use this history:

> *"Use your previous scoring decisions (visible in conversation history) as calibration reference. Similar reviews must receive similar scores."*

### Practical notes

- **One `thread_id` per evaluation run.** The evaluation script generates a single UUID and reuses it across all 10 batches. This ties the entire run into one calibrated session.
- **Different runs get different `thread_id`s.** This keeps runs independent — you don't want last week's calibration leaking into today's evaluation.
- **In-memory storage is intentional.** For an evaluation workload (start server → run eval → stop server), persistence isn't needed. For production, swap `self.sessions` with Redis or a database.
- **Batch size of 10 is a balance.** Too small (1–3) and there's not enough relative context. Too large (50+) and you risk hitting token limits. 10 gives good intra-batch calibration without overloading the context window.

## Running the agent

```bash
# Set up environment
cp .env.example .env    # add your GROQ_API_KEY

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn agent.server:app --reload --port 8000
```

## API

### `POST /analyze`

```json
{
  "reviews": [
    {"title": "Lagaan", "review": "A masterpiece that blends cricket..."},
    {"title": "Race 3", "review": "Absolute disaster..."}
  ],
  "thread_id": "ddeb2595-e03e-4c8f-bb8b-d4b1c59b9e65"
}
```

Response:

```json
{
  "thread_id": "ddeb2595-e03e-4c8f-bb8b-d4b1c59b9e65",
  "results": [
    {"review": "A masterpiece that blends cricket...", "sentiment": "positive", "score": 92},
    {"review": "Absolute disaster...", "sentiment": "negative", "score": 8}
  ]
}
```

If `thread_id` is omitted, a new UUID is generated. Pass it back on subsequent requests to maintain calibration context.

### `GET /health`

Returns `{"status": "ok"}`.

## Files

| File | Purpose |
|---|---|
| `agent.py` | Core agent: LLM setup, batch processing, session history |
| `server.py` | FastAPI wrapper exposing the agent as an HTTP endpoint |
