"""
Evaluation script: batched agent calls with thread_id for consistency,
followed by LLM-as-a-Judge scoring against the ground truth dataset.

Usage:
    1. Start the agent server:  uvicorn server:app --reload
    2. Run evaluation:          python evaluation/evaluate.py
"""

import csv
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT", "http://localhost:8000")
GROUND_TRUTH_PATH = Path(__file__).resolve().parent.parent / "ground_truth_dataset.csv"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.groq.com/openai/v1")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or os.getenv("GROQ_API_KEY")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    eval_score: int = Field(ge=0, le=100)
    eval_reason: str


JUDGE_SYSTEM = """You are an evaluation judge. You compare an agent's sentiment-analysis
output against human-verified ground truth and score how close the agent got.

Scoring rules:
- Sentiment mismatch: cap eval_score at 40 regardless of score proximity.
- Score within ±5 of ground truth: near-perfect on the score dimension.
- Score within ±15: good.
- Score beyond ±30: weak.
- Combine sentiment match + score proximity into a single eval_score (0–100).

Return JSON: {"eval_score": int, "eval_reason": str}.
Be concise — cite the numeric gap and whether sentiment matched."""

JUDGE_TEMPLATE = """REVIEW:
{review}

GROUND TRUTH (human-verified):
  sentiment: {gt_sentiment}
  score:     {gt_score}

AGENT RESPONSE:
  sentiment: {agent_sentiment}
  score:     {agent_score}

Score the agent 0–100."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        return list(csv.DictReader(f))


def batch_analyze(rows: list[dict], thread_id: str) -> list[dict]:
    payload = {
        "reviews": [{"title": r["gt_title"], "review": r["review"]} for r in rows],
        "thread_id": thread_id,
    }
    resp = requests.post(f"{AGENT_ENDPOINT}/analyze", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["results"]


def build_judge():
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        base_url=JUDGE_BASE_URL,
        api_key=JUDGE_API_KEY,
        temperature=0,
    )
    return llm.with_structured_output(JudgeVerdict)


def judge_row(judge, row: dict) -> JudgeVerdict:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=JUDGE_TEMPLATE.format(**row)),
    ]
    return judge.invoke(messages)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    ground_truth = load_ground_truth()
    thread_id = str(uuid.uuid4())
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_batches = (len(ground_truth) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Run ID        : {run_id}")
    print(f"Thread ID     : {thread_id}")
    print(f"Samples       : {len(ground_truth)}")
    print(f"Batch size    : {BATCH_SIZE}  ({total_batches} batches)")
    print(f"Judge model   : {JUDGE_MODEL}")
    print("-" * 60)

    # ---- Step 1: Agent inference (batched, threaded) ----
    all_results = []
    for i in range(0, len(ground_truth), BATCH_SIZE):
        batch = ground_truth[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"[Agent]  Batch {batch_num}/{total_batches} ...", end=" ", flush=True)

        agent_results = batch_analyze(batch, thread_id)

        for row, agent_out in zip(batch, agent_results):
            all_results.append({
                "gt_title": row["gt_title"],
                "review": row["review"][:200] + "..." if len(row["review"]) > 200 else row["review"],
                "review_full": row["review"],
                "gt_sentiment": row["gt_sentiment"],
                "gt_score": int(row["gt_score"]),
                "agent_sentiment": agent_out["sentiment"],
                "agent_score": agent_out["score"],
            })
        print("done")

    # ---- Step 2: Judge evaluation ----
    judge = build_judge()
    for i, row in enumerate(all_results):
        print(f"[Judge]  {i + 1}/{len(all_results)}  {row['gt_title']}", end=" ... ", flush=True)
        judge_input = {k: row[k] for k in ("review_full", "gt_sentiment", "gt_score", "agent_sentiment", "agent_score")}
        judge_input["review"] = judge_input.pop("review_full")
        verdict = judge_row(judge, judge_input)
        row["eval_score"] = verdict.eval_score
        row["eval_reason"] = verdict.eval_reason
        print(f"eval_score={verdict.eval_score}")

    # ---- Step 3: Metrics ----
    eval_scores = [r["eval_score"] for r in all_results]
    sentiment_matches = [r["gt_sentiment"] == r["agent_sentiment"] for r in all_results]
    score_deltas = [abs(r["gt_score"] - r["agent_score"]) for r in all_results]

    metrics = {
        "run_id": run_id,
        "thread_id": thread_id,
        "total_samples": len(all_results),
        "baseline_score": round(mean(eval_scores), 2),
        "sentiment_accuracy_pct": round(sum(sentiment_matches) / len(sentiment_matches) * 100, 2),
        "score_mae": round(mean(score_deltas), 2),
        "agent_model": os.getenv("AGENT_MODEL", "llama-3.1-70b-versatile"),
        "judge_model": JUDGE_MODEL,
        "batch_size": BATCH_SIZE,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:.<30} {v}")

    # ---- Save ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"eval_{run_id}.json"

    for r in all_results:
        r.pop("review_full", None)

    with open(results_path, "w") as f:
        json.dump({"metrics": metrics, "details": all_results}, f, indent=2)

    print(f"\nResults saved: {results_path}")
    return metrics


if __name__ == "__main__":
    try:
        requests.get(f"{AGENT_ENDPOINT}/health", timeout=5)
    except requests.ConnectionError:
        print(f"ERROR: Agent server not reachable at {AGENT_ENDPOINT}")
        print("Start it first:  uvicorn server:app --reload")
        sys.exit(1)

    run()
