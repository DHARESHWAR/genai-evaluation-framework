import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class ReviewResult(BaseModel):
    review: str = Field(description="The original review text")
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment label")
    score: int = Field(ge=0, le=100, description="0-100 positivity score")


class BatchOutput(BaseModel):
    results: list[ReviewResult]


SYSTEM_PROMPT = """You are a movie-review sentiment analyst. For each review provided, return:
- title: the movie title as given
- sentiment: "positive" or "negative"
- score: integer 0–100 (0 = fully negative, 100 = fully positive)

Scoring guidelines for calibration:
  85–100  Overwhelmingly positive
  65–84   Mostly positive, minor flaws noted
  51–64   Mixed, leaning positive
  36–50   Mixed, leaning negative
  16–35   Mostly negative, some merit acknowledged
  0–15    Overwhelmingly negative

IMPORTANT: You will receive reviews in batches. Use your previous scoring decisions
(visible in conversation history) as calibration reference. Similar reviews must
receive similar scores. Maintain consistent standards across all batches."""


class SentimentAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("AGENT_MODEL", "llama-3.1-70b-versatile"),
            base_url=os.getenv("AGENT_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
        self.structured_llm = self.llm.with_structured_output(BatchOutput)
        self.sessions: dict[str, list] = {}

    def analyze_batch(self, reviews: list[dict], thread_id: str) -> list[ReviewResult]:
        history = self.sessions.get(thread_id, [])

        review_block = "\n\n".join(
            f"Review {i + 1} — \"{r['title']}\":\n{r['review']}"
            for i, r in enumerate(reviews)
        )
        user_content = f"Analyze these {len(reviews)} reviews:\n\n{review_block}"

        messages = (
                [SystemMessage(content=SYSTEM_PROMPT)]
                + history
                + [HumanMessage(content=user_content)]
        )

        result = self.structured_llm.invoke(messages)

        summary = json.dumps(
            [{"review": r.review[:100], "sentiment": r.sentiment, "score": r.score} for r in result.results],
            indent=2,
        )
        history.append(HumanMessage(content=user_content))
        history.append(AIMessage(content=summary))
        self.sessions[thread_id] = history

        return result.results

