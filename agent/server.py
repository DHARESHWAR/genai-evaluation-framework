import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.agent import SentimentAgent

app = FastAPI(title="Sentiment Analysis Agent")
agent = SentimentAgent()


class ReviewInput(BaseModel):
    title: str
    review: str


class AnalyzeRequest(BaseModel):
    reviews: list[ReviewInput]
    thread_id: str | None = None


class ResultItem(BaseModel):
    review: str
    sentiment: str
    score: int


class AnalyzeResponse(BaseModel):
    thread_id: str
    results: list[ResultItem]


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not request.reviews:
        raise HTTPException(status_code=400, detail="reviews list cannot be empty")

    thread_id = request.thread_id or str(uuid.uuid4())

    results = agent.analyze_batch(
        [r.model_dump() for r in request.reviews],
        thread_id,
    )

    return AnalyzeResponse(
        thread_id=thread_id,
        results=[
            ResultItem(review=r.review, sentiment=r.sentiment, score=r.score)
            for r in results
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
