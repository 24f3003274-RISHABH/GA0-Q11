
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

# Allow requests from the TDS webpage.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()


class SentimentRequest(BaseModel):
    sentences: List[str]


def analyze_sentiment(sentence: str) -> str:
    scores = analyzer.polarity_scores(sentence)
    compound = scores["compound"]

    if compound >= 0.05:
        return "happy"
    elif compound <= -0.05:
        return "sad"
    else:
        return "neutral"


@app.post("/sentiment")
async def analyze_batch(request: SentimentRequest):
    results = []

    for sentence in request.sentences:
        results.append({
            "sentence": sentence,
            "sentiment": analyze_sentiment(sentence),
        })

    return {"results": results}


@app.get("/")
async def root():
    return {"message": "Sentiment API is running"}
