from fastapi import FastAPI
from pydantic import BaseModel

from src.agent_core import ask_agent


app = FastAPI(
    title="Research Agent API",
    description="Local AI research assistant powered by Qwen3",
    version="1.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Research Agent API is running"
    }


@app.post("/ask")
def ask(request: QuestionRequest):

    try:

        answer = ask_agent(
            request.question
        )

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:

        return {
            "error": str(e)
        }