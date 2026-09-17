from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from anomaly_detector import AnomalyDetector
from data_engine import TicketDataEngine
from llm_engine import LLMEngine


app = FastAPI(
    title="DotMappers AI Support Ticket API",
    description=(
        "AI-powered support ticket analytics API "
        "with natural-language querying and anomaly detection."
    ),
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


data_engine = TicketDataEngine()

llm_engine = LLMEngine()

anomaly_detector = AnomalyDetector(
    data_engine.get_dataframe()
)


@app.get("/")
def home():
    return {
        "message": "DotMappers AI Support Ticket API is running",
        "status": "success",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "total_tickets": data_engine.get_total_tickets(),
    }


@app.get("/summary")
def summary():
    return data_engine.get_summary()


@app.get("/anomalies")
def anomalies():
    detected = anomaly_detector.detect_anomalies()

    columns = [
        "ticket_id",
        "category",
        "priority",
        "status",
        "response_time_hrs",
        "resolution_time_hrs",
        "customer_rating",
        "anomaly_reasons",
    ]

    anomaly_records = (
        detected[columns]
        .astype(object)
        .where(
            detected[columns].notna(),
            None,
        )
        .to_dict(orient="records")
    )

    return {
        "summary": anomaly_detector.get_summary(),
        "anomalies": anomaly_records,
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    answer = llm_engine.ask(question)

    return {
        "question": question,
        "answer": answer,
    }