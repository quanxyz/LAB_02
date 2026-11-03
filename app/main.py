from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.model import QuestionAnsweringModel
model_instance: Optional[QuestionAnsweringModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
  global model_instance
  try:
    model_instance = QuestionAnsweringModel(config_path="config.yaml")
  except Exception as e:
    print(f"Error loading model during startup: {e}")
    model_instance = None
  yield


app = FastAPI(
    title="Question Answering API",
    description=(
        "A FastAPI service that answers questions from a given context "
        "using Hugging Face's"
        " distilbert-base-cased-distilled-squad."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QARequest(BaseModel):
  question: str = Field(
      ...,
      description="The question to be answered.",
      examples=["What is the capital of France?"],
  )
  context: str = Field(
      ...,
      description="The context paragraph containing the answer.",
      examples=["Paris is the capital and most populous city of France."],
  )

  @field_validator("question", "context")
  @classmethod
  def check_not_empty(cls, value: str) -> str:
    if not value or not value.strip():
      raise ValueError("Field cannot be empty or contain only whitespace.")
    return value.strip()


class QAResponse(BaseModel):
  answer: str = Field(..., description="The extracted answer span.")
  score: float = Field(..., description="Model confidence score (0.0 to 1.0).")
  start: int = Field(
      ..., description="Start character index of answer in context."
  )
  end: int = Field(..., description="End character index of answer in context.")

@app.get("/", tags=["General"])
def root():
  return {
      "message": "Welcome to the Question Answering API!",
      "description": (
          "This service extracts answers to user questions from a provided"
          " context paragraph using a fine-tuned Hugging Face transformer"
          " model."
      ),
      "model": (
          model_instance.model_path
          if model_instance
          else "distilbert/distilbert-base-cased-distilled-squad"
      ),
      "endpoints": {
          "GET /": "API overview and documentation link",
          "GET /health": "System and model health status",
          "POST /predict": "Submit question and context for QA inference",
      },
      "docs_url": "/docs",
  }


@app.get("/health", tags=["General"])
def health_check():
  is_ready = model_instance is not None
  return {
      "status": "healthy" if is_ready else "degraded",
      "model_loaded": is_ready,
  }


@app.post(
    "/predict",
    response_model=QAResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def predict(request: QARequest):
  if model_instance is None:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model is not ready or failed to initialize.",
    )

  try:
    result = model_instance.predict(
        question=request.question, context=request.context
    )
    return result
  except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Inference error: {str(e)}",
    )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
