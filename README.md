# Question Answering Web API (FastAPI & Hugging Face)

> **Computational Thinking - Practice 01: Application Programming Interface (API)**  
> Faculty of Information Technology – VNU-HCM University of Science (HCMUS)

---

## 1. Student Information
- **Full Name**: Trương Đức Quân
- **Student ID**: 25127232
- **Class**: 25C09
- **Instructors**: Nguyen Ngoc Minh Chau, Le Duc Khoan

---

## 2. Selected Model
- **Model Name**: `distilbert/distilbert-base-cased-distilled-squad`
- **Hugging Face Link**: [https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad)
- **Model Description**:  
  This model is a fine-tuned checkpoint of `DistilBERT-base-cased` on the Stanford Question Answering Dataset (SQuAD v1.1). It is an extractive Question Answering model that reads a context paragraph and a question, then identifies the start and end character positions of the answer within the context, accompanied by a probability confidence score.

---

## 3. System Functionality Overview
This project provides a robust, production-ready Web API using **FastAPI** to serve the Question Answering model. Key capabilities include:
- **Root Overview (`GET /`)**: Returns system information, active model identifier, and interactive documentation links.
- **Health Monitoring (`GET /health`)**: Verifies service status and confirms whether the transformer model is loaded in memory.
- **Inference (`POST /predict`)**: Accepts a JSON payload containing `question` and `context`, runs inference using PyTorch, and returns structured JSON with the extracted answer, confidence score, and token/character indices.
- **Input Validation & Error Handling**: Validates input data (rejection of empty/whitespace inputs via Pydantic schema validation) and returns appropriate HTTP status codes (`200`, `400`, `422`, `500`, `503`).

---

## 4. Installation Instructions

### Prerequisites
- Python 3.10+ (Python 3.10, 3.11, 3.12, or 3.13)
- `pip` package manager

### Steps
1. **Create and activate a virtual environment:**
   - **On Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **On Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Running the Application

### Option 1: Direct Run
```bash
python main.py
```

### Option 2: Using Uvicorn CLI
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Once started:
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger UI Docs: `http://127.0.0.1:8000/docs`
- ReDoc UI Docs: `http://127.0.0.1:8000/redoc`

---

## 6. Calling the API

### 6.1 Automated Testing via Python Script
To run the complete automated test suite (implemented with the `requests` library):
```bash
python test_api.py
```

### 6.2 Example Requests & Responses

#### Endpoint: `GET /`
- **Request:**
  ```bash
  curl -X GET http://127.0.0.1:8000/
  ```
- **Response:**
  ```json
  {
    "message": "Welcome to the Question Answering API!",
    "description": "This service extracts answers to user questions from a provided context paragraph using a fine-tuned Hugging Face transformer model.",
    "model": "distilbert/distilbert-base-cased-distilled-squad",
    "endpoints": {
      "GET /": "API overview and documentation link",
      "GET /health": "System and model health status",
      "POST /predict": "Submit question and context for QA inference"
    },
    "docs_url": "/docs"
  }
  ```

#### Endpoint: `GET /health`
- **Request (cURL):**
  ```bash
  curl -X GET http://127.0.0.1:8000/health
  ```
- **Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

#### Endpoint: `POST /predict` Example 1:
- **Request (cURL):**
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"What is the capital of France?\", \"context\": \"Paris is the capital and most populous city of France.\"}"
  ```
- **Request (Python `requests`):**
  ```python
  import requests

  payload = {
      "question": "What is the capital of France?",
      "context": "Paris is the capital and most populous city of France.",
  }
  response = requests.post("http://127.0.0.1:8000/predict", json=payload)
  print(response.json())
  ```
- **Response:**
  ```json
  {
    "answer": "Paris",
    "score": 0.9946,
    "start": 0,
    "end": 5
  }
  ```

#### Endpoint: `POST /predict` Example 2: 
- **Request (Python `requests`):**
  ```python
  import requests

  payload = {
      "question": "Who created Python?",
      "context": "Python was created by Guido van Rossum and was first released in 1991."
  }
  response = requests.post("http://127.0.0.1:8000/predict", json=payload)
  print(response.json())
  ```
- **Response:**
  ```json
  {
    "answer": "Guido van Rossum",
    "score": 0.9981,
    "start": 22,
    "end": 38
  }
  ```

#### Endpoint: `POST /predict` (Input Validation / Error Handling)
- **Request with Empty Question:**
  ```json
  {
    "question": "   ",
    "context": "Paris is the capital of France."
  }
  ```
- **Response (`422 Unprocessable Entity`):**
  ```json
  {
    "detail": [
      {
        "type": "value_error",
        "loc": ["body", "question"],
        "msg": "Value error, Field cannot be empty or contain only whitespace.",
        "input": "   "
      }
    ]
  }
  ```

---

## 7. Demonstration Video Link
- **Video Link**: [https://drive.google.com/file/d/1D4oWmXFzXfRZe9PFNx2L2crD8O7LOcqR/view?usp=sharing](https://drive.google.com/file/d/1D4oWmXFzXfRZe9PFNx2L2crD8O7LOcqR/view?usp=sharing)

---

## 8. AI Declaration
- **AI Tools Used**: Antigravity/Gemini was used during this homework.
- **It was used for**:
  - Guidance on structuring the FastAPI endpoints (`GET /`, `GET /health`, `POST /predict`) and Pydantic validation schemas.
  - Reference patterns for loading and executing Hugging Face's `AutoTokenizer` and `AutoModelForQuestionAnswering` with PyTorch.
  - Scaffolding automated test cases with the `requests` library and drafting project documentation.

