import sys
import json
import time
import requests

BASE_URL = "http://127.0.0.1:8000"


def print_section(title: str):
    print("\n" * 2)
    print(f" {title}")


def test_root():
    print_section("1. Testing GET / (Root Endpoint)")
    url = f"{BASE_URL}/"
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"


def test_health():
    print_section("2. Testing GET /health (Health Check)")
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"


def test_predict_case_1():
    print_section("3. Testing POST /predict (Case 1: Geography)")
    url = f"{BASE_URL}/predict"
    payload = {
        "question": "What is the capital of France?",
        "context": "Paris is the capital and most populous city of France, with an estimated population of 2.1 million residents."
    }
    print(f"Question: {payload['question']}")
    print(f"Context:  {payload['context']}")
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start_time
    
    print(f"Status Code: {response.status_code} (took {elapsed:.3f}s)")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "Paris" in data.get("answer", ""), f"Unexpected answer: {data.get('answer')}"


def test_predict_case_2():
    print_section("4. Testing POST /predict (Case 2: Computer Science)")
    url = f"{BASE_URL}/predict"
    payload = {
        "question": "Who created Python?",
        "context": "Python was created by Guido van Rossum and was first released in 1991 as a successor to the ABC programming language."
    }
    print(f"Question: {payload['question']}")
    print(f"Context:  {payload['context']}")
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start_time
    
    print(f"Status Code: {response.status_code} (took {elapsed:.3f}s)")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "Guido van Rossum" in data.get("answer", ""), f"Unexpected answer: {data.get('answer')}"


def test_predict_empty_validation():
    print_section("5. Testing POST /predict (Validation - Empty Whitespace Input)")
    url = f"{BASE_URL}/predict"
    payload = {
        "question": "   ",
        "context": "Paris is the capital of France."
    }
    print("Testing with whitespace-only question...")
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code} (Expected 422 Unprocessable Entity)")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("[PASS] Input validation successfully rejected empty whitespace string!")


def test_predict_missing_field():
    print_section("6. Testing POST /predict (Validation - Missing Context Field)")
    url = f"{BASE_URL}/predict"
    payload = {
        "question": "What is the capital of France?"
        # Missing 'context' field
    }
    print("Testing payload missing 'context' field...")
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code} (Expected 422 Unprocessable Entity)")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("[PASS] Input validation successfully rejected missing context field!")


def interactive_mode():
    print_section("Interactive Question Answering Mode")
    print("Type 'exit' or 'quit' at any prompt to stop.\n")
    url = f"{BASE_URL}/predict"
    while True:
        context = input("Enter Context: ").strip()
        if context.lower() in ["exit", "quit"]:
            break
        question = input("Enter Question: ").strip()
        if question.lower() in ["exit", "quit"]:
            break
        
        try:
            res = requests.post(url, json={"question": question, "context": context})
            if res.status_code == 200:
                data = res.json()
                print(f"\n=> Answer: {data['answer']}")
                print(f"=> Confidence: {data['score'] * 100:.2f}%")
                print(f"=> Position: [{data['start']}:{data['end']}]\n")
            else:
                print(f"\n[Error {res.status_code}]: {res.text}\n")
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            break


def main():
    if "--interactive" in sys.argv or "-i" in sys.argv:
        interactive_mode()
        return

    print("Starting comprehensive API test suite against:", BASE_URL)
    try:
        test_root()
        test_health()
        test_predict_case_1()
        test_predict_case_2()
        test_predict_empty_validation()
        test_predict_missing_field()
        print(" ALL 6 TEST SUITES PASSED SUCCESSFULLY! ")
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Could not connect to API server at {BASE_URL}")
        print("Please ensure the FastAPI server is running before running tests:")
        print("  python main.py")


if __name__ == "__main__":
    main()
