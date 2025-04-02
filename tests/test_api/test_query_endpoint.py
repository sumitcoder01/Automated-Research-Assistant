import pytest
from fastapi.testclient import TestClient
from src.research_assistant.main import app

client = TestClient(app)

def test_submit_query():
    """
    Test submitting a research query.
    """
    response = client.post(
        "/api/v1/query",
        json={
            "query": "What are the latest developments in quantum computing?",
            "session_id": None
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "query_id" in data
    assert "summary" in data
    assert "findings" in data
    assert "citations" in data
    assert "created_at" in data 