import pytest
from src.research_assistant.memory.chroma_store import ChromaStore

@pytest.mark.asyncio
async def test_create_session():
    """
    Test creating a new research session.
    """
    store = ChromaStore(host="localhost", port=8000)
    
    try:
        # Create a session
        session_id = await store.create_session("Test Session")
        
        # Verify session was created
        assert session_id is not None
        assert isinstance(session_id, str)
        
        # Retrieve the session
        session = await store.get_session(session_id)
        assert session is not None
        assert session["name"] == "Test Session"
        
    finally:
        store.close()

@pytest.mark.asyncio
async def test_store_query():
    """
    Test storing a research query.
    """
    store = ChromaStore(host="localhost", port=8000)
    
    try:
        # Test data
        query = "What are the latest developments in quantum computing?"
        result = {
            "query_id": "test-query-id",
            "summary": "Test summary",
            "analysis": [],
            "citations": []
        }
        
        # Store the query
        query_id = await store.store_query(query, result)
        
        # Verify query was stored
        assert query_id == "test-query-id"
        
        # Retrieve the query
        stored_query = await store.get_query(query_id)
        assert stored_query is not None
        assert stored_query["query"] == query
        assert stored_query["summary"] == "Test summary"
        
    finally:
        store.close() 