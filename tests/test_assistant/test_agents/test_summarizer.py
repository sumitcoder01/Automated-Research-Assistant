import pytest
from src.research_assistant.assistant.agents.summarizer import SummarizerAgent

@pytest.mark.asyncio
async def test_summarizer_process():
    """
    Test the summarizer agent's process method.
    """
    agent = SummarizerAgent()
    
    # Test data
    state = {
        "query": "What are the latest developments in quantum computing?",
        "search_results": [
            {
                "title": "Quantum Computing Breakthrough",
                "url": "https://example.com/1",
                "content": "Scientists have made significant progress in quantum computing..."
            },
            {
                "title": "New Quantum Algorithms",
                "url": "https://example.com/2",
                "content": "Researchers have developed new quantum algorithms..."
            }
        ]
    }
    
    # Process the state
    result = await agent.process(state)
    
    # Verify results
    assert "summary" in result
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0 