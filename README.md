# Automated Research Assistant

An intelligent research assistant powered by multiple LLM agents working together to perform comprehensive research tasks.
 
## Features

- Multi-agent architecture using LangGraph
- Web search integration via Tavily
- Persistent memory using ChromaDB and Pinecone
- Support for multiple LLM providers (OpenAI, Google Gemini, DeepSeek, Anthropic)
- FastAPI-based REST API
- Comprehensive test suite
- Document processing using Pytesseract (OCR)

## Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- API keys for LLM providers (e.g., OpenAI, Gemini, DeepSeek, Anthropic)
- API key for Tavily (for web search functionality)
- API key for Pinecone (for vector database)
- pytesseract (for OCR-based document processing)
- poppler-utils (required for PDF to image conversion used in OCR)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sumitcoder01/Automated-Research-Assistant.git
   cd automated-research-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Usage

1. Start the API server:
   ```bash
   uvicorn src.research_assistant.main:app --reload
   ```

2. Access the API documentation at [`http://localhost:8000/docs`](http://localhost:8000/docs)

---

## API Usage Examples

### 1. Check API Status
Verify that the API is running.
```bash
curl http://localhost:8000/
```
**Response:**
```json
{
  "status": "Automated Research Assistant API is running!"
}
```

### 2. Create a New Research Session
```bash
curl -X POST http://localhost:8000/api/v1/sessions \
     -H "Content-Type: application/json" \
     -d '{}'
```
**Response:**
```json
{
  "session_id": "sid_f4a7c8e1-b2d3-4a5e-8f6a-7b8c9d0e1f2a",
  "message": "Session created successfully"
}
```

Alternatively, specify a custom `session_id`:
```bash
curl -X POST http://localhost:8000/api/v1/sessions \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my-research-project-01"}'
```

### 3. Submit a Research Query
```bash
curl -X POST http://localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What were the main announcements from the latest Apple event?",
           "session_id": "YOUR_SESSION_ID",
           "llm_provider": "openai"
         }'
```
**Response:**
```json
{
  "session_id": "YOUR_SESSION_ID",
  "query": "What were the main announcements from the latest Apple event?",
  "response": "Based on recent search results, the main announcements from the latest Apple event included the new M-series chips, updates to the MacBook Pro line, and advancements in their augmented reality software framework...",
  "debug_info": null
}
```

### 4. Retrieve Session History
```bash
curl -X GET "http://localhost:8000/api/v1/sessions/YOUR_SESSION_ID/history"
```
**Response:**
```json
{
  "session_id": "YOUR_SESSION_ID",
  "history": [
    {
      "role": "human",
      "content": "Hi, how are you today?"
    },
    {
      "role": "ai",
      "content": "Hello! I'm doing well, ready to help you with your research. What can I assist you with?"
    },
    {
      "role": "human",
      "content": "What were the main announcements from the latest Apple event?"
    },
    {
      "role": "ai",
      "content": "Based on recent search results, the main announcements from the latest Apple event included the new M-series chips, updates to the MacBook Pro line, and advancements in their augmented reality software framework..."
    }
  ]
}
```

### 5. Retrieve Limited History (Pagination)
```bash
curl -X GET "http://localhost:8000/api/v1/sessions/YOUR_SESSION_ID/history?limit=5"
```
**Response:**
```json
{
  "session_id": "YOUR_SESSION_ID",
  "history": [
    {
      "role": "human",
      "content": "What were the main announcements from the latest Apple event?"
    },
    {
      "role": "ai",
      "content": "Based on recent search results, the main announcements from the latest Apple event included the new M-series chips, updates to the MacBook Pro line, and advancements in their augmented reality software framework..."
    }
  ],
  "limit": 5
}
```

---

## Project Structure

```
automated-research-assistant/
├── src/                     # Main source code
│   └── research_assistant/  # Core application package
│       ├── api/             # FastAPI components
│       ├── agents/          # Agents Logic
│       ├── assistant/       # Multi-agent workflows
│       ├── llms/            # LLM provider management
│       ├── tools/           # External tools for LLM
│       ├── memory/          # ChromaDB and Pinecone integration
│       ├── prompts/         # Prompt templates
│       ├── schemas/         # Pydantic models
└── requirements.txt/        # Required Libraries
└── .env/                    # Environment File
└── .env.example/            # Example Environment File
└── .gitignore/              # Gitignore File
└── README.md/               # README File
└── tests/                   # Test suite
```

---

## Testing

Run the test suite:
```bash
pytest
```

---

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Commit your changes  
4. Push to the branch  
5. Create a Pull Request  

