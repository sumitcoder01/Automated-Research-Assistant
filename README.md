# Automated Research Assistant

An intelligent research assistant powered by multiple LLM agents working together to perform comprehensive research tasks.

## Features

- Multi-agent architecture using LangGraph
- Web search integration via Searx
- Persistent memory using ChromaDB
- Support for multiple LLM providers (OpenAI, Google Gemini, DeepSeek)
- FastAPI-based REST API
- Comprehensive test suite

## Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- API keys for LLM providers
- Searx instance URL

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automated-research-assistant.git
cd automated-research-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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

2. Access the API documentation at `http://localhost:8000/docs`

## Project Structure

```
automated-research-assistant/
├── src/                      # Main source code
│   └── research_assistant/   # Core application package
│       ├── api/             # FastAPI components
│       ├── assistant/       # Multi-agent logic
│       ├── llms/           # LLM provider management
│       ├── memory/         # ChromaDB integration
│       ├── prompts/        # Prompt templates
│       ├── schemas/        # Pydantic models
│       └── utils/          # Utility functions
└── tests/                  # Test suite
```

## Testing

Run the test suite:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 