from typing import Generator

from fastapi import Depends
from .endpoints.query import QueryRequest, QueryResponse

from ..assistant.workflow import ResearchWorkflow
from ..memory.chroma_store import ChromaStore
from ..config import settings

def get_memory_store() -> Generator[ChromaStore, None, None]:
    """
    Dependency to get the ChromaDB memory store instance.
    """
    store = ChromaStore(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT
    )
    try:
        yield store
    finally:
        store.close()

def get_workflow(
    memory_store: ChromaStore = Depends(get_memory_store)
) -> Generator[ResearchWorkflow, None, None]:
    """
    Dependency to get the research workflow instance.
    """
    workflow = ResearchWorkflow(memory_store=memory_store)
    try:
        yield workflow
    finally:
        workflow.cleanup() 