from fastapi import APIRouter, Depends, HTTPException
from typing import List

from ...schemas.query import QueryRequest, QueryResponse
from ...assistant.workflow import ResearchWorkflow
from ..deps import get_workflow

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    workflow: ResearchWorkflow = Depends(get_workflow)
) -> QueryResponse:
    """
    Submit a research query to be processed by the research assistant.
    """
    try:
        result = await workflow.process_query(request.query)
        return QueryResponse(
            query_id=result.query_id,
            summary=result.summary,
            findings=result.findings,
            citations=result.citations
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        ) 