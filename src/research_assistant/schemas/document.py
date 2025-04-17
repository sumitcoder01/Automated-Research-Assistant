from pydantic import BaseModel
from typing import List

class DocumentExtractionResponse(BaseModel):
    """Response model for the document extraction endpoint."""
    extracted_texts: List[str]
    filenames: List[str]
