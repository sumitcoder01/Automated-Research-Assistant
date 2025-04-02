from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime
import chromadb
from chromadb.config import Settings

from .config import settings

class MemoryStore:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_api_impl="rest",
            chroma_server_host=settings.CHROMA_HOST,
            chroma_server_http_port=settings.CHROMA_PORT,
            persist_directory=settings.CHROMA_PERSIST_DIR
        ))
        
        # Create or get collections
        self.sessions = self.client.get_or_create_collection("sessions")
        self.interactions = self.client.get_or_create_collection("interactions")
        
    async def create_session(self) -> Dict[str, Any]:
        """Create a new research session."""
        session_id = str(uuid.uuid4())
        session_data = {
            "id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        self.sessions.add(
            documents=[str(session_data)],
            metadatas=[session_data],
            ids=[session_id]
        )
        
        return session_data
        
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by ID."""
        result = self.sessions.get(ids=[session_id])
        if result and result["documents"]:
            return eval(result["documents"][0])
        return None
        
    async def add_interaction(
        self,
        session_id: str,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add a new interaction to a session."""
        interaction_id = str(uuid.uuid4())
        interaction_data = {
            "id": interaction_id,
            "session_id": session_id,
            "query": query,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.interactions.add(
            documents=[str(interaction_data)],
            metadatas=[interaction_data],
            ids=[interaction_id]
        )
        
        # Update session timestamp
        session = await self.get_session(session_id)
        if session:
            session["updated_at"] = datetime.utcnow().isoformat()
            self.sessions.update(
                documents=[str(session)],
                metadatas=[session],
                ids=[session_id]
            )
            
        return interaction_data
        
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve the interaction history for a session."""
        results = self.interactions.query(
            query_texts=[""],
            where={"session_id": session_id},
            n_results=limit
        )
        
        if results and results["documents"]:
            return [eval(doc) for doc in results["documents"]]
        return []
        
    async def search_sessions(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for sessions based on their interaction history."""
        results = self.interactions.query(
            query_texts=[query],
            n_results=limit
        )
        
        if results and results["documents"]:
            interactions = [eval(doc) for doc in results["documents"]]
            session_ids = list(set(interaction["session_id"] for interaction in interactions))
            return [await self.get_session(sid) for sid in session_ids]
        return [] 