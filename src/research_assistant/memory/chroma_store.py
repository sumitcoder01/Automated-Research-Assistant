from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
from datetime import datetime

from ...config import settings

class ChromaStore:
    def __init__(self, host: str, port: int):
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        self.collection = self.client.get_or_create_collection(
            name="research_sessions"
        )
        
    async def create_session(self, name: str) -> str:
        """
        Create a new research session.
        """
        session_id = str(datetime.utcnow().timestamp())
        
        self.collection.add(
            documents=[f"Session: {name}"],
            metadatas=[{
                "type": "session",
                "name": name,
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "query_count": 0
            }],
            ids=[session_id]
        )
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve a research session.
        """
        try:
            result = self.collection.get(
                ids=[session_id]
            )
            
            if not result["documents"]:
                return None
                
            metadata = result["metadatas"][0]
            return {
                "session_id": session_id,
                "name": metadata["name"],
                "created_at": datetime.fromisoformat(metadata["created_at"]),
                "last_activity": datetime.fromisoformat(metadata["last_activity"]),
                "query_count": metadata["query_count"]
            }
            
        except Exception:
            return None
    
    async def store_query(self, query: str, result: Dict) -> str:
        """
        Store a research query and its results.
        """
        query_id = result["query_id"]
        
        # Store query and results
        self.collection.add(
            documents=[query],
            metadatas=[{
                "type": "query",
                "query": query,
                "summary": result["summary"],
                "analysis": result["analysis"],
                "citations": result["citations"],
                "created_at": datetime.utcnow().isoformat()
            }],
            ids=[query_id]
        )
        
        return query_id
    
    async def get_query(self, query_id: str) -> Optional[Dict]:
        """
        Retrieve a research query and its results.
        """
        try:
            result = self.collection.get(
                ids=[query_id]
            )
            
            if not result["documents"]:
                return None
                
            metadata = result["metadatas"][0]
            return {
                "query_id": query_id,
                "query": metadata["query"],
                "summary": metadata["summary"],
                "analysis": metadata["analysis"],
                "citations": metadata["citations"],
                "created_at": datetime.fromisoformat(metadata["created_at"])
            }
            
        except Exception:
            return None
    
    def close(self):
        """
        Close the ChromaDB client.
        """
        self.client.close() 