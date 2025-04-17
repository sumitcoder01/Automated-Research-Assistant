from typing import Protocol, List
from langchain_core.messages import BaseMessage

class BaseSessionStore(Protocol):
    """Defines the interface for session history storage."""

    def add_message(self, session_id: str, message: BaseMessage, embedding_provider: str = "google"):
        """Adds a message to the session history.

        Args:
            session_id: The ID of the session.
            message: The message object to add.
            embedding_provider: The provider used for generating embeddings if needed (e.g., 'google', 'openai').
        """
        ...

    def get_history(self, session_id: str, limit: int = 10) -> list[BaseMessage]:
        """Retrieves the most recent messages from the session history.

        Args:
            session_id: The ID of the session.
            limit: The maximum number of messages to retrieve.

        Returns:
            A list of BaseMessage objects representing the history, ordered chronologically.
        """
        ...
