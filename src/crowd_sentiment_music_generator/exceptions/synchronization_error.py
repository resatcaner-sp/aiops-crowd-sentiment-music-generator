"""Synchronization error exception."""

from typing import Any, Dict, Optional


class SynchronizationError(Exception):
    """Exception raised for errors in the synchronization engine.
    
    Attributes:
        message: Explanation of the error
        status_code: Optional HTTP status code
        response: Optional response data
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize SynchronizationError with message and optional details.
        
        Args:
            message: Error message
            status_code: Optional HTTP status code
            response: Optional response data
        """
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)