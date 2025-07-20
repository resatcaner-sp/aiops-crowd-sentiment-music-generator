"""Exception for data API errors."""

from typing import Optional


class DataAPIError(Exception):
    """Exception raised for errors in the data API client.
    
    Attributes:
        message: Explanation of the error
        status_code: Optional HTTP status code
        response: Optional response data
    """
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None, 
        response: Optional[dict] = None
    ) -> None:
        """Initialize the exception.
        
        Args:
            message: Explanation of the error
            status_code: Optional HTTP status code
            response: Optional response data
        """
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)