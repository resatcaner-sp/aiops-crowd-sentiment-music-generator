"""Export error exception."""


class ExportError(Exception):
    """Exception raised for errors during export operations.
    
    Attributes:
        message: Explanation of the error
    """
    
    def __init__(self, message: str):
        """Initialize with error message.
        
        Args:
            message: Error message
        """
        self.message = message
        super().__init__(self.message)