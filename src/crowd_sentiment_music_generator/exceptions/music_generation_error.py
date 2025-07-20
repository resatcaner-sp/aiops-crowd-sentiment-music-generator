"""Music generation error exception."""


class MusicGenerationError(Exception):
    """Raised when music generation fails.
    
    Attributes:
        message: Error message
    """
    
    def __init__(self, message: str):
        """Initialize the exception.
        
        Args:
            message: Error message
        """
        self.message = message
        super().__init__(self.message)