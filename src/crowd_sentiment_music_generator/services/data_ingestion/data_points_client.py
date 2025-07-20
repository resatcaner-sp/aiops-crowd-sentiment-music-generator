"""Client for consuming real-time match data points."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp
from pydantic import ValidationError

from crowd_sentiment_music_generator.exceptions.data_api_error import DataAPIError
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.utils.error_handling import with_error_handling


class DataPointsClient:
    """Client for consuming real-time match data points.
    
    This client connects to a data points API and provides methods for
    subscribing to match events and retrieving match state.
    
    Attributes:
        base_url: Base URL for the data points API
        api_key: API key for authentication
        session: aiohttp client session
        ws_connection: WebSocket connection
        match_id: Current match ID
        is_connected: Connection status
        event_callbacks: Callbacks for match events
        logger: Logger instance
    """
    
    def __init__(
        self, 
        base_url: str, 
        api_key: Optional[str] = None,
        retry_interval: float = 5.0,
        max_retries: int = 3
    ) -> None:
        """Initialize the data points client.
        
        Args:
            base_url: Base URL for the data points API
            api_key: Optional API key for authentication
            retry_interval: Interval between connection retries in seconds
            max_retries: Maximum number of connection retries
        """
        self.base_url = base_url
        self.api_key = api_key
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None
        self.match_id: Optional[str] = None
        self.is_connected = False
        self.event_callbacks: List[Callable[[MatchEvent], Any]] = []
        self.reconnect_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling
    async def connect(self, match_id: str) -> None:
        """Establish connection to data points API for a specific match.
        
        Args:
            match_id: ID of the match to connect to
            
        Raises:
            DataAPIError: If connection fails
        """
        self.match_id = match_id
        
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        # Construct WebSocket URL
        ws_url = f"{self.base_url}/ws/matches/{match_id}/events"
        if self.api_key:
            ws_url += f"?api_key={self.api_key}"
        
        try:
            self.logger.info(f"Connecting to data points API for match {match_id}")
            self.ws_connection = await self.session.ws_connect(ws_url)
            self.is_connected = True
            self.logger.info(f"Connected to data points API for match {match_id}")
            
            # Start listening for events
            asyncio.create_task(self._listen_for_events())
        except aiohttp.ClientError as e:
            self.is_connected = False
            raise DataAPIError(f"Failed to connect to data points API: {str(e)}")
    
    @with_error_handling
    async def disconnect(self) -> None:
        """Disconnect from the data points API.
        
        Raises:
            DataAPIError: If disconnection fails
        """
        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()
            self.reconnect_task = None
        
        if self.ws_connection:
            try:
                await self.ws_connection.close()
                self.logger.info("Disconnected from data points API")
            except Exception as e:
                raise DataAPIError(f"Failed to disconnect from data points API: {str(e)}")
            finally:
                self.ws_connection = None
                self.is_connected = False
        
        if self.session:
            await self.session.close()
            self.session = None
    
    @with_error_handling
    async def subscribe_to_events(self, callback: Callable[[MatchEvent], Any]) -> None:
        """Subscribe to match events with a callback function.
        
        Args:
            callback: Function to call when an event is received
            
        Raises:
            DataAPIError: If not connected to the API
        """
        if not self.is_connected:
            raise DataAPIError("Not connected to data points API")
        
        self.event_callbacks.append(callback)
        self.logger.info("Subscribed to match events")
    
    @with_error_handling
    async def get_latest_match_state(self) -> Dict[str, Any]:
        """Get the current match state including score, time, etc.
        
        Returns:
            Dictionary containing match state
            
        Raises:
            DataAPIError: If API request fails
        """
        if not self.match_id:
            raise DataAPIError("No match ID specified")
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Construct API URL
        url = f"{self.base_url}/api/matches/{self.match_id}/state"
        if self.api_key:
            url += f"?api_key={self.api_key}"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise DataAPIError(
                        f"Failed to get match state: {error_text}",
                        status_code=response.status,
                        response={"error": error_text}
                    )
                
                data = await response.json()
                return data
        except aiohttp.ClientError as e:
            raise DataAPIError(f"Failed to get match state: {str(e)}")
    
    async def _listen_for_events(self) -> None:
        """Listen for events from the WebSocket connection.
        
        This method runs in a loop until the connection is closed.
        """
        if not self.ws_connection:
            self.logger.error("WebSocket connection not established")
            return
        
        try:
            async for msg in self.ws_connection:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._process_event(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket connection closed with error: {self.ws_connection.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    self.logger.info("WebSocket connection closed")
                    break
        except Exception as e:
            self.logger.error(f"Error in WebSocket connection: {str(e)}")
        finally:
            self.is_connected = False
            # Try to reconnect
            if self.match_id:
                self.reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect to the data points API."""
        if not self.match_id:
            self.logger.error("Cannot reconnect: No match ID specified")
            return
        
        for attempt in range(1, self.max_retries + 1):
            self.logger.info(f"Reconnection attempt {attempt}/{self.max_retries}")
            try:
                await self.connect(self.match_id)
                self.logger.info("Reconnected successfully")
                return
            except DataAPIError as e:
                self.logger.error(f"Reconnection failed: {e.message}")
                if attempt < self.max_retries:
                    self.logger.info(f"Retrying in {self.retry_interval} seconds")
                    await asyncio.sleep(self.retry_interval)
                else:
                    self.logger.error("Max reconnection attempts reached")
    
    async def _process_event(self, data: str) -> None:
        """Process an event received from the WebSocket.
        
        Args:
            data: JSON string containing event data
        """
        try:
            event_data = json.loads(data)
            event = MatchEvent(**event_data)
            
            # Call all registered callbacks
            for callback in self.event_callbacks:
                try:
                    await callback(event) if asyncio.iscoroutinefunction(callback) else callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode event data: {str(e)}")
        except ValidationError as e:
            self.logger.error(f"Invalid event data: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error processing event: {str(e)}")