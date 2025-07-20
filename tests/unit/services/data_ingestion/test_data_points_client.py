"""Unit tests for data points client."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from aiohttp import WSMsgType

from crowd_sentiment_music_generator.exceptions.data_api_error import DataAPIError
from crowd_sentiment_music_generator.models.data.match_event import MatchEvent
from crowd_sentiment_music_generator.services.data_ingestion.data_points_client import DataPointsClient


class TestDataPointsClient:
    """Test cases for DataPointsClient class."""
    
    @pytest.fixture
    def client(self) -> DataPointsClient:
        """Create a DataPointsClient instance for testing."""
        return DataPointsClient(base_url="https://api.example.com", api_key="test_key")
    
    @pytest.fixture
    def sample_match_event_dict(self) -> dict:
        """Create a sample match event dictionary."""
        return {
            "id": "123",
            "type": "goal",
            "timestamp": 1625097600.0,
            "team_id": "team1",
            "player_id": "player1",
            "position": {"x": 10.5, "y": 20.3},
            "additional_data": {"speed": 25.6, "angle": 45.0},
        }
    
    @pytest.fixture
    def sample_match_state(self) -> dict:
        """Create a sample match state dictionary."""
        return {
            "match_id": "123",
            "home_team": {
                "id": "team1",
                "name": "Home Team",
                "score": 2
            },
            "away_team": {
                "id": "team2",
                "name": "Away Team",
                "score": 1
            },
            "match_time": 75.5,
            "period": 2,
            "status": "in_progress"
        }
    
    @pytest.mark.asyncio
    async def test_initialization(self, client: DataPointsClient) -> None:
        """Test client initialization."""
        assert client.base_url == "https://api.example.com"
        assert client.api_key == "test_key"
        assert client.session is None
        assert client.ws_connection is None
        assert client.match_id is None
        assert not client.is_connected
        assert client.event_callbacks == []
    
    @pytest.mark.asyncio
    async def test_connect_success(self, client: DataPointsClient) -> None:
        """Test successful connection."""
        # Mock session and ws_connect
        mock_session = AsyncMock()
        mock_ws = AsyncMock()
        mock_session.ws_connect.return_value = mock_ws
        
        # Patch aiohttp.ClientSession to return our mock
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Connect to API
            await client.connect("match123")
            
            # Verify session was created
            assert client.session == mock_session
            
            # Verify ws_connect was called with correct URL
            mock_session.ws_connect.assert_called_once_with(
                "https://api.example.com/ws/matches/match123/events?api_key=test_key"
            )
            
            # Verify connection state
            assert client.is_connected
            assert client.match_id == "match123"
            assert client.ws_connection == mock_ws
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, client: DataPointsClient) -> None:
        """Test connection failure."""
        # Mock session that raises an exception
        mock_session = AsyncMock()
        mock_session.ws_connect.side_effect = aiohttp.ClientError("Connection failed")
        
        # Patch aiohttp.ClientSession to return our mock
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Connect to API should raise an exception
            with pytest.raises(DataAPIError) as excinfo:
                await client.connect("match123")
            
            # Verify error message
            assert "Failed to connect to data points API" in str(excinfo.value)
            
            # Verify connection state
            assert not client.is_connected
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client: DataPointsClient) -> None:
        """Test disconnection."""
        # Set up client with mock connection
        mock_ws = AsyncMock()
        mock_session = AsyncMock()
        client.ws_connection = mock_ws
        client.session = mock_session
        client.is_connected = True
        
        # Disconnect
        await client.disconnect()
        
        # Verify ws_connection was closed
        mock_ws.close.assert_called_once()
        
        # Verify session was closed
        mock_session.close.assert_called_once()
        
        # Verify connection state
        assert not client.is_connected
        assert client.ws_connection is None
        assert client.session is None
    
    @pytest.mark.asyncio
    async def test_disconnect_with_error(self, client: DataPointsClient) -> None:
        """Test disconnection with error."""
        # Set up client with mock connection that raises an exception
        mock_ws = AsyncMock()
        mock_ws.close.side_effect = Exception("Close failed")
        mock_session = AsyncMock()
        client.ws_connection = mock_ws
        client.session = mock_session
        client.is_connected = True
        
        # Disconnect should raise an exception
        with pytest.raises(DataAPIError) as excinfo:
            await client.disconnect()
        
        # Verify error message
        assert "Failed to disconnect from data points API" in str(excinfo.value)
        
        # Verify connection state
        assert not client.is_connected
        assert client.ws_connection is None
    
    @pytest.mark.asyncio
    async def test_subscribe_to_events(self, client: DataPointsClient) -> None:
        """Test subscribing to events."""
        # Set up client as connected
        client.is_connected = True
        
        # Create mock callback
        callback = MagicMock()
        
        # Subscribe to events
        await client.subscribe_to_events(callback)
        
        # Verify callback was added
        assert len(client.event_callbacks) == 1
        assert client.event_callbacks[0] == callback
    
    @pytest.mark.asyncio
    async def test_subscribe_to_events_not_connected(self, client: DataPointsClient) -> None:
        """Test subscribing to events when not connected."""
        # Create mock callback
        callback = MagicMock()
        
        # Subscribe to events should raise an exception
        with pytest.raises(DataAPIError) as excinfo:
            await client.subscribe_to_events(callback)
        
        # Verify error message
        assert "Not connected to data points API" in str(excinfo.value)
        
        # Verify no callback was added
        assert len(client.event_callbacks) == 0
    
    @pytest.mark.asyncio
    async def test_get_latest_match_state_success(
        self, client: DataPointsClient, sample_match_state: dict
    ) -> None:
        """Test getting latest match state successfully."""
        # Set up client with match ID
        client.match_id = "match123"
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = sample_match_state
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session
        
        # Get match state
        result = await client.get_latest_match_state()
        
        # Verify session.get was called with correct URL
        mock_session.get.assert_called_once_with(
            "https://api.example.com/api/matches/match123/state?api_key=test_key"
        )
        
        # Verify result
        assert result == sample_match_state
    
    @pytest.mark.asyncio
    async def test_get_latest_match_state_error(self, client: DataPointsClient) -> None:
        """Test getting latest match state with error."""
        # Set up client with match ID
        client.match_id = "match123"
        
        # Mock response with error
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text.return_value = "Match not found"
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session
        
        # Get match state should raise an exception
        with pytest.raises(DataAPIError) as excinfo:
            await client.get_latest_match_state()
        
        # Verify error message
        assert "Failed to get match state" in str(excinfo.value)
        assert excinfo.value.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_latest_match_state_no_match_id(self, client: DataPointsClient) -> None:
        """Test getting latest match state with no match ID."""
        # Get match state should raise an exception
        with pytest.raises(DataAPIError) as excinfo:
            await client.get_latest_match_state()
        
        # Verify error message
        assert "No match ID specified" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_process_event(
        self, client: DataPointsClient, sample_match_event_dict: dict
    ) -> None:
        """Test processing an event."""
        # Create mock callbacks
        sync_callback = MagicMock()
        async_callback = AsyncMock()
        client.event_callbacks = [sync_callback, async_callback]
        
        # Process event
        await client._process_event(json.dumps(sample_match_event_dict))
        
        # Verify callbacks were called
        sync_callback.assert_called_once()
        async_callback.assert_called_once()
        
        # Verify event object
        event_arg = sync_callback.call_args[0][0]
        assert isinstance(event_arg, MatchEvent)
        assert event_arg.id == sample_match_event_dict["id"]
        assert event_arg.type == sample_match_event_dict["type"]
    
    @pytest.mark.asyncio
    async def test_process_event_invalid_json(self, client: DataPointsClient) -> None:
        """Test processing an event with invalid JSON."""
        # Create mock callback
        callback = MagicMock()
        client.event_callbacks = [callback]
        
        # Process invalid event
        await client._process_event("invalid json")
        
        # Verify callback was not called
        callback.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_event_invalid_data(self, client: DataPointsClient) -> None:
        """Test processing an event with invalid data."""
        # Create mock callback
        callback = MagicMock()
        client.event_callbacks = [callback]
        
        # Process event with missing required fields
        await client._process_event(json.dumps({"id": "123"}))
        
        # Verify callback was not called
        callback.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_listen_for_events(self, client: DataPointsClient, sample_match_event_dict: dict) -> None:
        """Test listening for events."""
        # Mock WebSocket connection
        mock_ws = AsyncMock()
        
        # Create messages
        text_msg = MagicMock()
        text_msg.type = WSMsgType.TEXT
        text_msg.data = json.dumps(sample_match_event_dict)
        
        error_msg = MagicMock()
        error_msg.type = WSMsgType.ERROR
        
        # Configure mock to yield messages then stop
        mock_ws.__aiter__.return_value = [text_msg, error_msg]
        
        # Set up client
        client.ws_connection = mock_ws
        client.is_connected = True
        
        # Mock _process_event
        with patch.object(client, "_process_event") as mock_process:
            # Listen for events
            await client._listen_for_events()
            
            # Verify _process_event was called
            mock_process.assert_called_once_with(json.dumps(sample_match_event_dict))
            
            # Verify connection state
            assert not client.is_connected
    
    @pytest.mark.asyncio
    async def test_reconnect_success(self, client: DataPointsClient) -> None:
        """Test successful reconnection."""
        # Set up client
        client.match_id = "match123"
        
        # Mock connect method
        with patch.object(client, "connect") as mock_connect:
            # Reconnect
            await client._reconnect()
            
            # Verify connect was called
            mock_connect.assert_called_once_with("match123")
    
    @pytest.mark.asyncio
    async def test_reconnect_failure(self, client: DataPointsClient) -> None:
        """Test reconnection failure."""
        # Set up client
        client.match_id = "match123"
        client.max_retries = 2
        client.retry_interval = 0.01  # Short interval for testing
        
        # Mock connect method to always fail
        with patch.object(client, "connect") as mock_connect:
            mock_connect.side_effect = DataAPIError("Connection failed")
            
            # Reconnect
            await client._reconnect()
            
            # Verify connect was called twice (max_retries)
            assert mock_connect.call_count == 2
            mock_connect.assert_called_with("match123")
    
    @pytest.mark.asyncio
    async def test_reconnect_no_match_id(self, client: DataPointsClient) -> None:
        """Test reconnection with no match ID."""
        # Mock connect method
        with patch.object(client, "connect") as mock_connect:
            # Reconnect
            await client._reconnect()
            
            # Verify connect was not called
            mock_connect.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client: DataPointsClient) -> None:
        """Test that error handling decorator is applied to public methods."""
        # Verify error handling is applied to public methods
        assert hasattr(client.connect, "__wrapped__")
        assert hasattr(client.disconnect, "__wrapped__")
        assert hasattr(client.subscribe_to_events, "__wrapped__")
        assert hasattr(client.get_latest_match_state, "__wrapped__")