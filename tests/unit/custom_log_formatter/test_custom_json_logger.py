import json
import logging
import unittest
from logging.handlers import QueueHandler
from pathlib import Path
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.log_formatter.custom_json_logger import (
    JSONFormatter,
    NonErrorFilter,
    setup_logging,
)


class TestJSONFormatter(unittest.TestCase):
    def test_format_with_custom_keys(self) -> None:
        formatter = JSONFormatter(fmt_keys={"custom_message": "message", "custom_level": "levelname"})
        log_record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_path",
            lineno=10,
            msg="Test log message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(log_record)
        log_dict = json.loads(formatted)
        self.assertIn("custom_message", log_dict)
        self.assertIn("custom_level", log_dict)
        self.assertEqual(log_dict["custom_message"], "Test log message")
        self.assertEqual(log_dict["custom_level"], "INFO")
        self.assertIn("timestamp", log_dict)

    def test_format_with_exception(self) -> None:
        formatter = JSONFormatter()
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            log_record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test_path",
                lineno=10,
                msg="Test log message",
                args=(),
                exc_info=(type(e), e, e.__traceback__),
            )
        formatted = formatter.format(log_record)
        log_dict = json.loads(formatted)
        self.assertIn("exc_info", log_dict)

    def test_format_with_stack_info(self) -> None:
        formatter = JSONFormatter()
        log_record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test_path",
            lineno=10,
            msg="Test log message",
            args=(),
            exc_info=None,
            sinfo="Stack info",
        )
        formatted = formatter.format(log_record)
        log_dict = json.loads(formatted)
        self.assertIn("stack_info", log_dict)
        self.assertEqual(log_dict["stack_info"], "Stack info")

    def test_format_with_default_keys(self) -> None:
        formatter = JSONFormatter()
        log_record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_path",
            lineno=10,
            msg="Test log message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(log_record)
        log_dict = json.loads(formatted)
        self.assertIn("message", log_dict)
        self.assertEqual(log_dict["message"], "Test log message")
        self.assertIn("timestamp", log_dict)


class TestNonErrorFilter(unittest.TestCase):
    def test_filter_allows_info(self) -> None:
        log_filter = NonErrorFilter()
        log_record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_path",
            lineno=10,
            msg="Test log message",
            args=(),
            exc_info=None,
        )
        self.assertTrue(log_filter.filter(log_record))

    def test_filter_blocks_error(self) -> None:
        log_filter = NonErrorFilter()
        log_record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test_path",
            lineno=10,
            msg="Test log message",
            args=(),
            exc_info=None,
        )
        self.assertFalse(log_filter.filter(log_record))

    def test_filter_allows_debug(self) -> None:
        log_filter = NonErrorFilter()
        log_record = logging.LogRecord(
            name="test_logger",
            level=logging.DEBUG,
            pathname="test_path",
            lineno=10,
            msg="Test log message",
            args=(),
            exc_info=None,
        )
        self.assertTrue(log_filter.filter(log_record))


class TestSetupLogging(unittest.TestCase):
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open")
    @patch("crowd_sentiment_music_generator.log_formatter.custom_json_logger.yaml.safe_load")
    @patch("crowd_sentiment_music_generator.log_formatter.custom_json_logger.logging.config.dictConfig")
    @patch("crowd_sentiment_music_generator.log_formatter.custom_json_logger.logging.getLogger")
    def test_setup_logging(self, mock_get_logger, mock_dict_config, mock_safe_load, mock_open, mock_mkdir) -> None:  # type: ignore
        mock_safe_load.return_value = {}
        mock_queue_handler = MagicMock(spec=QueueHandler)
        mock_queue_handler.listener = MagicMock()
        mock_logger = MagicMock()
        mock_logger.getEffectiveLevel.return_value = logging.INFO
        mock_logger.handlers = [mock_queue_handler]
        mock_get_logger.return_value = mock_logger

        mock_mkdir.return_value = None

        setup_logging()

        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_open.assert_called_once_with(Path("./config/log_config.yaml"))
        mock_dict_config.assert_called_once_with({})
        self.assertFalse(mock_queue_handler.listener.start.called, "Listener start method was not called.")

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open")
    @patch("crowd_sentiment_music_generator.log_formatter.custom_json_logger.yaml.safe_load")
    @patch("crowd_sentiment_music_generator.log_formatter.custom_json_logger.logging.config.dictConfig")
    @patch("crowd_sentiment_music_generator.log_formatter.custom_json_logger.logging.getLogger")
    def test_setup_logging_no_queue_handler(
        self,
        mock_get_logger: MagicMock,
        mock_dict_config: MagicMock,
        mock_safe_load: MagicMock,
        mock_open: MagicMock,
        mock_mkdir: MagicMock,
    ) -> None:
        mock_safe_load.return_value = {}
        mock_logger = MagicMock()
        mock_logger.getEffectiveLevel.return_value = logging.INFO
        mock_logger.handlers = []
        mock_get_logger.return_value = mock_logger

        mock_mkdir.return_value = None

        setup_logging()

        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_open.assert_called_once_with(Path("./config/log_config.yaml"))
        mock_dict_config.assert_called_once_with({})


if __name__ == "__main__":
    unittest.main()
