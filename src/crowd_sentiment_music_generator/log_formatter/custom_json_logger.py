import atexit
import datetime as dt
import json
import logging
import logging.config
import pathlib
from logging.handlers import QueueHandler
from typing import cast, override

import yaml

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    """Custom JSON log formatter.

    This formatter converts log records to JSON format with customizable keys.

    Attributes:
        fmt_keys (dict[str, str]): A dictionary mapping custom keys to log record attributes.
    """

    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        """Initializes the JSONFormatter with optional custom keys.

        Args:
            fmt_keys (dict[str, str], optional): A dictionary mapping custom keys to log record attributes. Defaults to None.
        """
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record as a JSON string.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record as a JSON string.
        """
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict[str, str]:
        """Prepares a dictionary of log record attributes.

        Args:
            record (logging.LogRecord): The log record to process.

        Returns:
            dict: A dictionary of log record attributes.
        """
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val if (msg_val := always_fields.pop(val, None)) is not None else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    """Filter to allow only non-error log records.

    This filter allows log records with a level number less than or equal to INFO.

    Methods:
        filter(record: logging.LogRecord) -> bool | logging.LogRecord: Determines if the log record should be logged.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        """Determines if the log record should be logged.

        Args:
            record (logging.LogRecord): The log record to filter.

        Returns:
            bool | logging.LogRecord: True if the log record level is less than or equal to INFO, otherwise False.
        """
        return record.levelno <= logging.INFO


def setup_logging() -> None:
    """Sets up the logging configuration.

    This function creates the logs folder if it does not exist, loads the logging configuration
    from a YAML file, and starts the queue handler listener if it is available.
    """
    logs_folder = pathlib.Path("./logs")
    logs_folder.mkdir(exist_ok=True)
    config_file = pathlib.Path("./config/log_config.yaml")
    with open(config_file) as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler and isinstance(queue_handler, QueueHandler):
        queue_handler = cast(QueueHandler, queue_handler)
        if queue_handler.listener:
            queue_handler.listener.start()
            atexit.register(queue_handler.listener.stop)
