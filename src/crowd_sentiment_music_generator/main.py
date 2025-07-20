import logging

from crowd_sentiment_music_generator.log_formatter.custom_json_logger import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function of the script."""
    print("project template")


if __name__ == "__main__":
    setup_logging()
    main()
