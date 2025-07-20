import unittest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.main import main


class TestMain(unittest.TestCase):
    @patch("builtins.print")
    def test_main(self, mock_print: MagicMock) -> None:
        main()
        mock_print.assert_called_once_with("project template")


if __name__ == "__main__":
    unittest.main()
