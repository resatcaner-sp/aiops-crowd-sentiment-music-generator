# Crowd Sentiment Music Generator

#### Author : Caner Bas <caner.bas@statsperform.com>
#### Version : 0.1.0
#### Python : 3.9 - 3.10

## Description

The Crowd Sentiment Music Generator is an AI-powered system that analyzes live crowd noise from sports events and generates real-time musical compositions reflecting the emotional journey of the match. The system operates in two primary modes:

1. **Live Mode**: Real-time analysis and music generation during live broadcasts
2. **Highlight Mode**: Post-processing for creating musical soundtracks for match highlights

## Features

- Crowd audio analysis with emotion classification
- Real-time music generation based on crowd emotions
- Event synchronization with match data
- Highlight music generation for post-match content
- User preference customization
- Web-based control interface

## Installation

### Prerequisites

- Python 3.9 - 3.10
- Poetry for dependency management

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/crowd-sentiment-music-generator.git
   cd crowd-sentiment-music-generator
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Usage

### Running the API Server

```bash
python -m src.crowd_sentiment_music_generator.main
```

The API server will start on http://localhost:8000 by default.

### API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Running Tests

```bash
pytest
```

For test coverage:
```bash
pytest --cov=src
```

## Project Structure

```
src/
├── crowd_sentiment_music_generator/
│   ├── api/                    # FastAPI application and routers
│   ├── models/                 # Data models
│   │   ├── data/               # Data-related models
│   │   └── music/              # Music-related models
│   ├── services/               # Business logic services
│   │   ├── crowd_analysis/     # Crowd audio analysis
│   │   ├── data_ingestion/     # Data ingestion from external sources
│   │   ├── highlight_generator/# Highlight music generation
│   │   ├── music_engine/       # Music generation engine
│   │   ├── music_trigger/      # Music trigger based on events
│   │   └── sync_engine/        # Event synchronization
│   ├── utils/                  # Utility functions
│   └── exceptions/             # Custom exceptions
tests/
├── unit/                       # Unit tests
│   ├── api/                    # API tests
│   ├── models/                 # Model tests
│   ├── services/               # Service tests
│   └── utils/                  # Utility tests
└── integration/                # Integration tests
```

## License

Proprietary - All rights reserved
