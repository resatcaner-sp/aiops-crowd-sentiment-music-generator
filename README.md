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

#### Option 1: Direct Python Execution

```bash
python -m src.crowd_sentiment_music_generator.main
```

The API server will start on http://localhost:8000 by default.

#### Option 2: Docker Compose (Recommended)

The project includes Docker Compose configuration for easy deployment with all dependencies.

1. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

2. For running in detached mode:
   ```bash
   docker-compose up -d
   ```

3. To stop the containers:
   ```bash
   docker-compose down
   ```

4. To view logs:
   ```bash
   docker-compose logs -f
   ```

5. To scale the application (for testing auto-scaling):
   ```bash
   docker-compose up -d --scale app=3
   ```

### Environment Variables

The following environment variables can be set in your `.env` file or directly in the Docker Compose configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Host to bind the server | `0.0.0.0` |
| `PORT` | Port to bind the server | `8000` |
| `REDIS_HOST` | Redis host for caching | `redis` |
| `REDIS_PORT` | Redis port | `6379` |
| `CACHE_ENABLED` | Enable or disable caching | `true` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MIN_INSTANCES` | Minimum number of instances for auto-scaling | `2` |
| `MAX_INSTANCES` | Maximum number of instances for auto-scaling | `10` |
| `CPU_THRESHOLD` | CPU threshold for auto-scaling | `70` |
| `MEMORY_THRESHOLD` | Memory threshold for auto-scaling | `80` |

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

## Auto-Scaling Infrastructure

The Crowd Sentiment Music Generator includes a robust auto-scaling infrastructure to handle multiple simultaneous matches and scale efficiently based on demand.

### Key Components

1. **Container-based Deployment**
   - Docker containers for consistent environments
   - Kubernetes orchestration for auto-scaling and self-healing

2. **Scaling Policies**
   - CPU-based scaling: Scales based on CPU utilization
   - Memory-based scaling: Scales based on memory utilization
   - Request rate-based scaling: Scales based on incoming request rate

3. **Load Balancing**
   - Multiple load balancing strategies (round robin, least connections, IP hash)
   - Automatic distribution of requests across instances

4. **Resource Monitoring**
   - Real-time monitoring of CPU, memory, and disk usage
   - Performance metrics tracking for optimization

### Kubernetes Deployment

To deploy the application to Kubernetes:

1. Apply the Kubernetes manifests:
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/hpa.yaml
   ```

2. Check the deployment status:
   ```bash
   kubectl get pods -n crowd-sentiment-music-generator
   ```

3. Access the service:
   ```bash
   kubectl port-forward svc/crowd-sentiment-music-generator -n crowd-sentiment-music-generator 8000:80
   ```

For more details on the auto-scaling infrastructure, see [docs/auto_scaling.md](docs/auto_scaling.md).

## Project Structure

```
src/
├── crowd_sentiment_music_generator/
│   ├── api/                    # FastAPI application and routers
│   │   └── routers/            # API endpoint routers
│   ├── models/                 # Data models
│   │   ├── data/               # Data-related models
│   │   └── music/              # Music-related models
│   ├── services/               # Business logic services
│   │   ├── audio_processing/   # Audio processing services
│   │   ├── crowd_analysis/     # Crowd audio analysis
│   │   ├── data_ingestion/     # Data ingestion from external sources
│   │   ├── highlight_generator/# Highlight music generation
│   │   ├── music_engine/       # Music generation engine
│   │   ├── music_trigger/      # Music trigger based on events
│   │   └── sync_engine/        # Event synchronization
│   ├── utils/                  # Utility functions
│   │   ├── auto_scaling.py     # Auto-scaling utilities
│   │   ├── cache.py            # Caching utilities
│   │   ├── parallel_processing.py # Parallel processing utilities
│   │   └── resource_monitoring.py # Resource monitoring utilities
│   └── exceptions/             # Custom exceptions
k8s/                            # Kubernetes manifests
├── deployment.yaml             # Deployment configuration
├── hpa.yaml                    # Horizontal Pod Autoscaler
├── namespace.yaml              # Namespace definition
└── service.yaml                # Service for load balancing
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
