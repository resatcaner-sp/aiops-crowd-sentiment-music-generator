# Auto-Scaling Infrastructure

This document describes the auto-scaling infrastructure for the Crowd Sentiment Music Generator system.

## Overview

The auto-scaling infrastructure enables the system to handle multiple simultaneous matches and scale efficiently based on demand. It includes:

1. **Container-based deployment** using Docker and Kubernetes
2. **Scaling policies** based on CPU, memory, and request rate metrics
3. **Load balancing** for distributed processing
4. **Resource monitoring** for tracking system performance

## Components

### 1. Container-based Deployment

The application is containerized using Docker, which provides:

- Consistent environment across development, testing, and production
- Isolation of dependencies and runtime environment
- Easy deployment and scaling

#### Docker Setup

- `Dockerfile`: Defines the container image
- `docker-compose.yml`: Defines local development environment

#### Kubernetes Setup

- `k8s/namespace.yaml`: Defines the Kubernetes namespace
- `k8s/deployment.yaml`: Defines the deployment configuration
- `k8s/service.yaml`: Defines the service for load balancing
- `k8s/hpa.yaml`: Defines the Horizontal Pod Autoscaler

### 2. Scaling Policies

The system supports multiple scaling policies:

- **CPU-based scaling**: Scales based on CPU utilization
- **Memory-based scaling**: Scales based on memory utilization
- **Request rate-based scaling**: Scales based on incoming request rate
- **Composite scaling**: Combines multiple policies

Scaling decisions include:

- When to scale (based on thresholds)
- How much to scale (based on scaling factors)
- Cooldown periods to prevent oscillation

### 3. Load Balancing

The system includes a load balancer that distributes requests across instances using different strategies:

- **Round Robin**: Distributes requests evenly across instances
- **Least Connections**: Sends requests to the instance with the fewest active connections
- **IP Hash**: Ensures requests from the same client go to the same instance

### 4. Resource Monitoring

The system monitors resource usage to inform scaling decisions:

- **CPU usage**: Percentage of CPU utilized
- **Memory usage**: Percentage of memory utilized
- **Disk I/O**: Read and write operations
- **Request rate**: Number of requests per second

## Configuration

### Scaling Configuration

Scaling behavior can be configured through environment variables or the API:

```yaml
# Environment variables
MIN_INSTANCES=2
MAX_INSTANCES=10
CPU_THRESHOLD=70
MEMORY_THRESHOLD=80
SCALING_COOLDOWN=300
LOAD_BALANCER_STRATEGY=round_robin
```

### Match Prioritization

The system supports prioritizing critical matches when resources are constrained:

```json
{
  "match_id": "match-123",
  "priority": 8,
  "resource_allocation": 30.0
}
```

## API Endpoints

The system provides API endpoints for monitoring and managing the auto-scaling infrastructure:

### Get Scaling Status

```
GET /scaling/status
```

Returns the current scaling status, including:
- Current number of instances
- Min/max instance limits
- Thresholds
- Current metrics

### Get Scaling Decision

```
GET /scaling/decision
```

Returns the current scaling decision, including:
- Whether scaling is needed
- Target number of instances
- Reason for scaling

### Manual Scaling

```
POST /scaling/scale
```

Manually scale the application to a specific number of instances.

Request body:
```json
{
  "target_instances": 5,
  "reason": "Handling major tournament"
}
```

### Get/Update Scaling Configuration

```
GET /scaling/config
PUT /scaling/config
```

Get or update the scaling configuration.

### Manage Match Priorities

```
GET /scaling/priorities
POST /scaling/priorities
DELETE /scaling/priorities/{match_id}
```

Manage match priorities for resource allocation.

## Deployment

### Local Development

```bash
# Build and run locally
docker-compose up --build
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

## Testing

The system includes tests for the auto-scaling infrastructure:

- **Unit tests**: Test individual components
- **Integration tests**: Test interactions between components
- **Load tests**: Test scaling under load

Run tests with:

```bash
pytest tests/unit/utils/test_auto_scaling.py
pytest tests/integration/test_auto_scaling.py
```

## Monitoring

Monitor the auto-scaling infrastructure using:

- Kubernetes Dashboard
- API endpoints
- Logging

Example log output:
```
INFO:crowd_sentiment_music_generator.utils.auto_scaling:Scaling from 2 to 3 instances: Scaling up based on cpu-based-scaling
```