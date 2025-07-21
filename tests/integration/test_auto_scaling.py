"""Integration tests for auto-scaling infrastructure.

This module contains integration tests for the auto-scaling infrastructure,
including container deployment, scaling policies, and load balancing.
"""

import os
import time
from typing import Dict, List, Optional

import pytest
import requests
from fastapi.testclient import TestClient

from crowd_sentiment_music_generator.api.app import create_app
from crowd_sentiment_music_generator.utils.auto_scaling import (
    CPUBasedScalingPolicy,
    LoadBalancer,
    MemoryBasedScalingPolicy,
    RequestRateScalingPolicy,
    ScalingManager,
)


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)


@pytest.mark.integration
def test_health_endpoint(test_client):
    """Test the health endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_scaling_policies():
    """Test scaling policies with simulated metrics."""
    # CPU-based policy
    cpu_policy = CPUBasedScalingPolicy(threshold=70.0)
    
    # Test scaling up
    metrics = {"cpu_percent": 85.0}
    assert cpu_policy.should_scale(metrics) is True
    assert cpu_policy.get_scale_factor(metrics) > 1.0
    
    # Test no scaling
    metrics = {"cpu_percent": 60.0}
    assert cpu_policy.should_scale(metrics) is False
    assert cpu_policy.get_scale_factor(metrics) == 1.0
    
    # Test scaling down
    metrics = {"cpu_percent": 30.0}
    assert cpu_policy.should_scale(metrics) is True
    assert cpu_policy.get_scale_factor(metrics) < 1.0
    
    # Memory-based policy
    memory_policy = MemoryBasedScalingPolicy(threshold=80.0)
    
    # Test scaling up
    metrics = {"memory_percent": 90.0}
    assert memory_policy.should_scale(metrics) is True
    assert memory_policy.get_scale_factor(metrics) > 1.0
    
    # Test no scaling
    metrics = {"memory_percent": 70.0}
    assert memory_policy.should_scale(metrics) is False
    assert memory_policy.get_scale_factor(metrics) == 1.0
    
    # Test scaling down
    metrics = {"memory_percent": 30.0}
    assert memory_policy.should_scale(metrics) is True
    assert memory_policy.get_scale_factor(metrics) < 1.0
    
    # Request rate policy
    request_policy = RequestRateScalingPolicy(threshold=100.0)
    
    # Test scaling up
    metrics = {"request_rate": 150.0}
    assert request_policy.should_scale(metrics) is True
    assert request_policy.get_scale_factor(metrics) > 1.0
    
    # Test no scaling
    metrics = {"request_rate": 80.0}
    assert request_policy.should_scale(metrics) is False
    assert request_policy.get_scale_factor(metrics) == 1.0
    
    # Test scaling down
    metrics = {"request_rate": 40.0}
    assert request_policy.should_scale(metrics) is True
    assert request_policy.get_scale_factor(metrics) < 1.0


@pytest.mark.integration
def test_scaling_manager():
    """Test scaling manager with simulated metrics."""
    # Create scaling manager with test policies
    manager = ScalingManager(
        policy=CPUBasedScalingPolicy(threshold=70.0),
        min_instances=2,
        max_instances=10,
        cooldown_period=0,  # No cooldown for testing
    )
    
    # Test scaling up
    manager.monitor.get_current_usage = lambda: {"cpu_percent": 85.0, "memory_percent": 70.0}
    decision = manager.check_scaling()
    assert decision["should_scale"] is True
    assert decision["target_instances"] > manager.current_instances
    
    # Apply scaling
    assert manager.apply_scaling(decision) is True
    previous_instances = manager.current_instances
    
    # Test scaling down
    manager.monitor.get_current_usage = lambda: {"cpu_percent": 30.0, "memory_percent": 40.0}
    decision = manager.check_scaling()
    assert decision["should_scale"] is True
    assert decision["target_instances"] < previous_instances
    
    # Apply scaling
    assert manager.apply_scaling(decision) is True
    
    # Test no scaling needed
    manager.monitor.get_current_usage = lambda: {"cpu_percent": 60.0, "memory_percent": 60.0}
    decision = manager.check_scaling()
    assert decision["should_scale"] is False


@pytest.mark.integration
def test_load_balancer():
    """Test load balancer with simulated instances."""
    # Create load balancer
    lb = LoadBalancer(strategy="round_robin")
    
    # Register instances
    lb.register_instance({"id": "instance-1", "host": "host-1", "port": 8000})
    lb.register_instance({"id": "instance-2", "host": "host-2", "port": 8000})
    lb.register_instance({"id": "instance-3", "host": "host-3", "port": 8000})
    
    # Test round robin
    instance1 = lb.get_instance()
    instance2 = lb.get_instance()
    instance3 = lb.get_instance()
    instance4 = lb.get_instance()
    
    assert instance1["id"] == "instance-1"
    assert instance2["id"] == "instance-2"
    assert instance3["id"] == "instance-3"
    assert instance4["id"] == "instance-1"  # Back to first instance
    
    # Test deregistration
    lb.deregister_instance("instance-2")
    
    instance1 = lb.get_instance()
    instance2 = lb.get_instance()
    
    assert instance1["id"] == "instance-1"
    assert instance2["id"] == "instance-3"  # Skips deregistered instance
    
    # Test least connections
    lb = LoadBalancer(strategy="least_connections")
    
    lb.register_instance({"id": "instance-1", "host": "host-1", "port": 8000})
    lb.register_instance({"id": "instance-2", "host": "host-2", "port": 8000})
    
    # First request goes to either instance
    instance1 = lb.get_instance()
    
    # Add connections to instance-1
    lb.connections[instance1["id"]] = 5
    
    # Next request should go to the instance with fewer connections
    instance2 = lb.get_instance()
    assert instance1["id"] != instance2["id"]
    
    # Release connection
    lb.release_instance(instance1["id"])
    assert lb.connections[instance1["id"]] == 4


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUN_DOCKER_TESTS"),
    reason="Docker tests are disabled. Set RUN_DOCKER_TESTS=1 to enable.",
)
def test_docker_container():
    """Test Docker container deployment and scaling.
    
    This test requires Docker to be installed and running.
    It will build and run a Docker container for the application.
    """
    import docker
    
    # Create Docker client
    client = docker.from_env()
    
    # Build image
    image, logs = client.images.build(
        path=".",
        tag="crowd-sentiment-music-generator:test",
        rm=True,
    )
    
    # Run container
    container = client.containers.run(
        "crowd-sentiment-music-generator:test",
        detach=True,
        ports={"8000/tcp": 8000},
        environment={
            "HOST": "0.0.0.0",
            "PORT": "8000",
        },
    )
    
    try:
        # Wait for container to start
        time.sleep(5)
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        
    finally:
        # Clean up
        container.stop()
        container.remove()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("RUN_K8S_TESTS"),
    reason="Kubernetes tests are disabled. Set RUN_K8S_TESTS=1 to enable.",
)
def test_kubernetes_deployment():
    """Test Kubernetes deployment and scaling.
    
    This test requires kubectl to be installed and configured.
    It will deploy the application to a Kubernetes cluster.
    """
    import subprocess
    
    # Create namespace
    subprocess.run(["kubectl", "apply", "-f", "k8s/namespace.yaml"], check=True)
    
    try:
        # Deploy application
        subprocess.run(["kubectl", "apply", "-f", "k8s/deployment.yaml"], check=True)
        subprocess.run(["kubectl", "apply", "-f", "k8s/service.yaml"], check=True)
        
        # Wait for deployment to be ready
        subprocess.run(
            [
                "kubectl",
                "wait",
                "--namespace=crowd-sentiment-music-generator",
                "--for=condition=available",
                "deployment/crowd-sentiment-music-generator",
                "--timeout=60s",
            ],
            check=True,
        )
        
        # Test scaling
        subprocess.run(
            [
                "kubectl",
                "scale",
                "--namespace=crowd-sentiment-music-generator",
                "deployment/crowd-sentiment-music-generator",
                "--replicas=3",
            ],
            check=True,
        )
        
        # Wait for scaling to complete
        time.sleep(5)
        
        # Check number of replicas
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "deployment",
                "crowd-sentiment-music-generator",
                "--namespace=crowd-sentiment-music-generator",
                "-o=jsonpath='{.status.replicas}'",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        
        assert result.stdout.strip("'") == "3"
        
    finally:
        # Clean up
        subprocess.run(["kubectl", "delete", "-f", "k8s/deployment.yaml"], check=False)
        subprocess.run(["kubectl", "delete", "-f", "k8s/service.yaml"], check=False)
        subprocess.run(["kubectl", "delete", "-f", "k8s/namespace.yaml"], check=False)


@pytest.mark.integration
def test_load_distribution(test_client):
    """Test load distribution across multiple instances."""
    # Create a mock load balancer
    lb = LoadBalancer(strategy="round_robin")
    
    # Register mock instances
    lb.register_instance({"id": "instance-1", "host": "host-1", "port": 8000})
    lb.register_instance({"id": "instance-2", "host": "host-2", "port": 8000})
    lb.register_instance({"id": "instance-3", "host": "host-3", "port": 8000})
    
    # Track request distribution
    instance_counts = {"instance-1": 0, "instance-2": 0, "instance-3": 0}
    
    # Simulate multiple requests
    for _ in range(100):
        instance = lb.get_instance()
        instance_counts[instance["id"]] += 1
    
    # Check that requests are distributed evenly
    assert instance_counts["instance-1"] > 0
    assert instance_counts["instance-2"] > 0
    assert instance_counts["instance-3"] > 0
    
    # Check that the distribution is roughly even (within 10%)
    avg_requests = sum(instance_counts.values()) / len(instance_counts)
    for count in instance_counts.values():
        assert abs(count - avg_requests) / avg_requests < 0.1