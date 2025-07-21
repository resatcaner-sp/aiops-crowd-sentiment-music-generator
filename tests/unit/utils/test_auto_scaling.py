"""Unit tests for auto-scaling infrastructure utilities."""

import pytest
from unittest.mock import MagicMock, patch

from crowd_sentiment_music_generator.utils.auto_scaling import (
    CPUBasedScalingPolicy,
    LoadBalancer,
    MemoryBasedScalingPolicy,
    RequestRateScalingPolicy,
    ScalingManager,
    CompositeScalingPolicy,
    ContainerManager,
    get_container_manager,
    get_scaling_manager,
    get_load_balancer,
)


class TestScalingPolicies:
    """Test scaling policies."""
    
    def test_cpu_based_policy(self):
        """Test CPU-based scaling policy."""
        policy = CPUBasedScalingPolicy(threshold=70.0)
        
        # Test scaling up
        assert policy.should_scale({"cpu_percent": 80.0}) is True
        assert policy.get_scale_factor({"cpu_percent": 80.0}) > 1.0
        
        # Test no scaling
        assert policy.should_scale({"cpu_percent": 60.0}) is False
        assert policy.get_scale_factor({"cpu_percent": 60.0}) == 1.0
        
        # Test scaling down
        assert policy.should_scale({"cpu_percent": 30.0}) is True
        assert policy.get_scale_factor({"cpu_percent": 30.0}) < 1.0
        
        # Test missing metrics
        assert policy.should_scale({}) is False
        assert policy.get_scale_factor({}) == 1.0
    
    def test_memory_based_policy(self):
        """Test memory-based scaling policy."""
        policy = MemoryBasedScalingPolicy(threshold=80.0)
        
        # Test scaling up
        assert policy.should_scale({"memory_percent": 90.0}) is True
        assert policy.get_scale_factor({"memory_percent": 90.0}) > 1.0
        
        # Test no scaling
        assert policy.should_scale({"memory_percent": 70.0}) is False
        assert policy.get_scale_factor({"memory_percent": 70.0}) == 1.0
        
        # Test scaling down
        assert policy.should_scale({"memory_percent": 30.0}) is True
        assert policy.get_scale_factor({"memory_percent": 30.0}) < 1.0
        
        # Test missing metrics
        assert policy.should_scale({}) is False
        assert policy.get_scale_factor({}) == 1.0
    
    def test_request_rate_policy(self):
        """Test request rate-based scaling policy."""
        policy = RequestRateScalingPolicy(threshold=100.0)
        
        # Test scaling up
        assert policy.should_scale({"request_rate": 120.0}) is True
        assert policy.get_scale_factor({"request_rate": 120.0}) > 1.0
        
        # Test no scaling
        assert policy.should_scale({"request_rate": 80.0}) is False
        assert policy.get_scale_factor({"request_rate": 80.0}) == 1.0
        
        # Test scaling down
        assert policy.should_scale({"request_rate": 40.0}) is True
        assert policy.get_scale_factor({"request_rate": 40.0}) < 1.0
        
        # Test missing metrics
        assert policy.should_scale({}) is False
        assert policy.get_scale_factor({}) == 1.0
    
    def test_composite_policy(self):
        """Test composite scaling policy."""
        cpu_policy = CPUBasedScalingPolicy(threshold=70.0)
        memory_policy = MemoryBasedScalingPolicy(threshold=80.0)
        request_policy = RequestRateScalingPolicy(threshold=100.0)
        
        composite_policy = CompositeScalingPolicy([cpu_policy, memory_policy, request_policy])
        
        # Test scaling up (CPU only)
        metrics = {"cpu_percent": 80.0, "memory_percent": 70.0, "request_rate": 80.0}
        assert composite_policy.should_scale(metrics) is True
        assert composite_policy.get_scale_factor(metrics) > 1.0
        
        # Test scaling up (memory only)
        metrics = {"cpu_percent": 60.0, "memory_percent": 90.0, "request_rate": 80.0}
        assert composite_policy.should_scale(metrics) is True
        assert composite_policy.get_scale_factor(metrics) > 1.0
        
        # Test scaling up (request rate only)
        metrics = {"cpu_percent": 60.0, "memory_percent": 70.0, "request_rate": 120.0}
        assert composite_policy.should_scale(metrics) is True
        assert composite_policy.get_scale_factor(metrics) > 1.0
        
        # Test no scaling
        metrics = {"cpu_percent": 60.0, "memory_percent": 70.0, "request_rate": 80.0}
        assert composite_policy.should_scale(metrics) is False
        assert composite_policy.get_scale_factor(metrics) == 1.0
        
        # Test scaling down (all metrics)
        metrics = {"cpu_percent": 30.0, "memory_percent": 30.0, "request_rate": 40.0}
        assert composite_policy.should_scale(metrics) is True
        assert composite_policy.get_scale_factor(metrics) < 1.0
        
        # Test conflicting directions (prioritize scaling up)
        metrics = {"cpu_percent": 80.0, "memory_percent": 30.0, "request_rate": 80.0}
        assert composite_policy.should_scale(metrics) is True
        assert composite_policy.get_scale_factor(metrics) > 1.0


class TestScalingManager:
    """Test scaling manager."""
    
    def test_check_scaling(self):
        """Test check_scaling method."""
        # Create mock policy and monitor
        policy = MagicMock()
        monitor = MagicMock()
        
        # Configure mocks
        policy.should_scale.return_value = True
        policy.get_scale_factor.return_value = 1.5
        policy.name = "test-policy"
        monitor.get_current_usage.return_value = {"cpu_percent": 80.0}
        
        # Create scaling manager
        manager = ScalingManager(
            policy=policy,
            monitor=monitor,
            min_instances=2,
            max_instances=10,
            cooldown_period=0,  # No cooldown for testing
        )
        
        # Test scaling up
        decision = manager.check_scaling()
        assert decision["should_scale"] is True
        assert decision["target_instances"] == 3  # 2 * 1.5 = 3
        
        # Test applying scaling
        assert manager.apply_scaling(decision) is True
        assert manager.current_instances == 3
        
        # Test cooldown period
        manager.cooldown_period = 300
        manager.last_scale_time = 9999999999  # Far in the future
        
        decision = manager.check_scaling()
        assert decision["should_scale"] is False
        assert "cooldown" in decision["reason"].lower()
        
        # Test no scaling needed
        manager.cooldown_period = 0
        policy.should_scale.return_value = False
        
        decision = manager.check_scaling()
        assert decision["should_scale"] is False
        
        # Test max instances limit
        policy.should_scale.return_value = True
        policy.get_scale_factor.return_value = 10.0
        manager.current_instances = 5
        
        decision = manager.check_scaling()
        assert decision["should_scale"] is True
        assert decision["target_instances"] == 10  # Limited by max_instances
        
        # Test min instances limit
        policy.get_scale_factor.return_value = 0.1
        manager.current_instances = 3
        
        decision = manager.check_scaling()
        assert decision["should_scale"] is True
        assert decision["target_instances"] == 2  # Limited by min_instances


class TestLoadBalancer:
    """Test load balancer."""
    
    def test_round_robin(self):
        """Test round robin load balancing strategy."""
        lb = LoadBalancer(strategy="round_robin")
        
        # Register instances
        lb.register_instance({"id": "instance-1"})
        lb.register_instance({"id": "instance-2"})
        lb.register_instance({"id": "instance-3"})
        
        # Test round robin distribution
        assert lb.get_instance()["id"] == "instance-1"
        assert lb.get_instance()["id"] == "instance-2"
        assert lb.get_instance()["id"] == "instance-3"
        assert lb.get_instance()["id"] == "instance-1"  # Back to first instance
        
        # Test deregistration
        lb.deregister_instance("instance-2")
        
        assert lb.get_instance()["id"] == "instance-1"
        assert lb.get_instance()["id"] == "instance-3"  # Skips deregistered instance
    
    def test_least_connections(self):
        """Test least connections load balancing strategy."""
        lb = LoadBalancer(strategy="least_connections")
        
        # Register instances
        lb.register_instance({"id": "instance-1"})
        lb.register_instance({"id": "instance-2"})
        
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
    
    def test_ip_hash(self):
        """Test IP hash load balancing strategy."""
        lb = LoadBalancer(strategy="ip_hash")
        
        # Register instances
        lb.register_instance({"id": "instance-1"})
        lb.register_instance({"id": "instance-2"})
        
        # Same IP should always go to the same instance
        request_info = {"client_ip": "192.168.1.1"}
        instance1 = lb.get_instance(request_info)
        instance2 = lb.get_instance(request_info)
        assert instance1["id"] == instance2["id"]
        
        # Different IP may go to a different instance
        request_info = {"client_ip": "192.168.1.2"}
        instance3 = lb.get_instance(request_info)
        
        # No IP info should fall back to round robin
        instance4 = lb.get_instance()
        instance5 = lb.get_instance()
        assert instance4["id"] != instance5["id"]


class TestContainerManager:
    """Test container manager."""
    
    def test_get_full_image_name(self):
        """Test get_full_image_name method."""
        # Without registry
        manager = ContainerManager(image="test-image", tag="latest")
        assert manager.get_full_image_name() == "test-image:latest"
        
        # With registry
        manager = ContainerManager(image="test-image", tag="latest", registry="my-registry")
        assert manager.get_full_image_name() == "my-registry/test-image:latest"
    
    @patch("logging.Logger.info")
    def test_build_image(self, mock_logger):
        """Test build_image method."""
        manager = ContainerManager(image="test-image", tag="latest")
        assert manager.build_image() is True
        mock_logger.assert_called_once()
    
    @patch("logging.Logger.info")
    def test_push_image(self, mock_logger):
        """Test push_image method."""
        manager = ContainerManager(image="test-image", tag="latest")
        assert manager.push_image() is True
        mock_logger.assert_called_once()
    
    @patch("logging.Logger.info")
    def test_deploy(self, mock_logger):
        """Test deploy method."""
        manager = ContainerManager(image="test-image", tag="latest")
        assert manager.deploy(replicas=3) is True
        mock_logger.assert_called_once()
    
    @patch("logging.Logger.info")
    def test_scale(self, mock_logger):
        """Test scale method."""
        manager = ContainerManager(image="test-image", tag="latest")
        assert manager.scale(replicas=5) is True
        mock_logger.assert_called_once()
    
    def test_get_status(self):
        """Test get_status method."""
        manager = ContainerManager(image="test-image", tag="latest")
        status = manager.get_status()
        assert status["image"] == "test-image:latest"
        assert status["replicas"] == 2
        assert status["available_replicas"] == 2


class TestFactoryFunctions:
    """Test factory functions."""
    
    @patch("os.environ.get")
    def test_get_container_manager(self, mock_env_get):
        """Test get_container_manager function."""
        mock_env_get.side_effect = lambda key, default=None: {
            "CONTAINER_IMAGE": "custom-image",
            "CONTAINER_TAG": "v1.0",
            "CONTAINER_REGISTRY": "my-registry",
            "KUBERNETES_NAMESPACE": "custom-namespace",
        }.get(key, default)
        
        manager = get_container_manager()
        assert manager.image == "custom-image"
        assert manager.tag == "v1.0"
        assert manager.registry == "my-registry"
        assert manager.namespace == "custom-namespace"
    
    @patch("os.environ.get")
    def test_get_scaling_manager(self, mock_env_get):
        """Test get_scaling_manager function."""
        mock_env_get.side_effect = lambda key, default=None: {
            "MIN_INSTANCES": "3",
            "MAX_INSTANCES": "15",
            "SCALING_COOLDOWN": "600",
        }.get(key, default)
        
        manager = get_scaling_manager()
        assert manager.min_instances == 3
        assert manager.max_instances == 15
        assert manager.cooldown_period == 600
    
    @patch("os.environ.get")
    def test_get_load_balancer(self, mock_env_get):
        """Test get_load_balancer function."""
        mock_env_get.side_effect = lambda key, default=None: {
            "LOAD_BALANCER_STRATEGY": "least_connections",
        }.get(key, default)
        
        lb = get_load_balancer()
        assert lb.strategy == "least_connections"