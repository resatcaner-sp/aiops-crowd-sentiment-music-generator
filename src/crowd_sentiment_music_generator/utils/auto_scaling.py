"""Auto-scaling infrastructure utilities.

This module provides utilities for managing auto-scaling infrastructure,
including resource monitoring, scaling policies, and load balancing.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import psutil

from crowd_sentiment_music_generator.utils.resource_monitoring import ResourceUsageStats, SystemResourceMonitor

logger = logging.getLogger(__name__)


class ScalingPolicy:
    """Base class for scaling policies.
    
    Attributes:
        name: Policy name
        description: Policy description
    """
    
    def __init__(self, name: str, description: str) -> None:
        """Initialize scaling policy.
        
        Args:
            name: Policy name
            description: Policy description
        """
        self.name = name
        self.description = description
    
    def should_scale(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scaling is needed based on metrics.
        
        Args:
            metrics: Resource usage metrics
            
        Returns:
            True if scaling is needed, False otherwise
        """
        raise NotImplementedError("Subclasses must implement should_scale")
    
    def get_scale_factor(self, metrics: Dict[str, Any]) -> float:
        """Get scaling factor based on metrics.
        
        Args:
            metrics: Resource usage metrics
            
        Returns:
            Scaling factor (e.g., 1.5 for 50% increase)
        """
        raise NotImplementedError("Subclasses must implement get_scale_factor")


class CPUBasedScalingPolicy(ScalingPolicy):
    """CPU-based scaling policy.
    
    Attributes:
        threshold: CPU usage threshold for scaling (percentage)
        scale_up_factor: Factor to scale up by when threshold is exceeded
        scale_down_factor: Factor to scale down by when usage is below threshold
    """
    
    def __init__(
        self,
        threshold: float = 70.0,
        scale_up_factor: float = 1.5,
        scale_down_factor: float = 0.8,
    ) -> None:
        """Initialize CPU-based scaling policy.
        
        Args:
            threshold: CPU usage threshold for scaling (percentage)
            scale_up_factor: Factor to scale up by when threshold is exceeded
            scale_down_factor: Factor to scale down by when usage is below threshold
        """
        super().__init__(
            name="cpu-based-scaling",
            description=f"Scale based on CPU usage (threshold: {threshold}%)",
        )
        self.threshold = threshold
        self.scale_up_factor = scale_up_factor
        self.scale_down_factor = scale_down_factor
    
    def should_scale(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scaling is needed based on CPU metrics.
        
        Args:
            metrics: Resource usage metrics
            
        Returns:
            True if scaling is needed, False otherwise
        """
        if "cpu_percent" not in metrics:
            return False
        
        cpu_percent = metrics["cpu_percent"]
        return cpu_percent > self.threshold or cpu_percent < (self.threshold * 0.5)
    
    def get_scale_factor(self, metrics: Dict[str, Any]) -> float:
        """Get scaling factor based on CPU metrics.
        
        Args:
            metrics: Resource usage metrics
            
        Returns:
            Scaling factor
        """
        if "cpu_percent" not in metrics:
            return 1.0
        
        cpu_percent = metrics["cpu_percent"]
        if cpu_percent > self.threshold:
            # Scale up
            return self.scale_up_factor
        elif cpu_percent < (self.threshold * 0.5):
            # Scale down
            return self.scale_down_factor
        else:
            # No change
            return 1.0


class MemoryBasedScalingPolicy(ScalingPolicy):
    """Memory-based scaling policy.
    
    Attributes:
        threshold: Memory usage threshold for scaling (percentage)
        scale_up_factor: Factor to scale up by when threshold is exceeded
        scale_down_factor: Factor to scale down by when usage is below threshold
    """
    
    def __init__(
        self,
        threshold: float = 80.0,
        scale_up_factor: float = 1.5,
        scale_down_factor: float = 0.8,
    ) -> None:
        """Initialize memory-based scaling policy.
        
        Args:
            threshold: Memory usage threshold for scaling (percentage)
            scale_up_factor: Factor to scale up by when threshold is exceeded
            scale_down_factor: Factor to scale down by when usage is below threshold
        """
        super().__init__(
            name="memory-based-scaling",
            description=f"Scale based on memory usage (threshold: {threshold}%)",
        )
        self.threshold = threshold
        self.scale_up_factor = scale_up_factor
        self.scale_down_factor = scale_down_factor
    
    def should_scale(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scaling is needed based on memory metrics.
        
        Args:
            metrics: Resource usage metrics
            
        Returns:
            True if scaling is needed, False otherwise
        """
        if "memory_percent" not in metrics:
            return False
        
        memory_percent = metrics["memory_percent"]
        return memory_percent > self.threshold or memory_percent < (self.threshold * 0.5)
    
    def get_scale_factor(self, metrics: Dict[str, Any]) -> float:
        """Get scaling factor based on memory metrics.
        
        Args:
            metrics: Resource usage metrics
            
        Returns:
            Scaling factor
        """
        if "memory_percent" not in metrics:
            return 1.0
        
        memory_percent = metrics["memory_percent"]
        if memory_percent > self.threshold:
            # Scale up
            return self.scale_up_factor
        elif memory_percent < (self.threshold * 0.5):
            # Scale down
            return self.scale_down_factor
        else:
            # No change
            return 1.0


class RequestRateScalingPolicy(ScalingPolicy):
    """Request rate-based scaling policy.
    
    Attributes:
        threshold: Request rate threshold for scaling (requests per second)
        scale_up_factor: Factor to scale up by when threshold is exceeded
        scale_down_factor: Factor to scale down by when rate is below threshold
    """
    
    def __init__(
        self,
        threshold: float = 100.0,
        scale_up_factor: float = 1.5,
        scale_down_factor: float = 0.8,
    ) -> None:
        """Initialize request rate-based scaling policy.
        
        Args:
            threshold: Request rate threshold for scaling (requests per second)
            scale_up_factor: Factor to scale up by when threshold is exceeded
            scale_down_factor: Factor to scale down by when rate is below threshold
        """
        super().__init__(
            name="request-rate-scaling",
            description=f"Scale based on request rate (threshold: {threshold} req/s)",
        )
        self.threshold = threshold
        self.scale_up_factor = scale_up_factor
        self.scale_down_factor = scale_down_factor
    
    def should_scale(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scaling is needed based on request rate metrics.
        
        Args:
            metrics: Resource usage metrics
            
        Returns:
            True if scaling is needed, False otherwise
        """
        if "request_rate" not in metrics:
            return False
        
        request_rate = metrics["request_rate"]
        return request_rate > self.threshold or request_rate < (self.threshold * 0.5)
    
    def get_scale_factor(self, metrics: Dict[str, Any]) -> float:
        """Get scaling factor based on request rate metrics.
        
        Args:
            metrics: Resource usage metrics
            
        Returns:
            Scaling factor
        """
        if "request_rate" not in metrics:
            return 1.0
        
        request_rate = metrics["request_rate"]
        if request_rate > self.threshold:
            # Scale up
            return self.scale_up_factor
        elif request_rate < (self.threshold * 0.5):
            # Scale down
            return self.scale_down_factor
        else:
            # No change
            return 1.0


class CompositeScalingPolicy(ScalingPolicy):
    """Composite scaling policy that combines multiple policies.
    
    Attributes:
        policies: List of scaling policies
    """
    
    def __init__(self, policies: List[ScalingPolicy]) -> None:
        """Initialize composite scaling policy.
        
        Args:
            policies: List of scaling policies
        """
        super().__init__(
            name="composite-scaling",
            description=f"Composite scaling policy with {len(policies)} sub-policies",
        )
        self.policies = policies
    
    def should_scale(self, metrics: Dict[str, Any]) -> bool:
        """Determine if scaling is needed based on any policy.
        
        Args:
            metrics: Resource usage metrics
            
        Returns:
            True if any policy indicates scaling is needed, False otherwise
        """
        return any(policy.should_scale(metrics) for policy in self.policies)
    
    def get_scale_factor(self, metrics: Dict[str, Any]) -> float:
        """Get maximum scaling factor from all policies.
        
        Args:
            metrics: Resource usage metrics
            
        Returns:
            Maximum scaling factor
        """
        factors = [policy.get_scale_factor(metrics) for policy in self.policies]
        # Return the most extreme factor (furthest from 1.0)
        max_up = max(factors)
        min_down = min(factors)
        
        if max_up > 1.0 and min_down < 1.0:
            # If we have conflicting scale directions, prioritize scaling up
            return max_up
        elif max_up > 1.0:
            return max_up
        elif min_down < 1.0:
            return min_down
        else:
            return 1.0


class ScalingManager:
    """Manager for auto-scaling operations.
    
    Attributes:
        policy: Scaling policy
        monitor: System resource monitor
        min_instances: Minimum number of instances
        max_instances: Maximum number of instances
        cooldown_period: Cooldown period between scaling operations (seconds)
    """
    
    def __init__(
        self,
        policy: Optional[ScalingPolicy] = None,
        monitor: Optional[SystemResourceMonitor] = None,
        min_instances: int = 2,
        max_instances: int = 10,
        cooldown_period: int = 300,
    ) -> None:
        """Initialize scaling manager.
        
        Args:
            policy: Scaling policy
            monitor: System resource monitor
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            cooldown_period: Cooldown period between scaling operations (seconds)
        """
        self.policy = policy or CompositeScalingPolicy([
            CPUBasedScalingPolicy(),
            MemoryBasedScalingPolicy(),
            RequestRateScalingPolicy(),
        ])
        self.monitor = monitor or SystemResourceMonitor()
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.cooldown_period = cooldown_period
        self.last_scale_time = 0
        self.current_instances = min_instances
    
    def check_scaling(self) -> Dict[str, Any]:
        """Check if scaling is needed and calculate new instance count.
        
        Returns:
            Dictionary with scaling decision
        """
        # Get current metrics
        metrics = self.monitor.get_current_usage()
        
        # Check if we're in cooldown period
        current_time = time.time()
        if current_time - self.last_scale_time < self.cooldown_period:
            return {
                "should_scale": False,
                "reason": "In cooldown period",
                "current_instances": self.current_instances,
                "target_instances": self.current_instances,
                "metrics": metrics,
            }
        
        # Check if scaling is needed
        if not self.policy.should_scale(metrics):
            return {
                "should_scale": False,
                "reason": "No scaling needed",
                "current_instances": self.current_instances,
                "target_instances": self.current_instances,
                "metrics": metrics,
            }
        
        # Calculate new instance count
        scale_factor = self.policy.get_scale_factor(metrics)
        target_instances = int(self.current_instances * scale_factor)
        
        # Ensure target is within bounds
        target_instances = max(self.min_instances, min(self.max_instances, target_instances))
        
        # If no change, return early
        if target_instances == self.current_instances:
            return {
                "should_scale": False,
                "reason": "Target instance count unchanged",
                "current_instances": self.current_instances,
                "target_instances": target_instances,
                "metrics": metrics,
            }
        
        # Update last scale time
        self.last_scale_time = current_time
        
        # Return scaling decision
        return {
            "should_scale": True,
            "reason": f"Scaling {'up' if target_instances > self.current_instances else 'down'} based on {self.policy.name}",
            "current_instances": self.current_instances,
            "target_instances": target_instances,
            "scale_factor": scale_factor,
            "metrics": metrics,
        }
    
    def apply_scaling(self, decision: Dict[str, Any]) -> bool:
        """Apply scaling decision.
        
        Args:
            decision: Scaling decision from check_scaling
            
        Returns:
            True if scaling was applied, False otherwise
        """
        if not decision.get("should_scale", False):
            return False
        
        target_instances = decision.get("target_instances", self.current_instances)
        
        # In a real implementation, this would call the Kubernetes API
        # to scale the deployment. For this implementation, we'll just
        # log the scaling decision.
        logger.info(
            f"Scaling from {self.current_instances} to {target_instances} instances: {decision['reason']}"
        )
        
        # Update current instances
        self.current_instances = target_instances
        
        return True


class LoadBalancer:
    """Load balancer for distributing requests across instances.
    
    Attributes:
        strategy: Load balancing strategy
        instances: List of available instances
    """
    
    def __init__(self, strategy: str = "round_robin") -> None:
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy (round_robin, least_connections, ip_hash)
        """
        self.strategy = strategy
        self.instances = []
        self.current_index = 0
        self.connections = {}
    
    def register_instance(self, instance: Dict[str, Any]) -> None:
        """Register an instance with the load balancer.
        
        Args:
            instance: Instance information
        """
        self.instances.append(instance)
    
    def deregister_instance(self, instance_id: str) -> None:
        """Deregister an instance from the load balancer.
        
        Args:
            instance_id: Instance ID
        """
        self.instances = [i for i in self.instances if i["id"] != instance_id]
    
    def get_instance(self, request_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get an instance to handle a request.
        
        Args:
            request_info: Request information
            
        Returns:
            Instance information or None if no instances available
        """
        if not self.instances:
            return None
        
        if self.strategy == "round_robin":
            instance = self.instances[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.instances)
            return instance
        elif self.strategy == "least_connections":
            instance = min(self.instances, key=lambda i: self.connections.get(i["id"], 0))
            self.connections[instance["id"]] = self.connections.get(instance["id"], 0) + 1
            return instance
        elif self.strategy == "ip_hash":
            if request_info and "client_ip" in request_info:
                # Simple hash function for IP address
                ip_hash = sum(ord(c) for c in request_info["client_ip"])
                index = ip_hash % len(self.instances)
                return self.instances[index]
            else:
                # Fall back to round robin
                return self.get_instance()
        else:
            # Default to round robin
            return self.get_instance()
    
    def release_instance(self, instance_id: str) -> None:
        """Release an instance after handling a request.
        
        Args:
            instance_id: Instance ID
        """
        if self.strategy == "least_connections" and instance_id in self.connections:
            self.connections[instance_id] = max(0, self.connections.get(instance_id, 0) - 1)


class ContainerManager:
    """Manager for container-based deployment.
    
    Attributes:
        image: Container image
        tag: Container image tag
        registry: Container registry
        namespace: Kubernetes namespace
    """
    
    def __init__(
        self,
        image: str = "crowd-sentiment-music-generator",
        tag: str = "latest",
        registry: Optional[str] = None,
        namespace: str = "crowd-sentiment-music-generator",
    ) -> None:
        """Initialize container manager.
        
        Args:
            image: Container image
            tag: Container image tag
            registry: Container registry
            namespace: Kubernetes namespace
        """
        self.image = image
        self.tag = tag
        self.registry = registry
        self.namespace = namespace
    
    def get_full_image_name(self) -> str:
        """Get full image name with registry and tag.
        
        Returns:
            Full image name
        """
        if self.registry:
            return f"{self.registry}/{self.image}:{self.tag}"
        else:
            return f"{self.image}:{self.tag}"
    
    def build_image(self, dockerfile_path: str = "Dockerfile") -> bool:
        """Build container image.
        
        Args:
            dockerfile_path: Path to Dockerfile
            
        Returns:
            True if successful, False otherwise
        """
        # In a real implementation, this would call Docker API
        # to build the image. For this implementation, we'll just
        # log the build command.
        logger.info(f"Building image {self.get_full_image_name()} from {dockerfile_path}")
        return True
    
    def push_image(self) -> bool:
        """Push container image to registry.
        
        Returns:
            True if successful, False otherwise
        """
        # In a real implementation, this would call Docker API
        # to push the image. For this implementation, we'll just
        # log the push command.
        logger.info(f"Pushing image {self.get_full_image_name()}")
        return True
    
    def deploy(self, replicas: int = 2) -> bool:
        """Deploy containers to Kubernetes.
        
        Args:
            replicas: Number of replicas
            
        Returns:
            True if successful, False otherwise
        """
        # In a real implementation, this would call Kubernetes API
        # to deploy the containers. For this implementation, we'll just
        # log the deployment command.
        logger.info(f"Deploying {replicas} replicas of {self.get_full_image_name()} to {self.namespace}")
        return True
    
    def scale(self, replicas: int) -> bool:
        """Scale deployment to specified number of replicas.
        
        Args:
            replicas: Number of replicas
            
        Returns:
            True if successful, False otherwise
        """
        # In a real implementation, this would call Kubernetes API
        # to scale the deployment. For this implementation, we'll just
        # log the scaling command.
        logger.info(f"Scaling deployment to {replicas} replicas")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status.
        
        Returns:
            Deployment status
        """
        # In a real implementation, this would call Kubernetes API
        # to get the deployment status. For this implementation, we'll just
        # return a mock status.
        return {
            "image": self.get_full_image_name(),
            "namespace": self.namespace,
            "replicas": 2,
            "available_replicas": 2,
            "ready_replicas": 2,
            "updated_replicas": 2,
            "conditions": [
                {
                    "type": "Available",
                    "status": "True",
                    "reason": "MinimumReplicasAvailable",
                    "message": "Deployment has minimum availability.",
                },
                {
                    "type": "Progressing",
                    "status": "True",
                    "reason": "NewReplicaSetAvailable",
                    "message": "ReplicaSet is up-to-date and available.",
                },
            ],
        }


def get_container_manager() -> ContainerManager:
    """Get container manager instance.
    
    Returns:
        Container manager instance
    """
    image = os.environ.get("CONTAINER_IMAGE", "crowd-sentiment-music-generator")
    tag = os.environ.get("CONTAINER_TAG", "latest")
    registry = os.environ.get("CONTAINER_REGISTRY")
    namespace = os.environ.get("KUBERNETES_NAMESPACE", "crowd-sentiment-music-generator")
    
    return ContainerManager(image=image, tag=tag, registry=registry, namespace=namespace)


def get_scaling_manager() -> ScalingManager:
    """Get scaling manager instance.
    
    Returns:
        Scaling manager instance
    """
    min_instances = int(os.environ.get("MIN_INSTANCES", "2"))
    max_instances = int(os.environ.get("MAX_INSTANCES", "10"))
    cooldown_period = int(os.environ.get("SCALING_COOLDOWN", "300"))
    
    return ScalingManager(
        min_instances=min_instances,
        max_instances=max_instances,
        cooldown_period=cooldown_period,
    )


def get_load_balancer() -> LoadBalancer:
    """Get load balancer instance.
    
    Returns:
        Load balancer instance
    """
    strategy = os.environ.get("LOAD_BALANCER_STRATEGY", "round_robin")
    return LoadBalancer(strategy=strategy)