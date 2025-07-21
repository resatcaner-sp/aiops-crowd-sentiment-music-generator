"""System configuration data model."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ScalingConfig(BaseModel):
    """Pydantic model for auto-scaling configuration.
    
    Attributes:
        enabled: Whether auto-scaling is enabled
        min_instances: Minimum number of instances
        max_instances: Maximum number of instances
        cpu_threshold: CPU usage threshold for scaling (percentage)
        memory_threshold: Memory usage threshold for scaling (percentage)
        cooldown_period: Cooldown period between scaling operations (seconds)
        load_balancer_strategy: Load balancing strategy
    """
    
    enabled: bool = True
    min_instances: int = 2
    max_instances: int = 10
    cpu_threshold: float = 70.0
    memory_threshold: float = 80.0
    cooldown_period: int = 300
    load_balancer_strategy: str = "round_robin"  # round_robin, least_connections, ip_hash


class ResourcePriority(BaseModel):
    """Pydantic model for resource priority configuration.
    
    Attributes:
        match_id: Match ID
        priority: Priority level (1-10, higher is more important)
        resource_allocation: Resource allocation percentage
    """
    
    match_id: str
    priority: int = Field(ge=1, le=10)
    resource_allocation: float = Field(ge=0.0, le=100.0)


class SystemConfig(BaseModel):
    """Pydantic model for system configuration.
    
    Attributes:
        audio_sample_rate: Sample rate for audio processing
        buffer_size: Size of the buffer for storing audio segments (in seconds)
        update_interval: Interval for updating the system state (in seconds)
        emotion_update_interval: Interval for updating emotion classification (in seconds)
        music_update_interval: Interval for updating music generation (in seconds)
        models_path: Path to the pre-trained models
        cultural_adaptation: Cultural adaptation setting (global or specific region)
        scaling: Auto-scaling configuration
        match_priorities: List of match priorities for resource allocation
    """
    
    audio_sample_rate: int = 22050
    buffer_size: int = 30  # seconds
    update_interval: float = 0.5  # seconds
    emotion_update_interval: float = 2.0  # seconds
    music_update_interval: float = 0.5  # seconds
    models_path: str = "./models"
    cultural_adaptation: str = "global"  # or specific region
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    match_priorities: List[ResourcePriority] = Field(default_factory=list)