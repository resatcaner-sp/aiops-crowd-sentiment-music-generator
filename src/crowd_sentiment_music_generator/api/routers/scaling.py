"""API endpoints for auto-scaling infrastructure management."""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from crowd_sentiment_music_generator.models.data.system_config import ScalingConfig, ResourcePriority
from crowd_sentiment_music_generator.utils.auto_scaling import (
    ContainerManager,
    LoadBalancer,
    ScalingManager,
    get_container_manager,
    get_scaling_manager,
    get_load_balancer,
)

router = APIRouter(prefix="/scaling", tags=["scaling"])
logger = logging.getLogger(__name__)


class ScalingStatus(BaseModel):
    """Scaling status response model."""
    
    enabled: bool
    current_instances: int
    min_instances: int
    max_instances: int
    cpu_threshold: float
    memory_threshold: float
    cooldown_period: int
    last_scaling_time: Optional[float] = None
    metrics: Dict[str, float]


class ScalingDecision(BaseModel):
    """Scaling decision response model."""
    
    should_scale: bool
    reason: str
    current_instances: int
    target_instances: int
    metrics: Dict[str, float]


class ScalingRequest(BaseModel):
    """Scaling request model."""
    
    target_instances: int = Field(ge=1)
    reason: Optional[str] = None


class MatchPriorityRequest(BaseModel):
    """Match priority request model."""
    
    match_id: str
    priority: int = Field(ge=1, le=10)
    resource_allocation: float = Field(ge=0.0, le=100.0)


@router.get("/status", response_model=ScalingStatus)
async def get_scaling_status(request: Request) -> ScalingStatus:
    """Get current scaling status.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Current scaling status
    """
    scaling_manager = get_scaling_manager()
    monitor = scaling_manager.monitor
    
    return ScalingStatus(
        enabled=True,
        current_instances=scaling_manager.current_instances,
        min_instances=scaling_manager.min_instances,
        max_instances=scaling_manager.max_instances,
        cpu_threshold=scaling_manager.policy.threshold if hasattr(scaling_manager.policy, "threshold") else 70.0,
        memory_threshold=80.0,  # Default value
        cooldown_period=scaling_manager.cooldown_period,
        last_scaling_time=scaling_manager.last_scale_time,
        metrics=monitor.get_current_usage(),
    )


@router.get("/decision", response_model=ScalingDecision)
async def get_scaling_decision(request: Request) -> ScalingDecision:
    """Get current scaling decision.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Current scaling decision
    """
    scaling_manager = get_scaling_manager()
    decision = scaling_manager.check_scaling()
    
    return ScalingDecision(
        should_scale=decision["should_scale"],
        reason=decision["reason"],
        current_instances=decision["current_instances"],
        target_instances=decision["target_instances"],
        metrics=decision["metrics"],
    )


@router.post("/scale", response_model=ScalingDecision, status_code=status.HTTP_202_ACCEPTED)
async def scale_manually(request: Request, scaling_request: ScalingRequest) -> ScalingDecision:
    """Scale the application manually.
    
    Args:
        request: FastAPI request object
        scaling_request: Scaling request
        
    Returns:
        Scaling decision
    """
    scaling_manager = get_scaling_manager()
    container_manager = get_container_manager()
    
    # Ensure target instances is within bounds
    target_instances = max(
        scaling_manager.min_instances,
        min(scaling_manager.max_instances, scaling_request.target_instances),
    )
    
    # Apply scaling
    success = container_manager.scale(target_instances)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to apply scaling",
        )
    
    # Update scaling manager's current instances
    scaling_manager.current_instances = target_instances
    
    # Return decision
    return ScalingDecision(
        should_scale=True,
        reason=scaling_request.reason or "Manual scaling",
        current_instances=scaling_manager.current_instances,
        target_instances=target_instances,
        metrics=scaling_manager.monitor.get_current_usage(),
    )


@router.get("/config", response_model=ScalingConfig)
async def get_scaling_config(request: Request) -> ScalingConfig:
    """Get current scaling configuration.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Current scaling configuration
    """
    if not hasattr(request.app.state, "config"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System configuration not available",
        )
    
    return request.app.state.config.scaling


@router.put("/config", response_model=ScalingConfig)
async def update_scaling_config(request: Request, config: ScalingConfig) -> ScalingConfig:
    """Update scaling configuration.
    
    Args:
        request: FastAPI request object
        config: New scaling configuration
        
    Returns:
        Updated scaling configuration
    """
    if not hasattr(request.app.state, "config"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System configuration not available",
        )
    
    # Update configuration
    request.app.state.config.scaling = config
    
    # Update scaling manager
    scaling_manager = get_scaling_manager()
    scaling_manager.min_instances = config.min_instances
    scaling_manager.max_instances = config.max_instances
    scaling_manager.cooldown_period = config.cooldown_period
    
    return config


@router.get("/priorities", response_model=List[ResourcePriority])
async def get_match_priorities(request: Request) -> List[ResourcePriority]:
    """Get current match priorities.
    
    Args:
        request: FastAPI request object
        
    Returns:
        List of match priorities
    """
    if not hasattr(request.app.state, "config"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System configuration not available",
        )
    
    return request.app.state.config.match_priorities


@router.post("/priorities", response_model=ResourcePriority)
async def add_match_priority(request: Request, priority: MatchPriorityRequest) -> ResourcePriority:
    """Add or update match priority.
    
    Args:
        request: FastAPI request object
        priority: Match priority
        
    Returns:
        Added or updated match priority
    """
    if not hasattr(request.app.state, "config"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System configuration not available",
        )
    
    # Create resource priority
    resource_priority = ResourcePriority(
        match_id=priority.match_id,
        priority=priority.priority,
        resource_allocation=priority.resource_allocation,
    )
    
    # Check if match already exists
    for i, existing in enumerate(request.app.state.config.match_priorities):
        if existing.match_id == priority.match_id:
            # Update existing priority
            request.app.state.config.match_priorities[i] = resource_priority
            return resource_priority
    
    # Add new priority
    request.app.state.config.match_priorities.append(resource_priority)
    
    return resource_priority


@router.delete("/priorities/{match_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_match_priority(request: Request, match_id: str) -> None:
    """Delete match priority.
    
    Args:
        request: FastAPI request object
        match_id: Match ID
    """
    if not hasattr(request.app.state, "config"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System configuration not available",
        )
    
    # Filter out the match priority
    original_length = len(request.app.state.config.match_priorities)
    request.app.state.config.match_priorities = [
        p for p in request.app.state.config.match_priorities if p.match_id != match_id
    ]
    
    # Check if match was found
    if len(request.app.state.config.match_priorities) == original_length:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match priority with ID {match_id} not found",
        )