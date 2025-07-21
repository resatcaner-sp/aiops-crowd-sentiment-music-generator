"""Export configuration data model."""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ExportFormat(str, Enum):
    """Enum for export formats."""
    
    MP4 = "mp4"
    MOV = "mov"
    MKV = "mkv"
    MP3 = "mp3"
    WAV = "wav"
    WEBM = "webm"


class QualityPreset(str, Enum):
    """Enum for quality presets."""
    
    LOW = "low"          # Low quality, small file size
    MEDIUM = "medium"    # Medium quality, balanced file size
    HIGH = "high"        # High quality, larger file size
    SOCIAL = "social"    # Optimized for social media
    BROADCAST = "broadcast"  # Broadcast quality


class ExportConfig(BaseModel):
    """Pydantic model for export configuration.
    
    Attributes:
        format: Export format
        quality: Quality preset
        output_path: Path where the exported file should be saved
        include_music: Whether to include music in the export
        include_commentary: Whether to include commentary in the export
        metadata: Optional metadata to embed in the exported file
        custom_options: Optional custom export options
    """
    
    format: ExportFormat = ExportFormat.MP4
    quality: QualityPreset = QualityPreset.HIGH
    output_path: str
    include_music: bool = True
    include_commentary: bool = True
    metadata: Optional[Dict[str, str]] = None
    custom_options: Optional[Dict[str, object]] = None