"""Export service for highlight music generator.

This module provides functionality to export music-enhanced highlights in various formats
with different quality options and embedded metadata.
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
from pydantic import BaseModel, Field

from crowd_sentiment_music_generator.config.cache_config import get_cache_settings
from crowd_sentiment_music_generator.exceptions.export_error import ExportError
from crowd_sentiment_music_generator.models.data.highlight_segment import HighlightSegment
from crowd_sentiment_music_generator.models.music.highlight_music import HighlightMusic
from crowd_sentiment_music_generator.utils.cache import RedisCache, cached, CacheManager
from crowd_sentiment_music_generator.utils.error_handling import with_error_handling


class ExportFormat(str, Enum):
    """Supported export formats."""
    
    MP4 = "mp4"
    MOV = "mov"
    WEBM = "webm"
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"


class QualityPreset(str, Enum):
    """Quality presets for different platforms."""
    
    LOW = "low"  # Low quality, small file size
    MEDIUM = "medium"  # Medium quality, balanced file size
    HIGH = "high"  # High quality, larger file size
    SOCIAL = "social"  # Optimized for social media platforms
    BROADCAST = "broadcast"  # Broadcast quality
    CUSTOM = "custom"  # Custom quality settings


class ExportMetadata(BaseModel):
    """Metadata for exported highlights.
    
    Attributes:
        title: Title of the highlight
        description: Description of the highlight
        tags: List of tags for the highlight
        author: Author or creator of the highlight
        copyright: Copyright information
        creation_date: Creation date in ISO format
        custom_fields: Additional custom metadata fields
    """
    
    title: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    copyright: Optional[str] = None
    creation_date: Optional[str] = None
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class ExportOptions(BaseModel):
    """Options for exporting highlights.
    
    Attributes:
        format: Export format
        quality_preset: Quality preset
        bitrate: Video bitrate in kbps (for video formats)
        audio_bitrate: Audio bitrate in kbps
        resolution: Video resolution (width, height)
        fps: Frames per second (for video formats)
        metadata: Metadata to embed in the exported file
    """
    
    format: ExportFormat
    quality_preset: QualityPreset = QualityPreset.MEDIUM
    bitrate: Optional[int] = None
    audio_bitrate: Optional[int] = None
    resolution: Optional[tuple[int, int]] = None
    fps: Optional[int] = None
    metadata: Optional[ExportMetadata] = None


class ExportService:
    """Service for exporting highlights with music.
    
    This service provides methods for exporting highlights with synchronized music
    in various formats and quality presets.
    
    Attributes:
        logger: Logger instance
        cache_manager: Cache manager for caching operations
    """
    
    # Quality preset configurations
    QUALITY_PRESETS = {
        QualityPreset.LOW: {
            ExportFormat.MP4: {"bitrate": 1500, "audio_bitrate": 128, "resolution": (640, 360), "fps": 24},
            ExportFormat.WEBM: {"bitrate": 1000, "audio_bitrate": 96, "resolution": (640, 360), "fps": 24},
            ExportFormat.MP3: {"audio_bitrate": 128},
            ExportFormat.WAV: {"audio_bitrate": 128},
            ExportFormat.OGG: {"audio_bitrate": 96},
        },
        QualityPreset.MEDIUM: {
            ExportFormat.MP4: {"bitrate": 4000, "audio_bitrate": 192, "resolution": (1280, 720), "fps": 30},
            ExportFormat.WEBM: {"bitrate": 2500, "audio_bitrate": 160, "resolution": (1280, 720), "fps": 30},
            ExportFormat.MP3: {"audio_bitrate": 192},
            ExportFormat.WAV: {"audio_bitrate": 256},
            ExportFormat.OGG: {"audio_bitrate": 160},
        },
        QualityPreset.HIGH: {
            ExportFormat.MP4: {"bitrate": 8000, "audio_bitrate": 320, "resolution": (1920, 1080), "fps": 30},
            ExportFormat.WEBM: {"bitrate": 6000, "audio_bitrate": 256, "resolution": (1920, 1080), "fps": 30},
            ExportFormat.MP3: {"audio_bitrate": 320},
            ExportFormat.WAV: {"audio_bitrate": 512},
            ExportFormat.OGG: {"audio_bitrate": 256},
        },
        QualityPreset.SOCIAL: {
            ExportFormat.MP4: {"bitrate": 5000, "audio_bitrate": 256, "resolution": (1080, 1080), "fps": 30},
            ExportFormat.WEBM: {"bitrate": 4000, "audio_bitrate": 192, "resolution": (1080, 1080), "fps": 30},
            ExportFormat.MP3: {"audio_bitrate": 256},
            ExportFormat.WAV: {"audio_bitrate": 320},
            ExportFormat.OGG: {"audio_bitrate": 192},
        },
        QualityPreset.BROADCAST: {
            ExportFormat.MP4: {"bitrate": 15000, "audio_bitrate": 384, "resolution": (1920, 1080), "fps": 60},
            ExportFormat.MOV: {"bitrate": 20000, "audio_bitrate": 384, "resolution": (1920, 1080), "fps": 60},
            ExportFormat.WAV: {"audio_bitrate": 1536},
        },
    }
    
    def __init__(self):
        """Initialize export service."""
        self.logger = logging.getLogger(__name__)
        self.cache_manager = CacheManager()
        self.cache_settings = get_cache_settings()
        self.highlight_cache_ttl = self.cache_settings.ttl_settings.get("highlight_data", 3600)
    
    @with_error_handling
    def export_highlight(
        self,
        highlight_segment: HighlightSegment,
        highlight_music: HighlightMusic,
        output_path: Union[str, Path],
        options: ExportOptions,
    ) -> str:
        """Export a highlight with synchronized music.
        
        Args:
            highlight_segment: Highlight segment to export
            highlight_music: Music composition for the highlight
            output_path: Path to save the exported file
            options: Export options
            
        Returns:
            Path to the exported file
            
        Raises:
            ValueError: If the highlight segment and music don't match
            MusicGenerationError: If export fails
            FileNotFoundError: If the video file doesn't exist
        """
        # Verify that highlight_segment and highlight_music match
        if highlight_segment.id != highlight_music.highlight_id:
            raise ValueError(
                f"Highlight segment ID ({highlight_segment.id}) "
                f"doesn't match music ID ({highlight_music.highlight_id})"
            )
        
        # Verify that the video file exists
        if not os.path.exists(highlight_segment.video_path):
            raise FileNotFoundError(f"Video file not found: {highlight_segment.video_path}")
        
        # Apply quality preset settings if not using custom settings
        options = self._apply_quality_preset(options)
        
        # Check if we have a cached export with the same parameters
        if self.cache_settings.enabled and options.quality_preset != QualityPreset.CUSTOM:
            cached_export = self.get_cached_export(
                highlight_segment.id, options.format, options.quality_preset
            )
            if cached_export:
                cached_path = cached_export.get("output_path")
                if os.path.exists(cached_path):
                    self.logger.info(
                        f"Using cached export for highlight {highlight_segment.id} "
                        f"in {options.format.value} format with {options.quality_preset.value} quality"
                    )
                    return cached_path
        
        # Determine export type (audio-only or video with audio)
        is_audio_only = options.format in [ExportFormat.MP3, ExportFormat.WAV, ExportFormat.OGG]
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            if is_audio_only:
                exported_path = self._export_audio_only(highlight_music, output_path, options)
            else:
                exported_path = self._export_video_with_audio(
                    highlight_segment, highlight_music, output_path, options
                )
            
            # Embed metadata if provided
            if options.metadata:
                self._embed_metadata(exported_path, options.metadata, options.format)
            
            # Cache the export result
            if self.cache_settings.enabled:
                self.cache_highlight_export_result(
                    highlight_segment.id, options.format, options.quality_preset, exported_path
                )
            
            self.logger.info(
                f"Successfully exported highlight {highlight_segment.id} to {exported_path} "
                f"in {options.format.value} format with {options.quality_preset.value} quality"
            )
            
            return exported_path
            
        except Exception as e:
            self.logger.error(f"Failed to export highlight {highlight_segment.id}: {str(e)}")
            raise ExportError(f"Export failed: {str(e)}")
    
    @with_error_handling
    def export_audio_only(
        self,
        highlight_music: HighlightMusic,
        output_path: Union[str, Path],
        options: ExportOptions,
    ) -> str:
        """Export only the audio part of a highlight.
        
        Args:
            highlight_music: Music composition for the highlight
            output_path: Path to save the exported file
            options: Export options
            
        Returns:
            Path to the exported file
            
        Raises:
            ValueError: If the format is not an audio format
            MusicGenerationError: If export fails
        """
        # Verify that the format is an audio format
        if options.format not in [ExportFormat.MP3, ExportFormat.WAV, ExportFormat.OGG]:
            raise ValueError(f"Format {options.format} is not an audio format")
        
        # Apply quality preset settings if not using custom settings
        options = self._apply_quality_preset(options)
        
        # Check if we have a cached export with the same parameters
        if self.cache_settings.enabled and options.quality_preset != QualityPreset.CUSTOM:
            cached_export = self.get_cached_export(
                highlight_music.highlight_id, options.format, options.quality_preset
            )
            if cached_export:
                cached_path = cached_export.get("output_path")
                if os.path.exists(cached_path):
                    self.logger.info(
                        f"Using cached audio export for highlight {highlight_music.highlight_id} "
                        f"in {options.format.value} format with {options.quality_preset.value} quality"
                    )
                    return cached_path
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            exported_path = self._export_audio_only(highlight_music, output_path, options)
            
            # Embed metadata if provided
            if options.metadata:
                self._embed_metadata(exported_path, options.metadata, options.format)
            
            # Cache the export result
            if self.cache_settings.enabled:
                self.cache_highlight_export_result(
                    highlight_music.highlight_id, options.format, options.quality_preset, exported_path
                )
            
            self.logger.info(
                f"Successfully exported audio for highlight {highlight_music.highlight_id} "
                f"to {exported_path} in {options.format.value} format with "
                f"{options.quality_preset.value} quality"
            )
            
            return exported_path
            
        except Exception as e:
            self.logger.error(
                f"Failed to export audio for highlight {highlight_music.highlight_id}: {str(e)}"
            )
            raise ExportError(f"Audio export failed: {str(e)}")
    
    def _apply_quality_preset(self, options: ExportOptions) -> ExportOptions:
        """Apply quality preset settings to options.
        
        Args:
            options: Export options
            
        Returns:
            Updated export options with preset settings applied
        """
        # If using custom quality, don't apply preset
        if options.quality_preset == QualityPreset.CUSTOM:
            return options
        
        # Get preset settings for the specified format
        preset_settings = self.QUALITY_PRESETS.get(options.quality_preset, {}).get(options.format, {})
        
        # Create a copy of options to modify
        updated_options = options.copy()
        
        # Apply preset settings if not already set
        if "bitrate" in preset_settings and updated_options.bitrate is None:
            updated_options.bitrate = preset_settings["bitrate"]
        
        if "audio_bitrate" in preset_settings and updated_options.audio_bitrate is None:
            updated_options.audio_bitrate = preset_settings["audio_bitrate"]
        
        if "resolution" in preset_settings and updated_options.resolution is None:
            updated_options.resolution = preset_settings["resolution"]
        
        if "fps" in preset_settings and updated_options.fps is None:
            updated_options.fps = preset_settings["fps"]
        
        return updated_options
    
    def _export_audio_only(
        self, highlight_music: HighlightMusic, output_path: Union[str, Path], options: ExportOptions
    ) -> str:
        """Export only the audio part of a highlight.
        
        Args:
            highlight_music: Music composition for the highlight
            output_path: Path to save the exported file
            options: Export options
            
        Returns:
            Path to the exported file
        """
        # In a real implementation, this would use a library like pydub or librosa
        # to render the music composition to an audio file
        # For this implementation, we'll simulate the export process
        
        self.logger.info(
            f"Exporting audio for highlight {highlight_music.highlight_id} "
            f"in {options.format.value} format with {options.audio_bitrate}kbps bitrate"
        )
        
        # Ensure the output path has the correct extension
        output_path = self._ensure_extension(output_path, options.format.value)
        
        # Simulate audio export
        # In a real implementation, this would generate the audio file
        with open(output_path, "w") as f:
            f.write(f"Simulated audio export for highlight {highlight_music.highlight_id}")
        
        return output_path
    
    def _export_video_with_audio(
        self,
        highlight_segment: HighlightSegment,
        highlight_music: HighlightMusic,
        output_path: Union[str, Path],
        options: ExportOptions,
    ) -> str:
        """Export a video with synchronized music.
        
        Args:
            highlight_segment: Highlight segment to export
            highlight_music: Music composition for the highlight
            output_path: Path to save the exported file
            options: Export options
            
        Returns:
            Path to the exported file
        """
        # In a real implementation, this would use a library like moviepy or ffmpeg
        # to combine the video and audio
        # For this implementation, we'll simulate the export process
        
        self.logger.info(
            f"Exporting video with audio for highlight {highlight_segment.id} "
            f"in {options.format.value} format with {options.bitrate}kbps video bitrate "
            f"and {options.audio_bitrate}kbps audio bitrate"
        )
        
        # Ensure the output path has the correct extension
        output_path = self._ensure_extension(output_path, options.format.value)
        
        # Simulate video export
        # In a real implementation, this would generate the video file with synchronized audio
        with open(output_path, "w") as f:
            f.write(
                f"Simulated video export for highlight {highlight_segment.id} "
                f"with resolution {options.resolution} and {options.fps} fps"
            )
        
        return output_path
    
    def _embed_metadata(
        self, file_path: Union[str, Path], metadata: ExportMetadata, format: ExportFormat
    ) -> None:
        """Embed metadata in the exported file.
        
        Args:
            file_path: Path to the exported file
            metadata: Metadata to embed
            format: File format
        """
        # In a real implementation, this would use a library like mutagen for audio files
        # or ffmpeg for video files to embed metadata
        # For this implementation, we'll simulate the metadata embedding process
        
        self.logger.info(f"Embedding metadata in {file_path}")
        
        # Simulate metadata embedding
        # In a real implementation, this would modify the file to include metadata
        pass
    
    def _ensure_extension(self, path: Union[str, Path], extension: str) -> str:
        """Ensure that the path has the correct file extension.
        
        Args:
            path: File path
            extension: Desired file extension (without dot)
            
        Returns:
            Path with the correct extension
        """
        path_str = str(path)
        
        # Remove any existing extension
        base_path = os.path.splitext(path_str)[0]
        
        # Add the correct extension
        return f"{base_path}.{extension}"
    
    @with_error_handling
    @cached(ttl=86400, key_prefix="export_formats", tag="export_config")
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get a dictionary of supported export formats and their file extensions.
        
        Returns:
            Dictionary mapping format names to lists of file extensions
            
        Note:
            This method is cached for 24 hours since the supported formats rarely change.
        """
        self.logger.debug("Cache miss for supported formats, generating fresh data")
        return {
            "video": [
                ExportFormat.MP4.value,
                ExportFormat.MOV.value,
                ExportFormat.WEBM.value,
            ],
            "audio": [
                ExportFormat.MP3.value,
                ExportFormat.WAV.value,
                ExportFormat.OGG.value,
            ],
        }
    
    @with_error_handling
    @cached(ttl=86400, key_prefix="quality_presets", tag="export_config")
    def get_quality_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get a dictionary of available quality presets and their settings.
        
        Returns:
            Dictionary mapping preset names to their settings
            
        Note:
            This method is cached for 24 hours since the quality presets rarely change.
        """
        self.logger.debug("Cache miss for quality presets, generating fresh data")
        return {preset.value: settings for preset, settings in self.QUALITY_PRESETS.items()}
        
    def cache_highlight_export_result(
        self, highlight_id: str, format: ExportFormat, quality: QualityPreset, output_path: str
    ) -> bool:
        """Cache the result of a highlight export.
        
        Args:
            highlight_id: ID of the highlight
            format: Export format
            quality: Quality preset used
            output_path: Path to the exported file
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.cache_settings.enabled:
            return False
            
        cache_key = f"highlight_export:{highlight_id}:{format.value}:{quality.value}"
        export_data = {
            "highlight_id": highlight_id,
            "format": format.value,
            "quality": quality.value,
            "output_path": str(output_path),
            "timestamp": os.path.getmtime(output_path) if os.path.exists(output_path) else 0,
        }
        
        # Cache with tags for efficient invalidation
        return self.cache_manager.set_with_tags(
            cache_key, 
            export_data, 
            tags=[f"highlight:{highlight_id}", "exports", f"format:{format.value}"],
            ttl=self.highlight_cache_ttl
        )
        
    def get_cached_export(
        self, highlight_id: str, format: ExportFormat, quality: QualityPreset
    ) -> Optional[Dict[str, Any]]:
        """Get cached export result if available.
        
        Args:
            highlight_id: ID of the highlight
            format: Export format
            quality: Quality preset
            
        Returns:
            Dictionary with export information or None if not cached
        """
        if not self.cache_settings.enabled:
            return None
            
        cache_key = f"highlight_export:{highlight_id}:{format.value}:{quality.value}"
        cached_data = self.cache_manager.cache.get_json(cache_key)
        
        if cached_data and os.path.exists(cached_data.get("output_path", "")):
            # Verify the file hasn't been modified since caching
            file_path = cached_data["output_path"]
            cached_timestamp = cached_data.get("timestamp", 0)
            current_timestamp = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
            
            if abs(current_timestamp - cached_timestamp) < 0.001:  # Allow small timestamp differences
                return cached_data
                
        return None
        
    def invalidate_highlight_cache(self, highlight_id: str) -> int:
        """Invalidate all cached exports for a specific highlight.
        
        Args:
            highlight_id: ID of the highlight
            
        Returns:
            Number of cache entries invalidated
        """
        if not self.cache_settings.enabled:
            return 0
            
        return self.cache_manager.invalidate_by_tag(f"highlight:{highlight_id}")