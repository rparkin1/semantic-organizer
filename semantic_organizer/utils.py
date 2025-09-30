"""
Utility functions for the semantic file organizer.

This module contains helper functions for directory validation, logging setup,
and other common operations.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[]
    )

    logger = logging.getLogger('semantic_organizer')
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def validate_directory(path: Union[str, Path], must_exist: bool = True, create_if_missing: bool = False) -> Path:
    """
    Validate and normalize a directory path.

    Args:
        path: Directory path to validate
        must_exist: If True, raise error if directory doesn't exist
        create_if_missing: If True, create directory if it doesn't exist

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If directory doesn't exist and must_exist is True
        NotADirectoryError: If path exists but is not a directory
        PermissionError: If directory is not accessible
    """
    path = Path(path).resolve()

    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {path}")

        # Check if directory is readable
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Directory is not readable: {path}")

        # Check if directory is writable (for output directory)
        if not os.access(path, os.W_OK):
            raise PermissionError(f"Directory is not writable: {path}")

    elif must_exist:
        raise FileNotFoundError(f"Directory does not exist: {path}")

    elif create_if_missing:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"Cannot create directory: {path}") from e

    return path


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create a safe filename by removing or replacing problematic characters.

    Args:
        filename: Original filename
        max_length: Maximum allowed filename length

    Returns:
        Safe filename string
    """
    # Characters that are problematic in filenames
    invalid_chars = '<>:"/\\|?*'

    # Replace invalid characters with underscores
    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')

    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip('. ')

    # Ensure filename is not empty
    if not safe_name:
        safe_name = 'unnamed_file'

    # Truncate if too long
    if len(safe_name) > max_length:
        name_parts = safe_name.rsplit('.', 1)
        if len(name_parts) == 2:
            name, ext = name_parts
            available_length = max_length - len(ext) - 1
            safe_name = name[:available_length] + '.' + ext
        else:
            safe_name = safe_name[:max_length]

    return safe_name


def format_bytes(size: int) -> str:
    """
    Format byte size in human-readable format.

    Args:
        size: Size in bytes

    Returns:
        Formatted size string (e.g., "1.2 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


class ProgressTracker:
    """Simple progress tracking utility."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description

    def update(self, increment: int = 1) -> None:
        """Update progress by specified increment."""
        self.current += increment
        if self.current % 10 == 0 or self.current == self.total:
            percentage = (self.current / self.total) * 100
            print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%)", end='', flush=True)

    def finish(self) -> None:
        """Mark progress as complete."""
        self.current = self.total
        percentage = (self.current / self.total) * 100
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - Complete!")