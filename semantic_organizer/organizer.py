"""
File organization module with comprehensive conflict resolution.

This module handles the actual moving of files and folders, with sophisticated
conflict resolution strategies and preservation of file attributes.
"""

import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from .utils import safe_filename

logger = logging.getLogger('semantic_organizer.organizer')


class FileOrganizer:
    """Main class for organizing files with conflict resolution."""

    def __init__(self, conflict_resolution: str = "rename", dry_run: bool = False, copy_files: bool = False):
        """
        Initialize the FileOrganizer.

        Args:
            conflict_resolution: Strategy for handling conflicts ("rename", "skip", "overwrite", "timestamp", "prompt")
            dry_run: If True, only simulate operations without actually moving/copying files
            copy_files: If True, copy files instead of moving them (preserves originals)
        """
        self.conflict_resolution = conflict_resolution
        self.dry_run = dry_run
        self.copy_files = copy_files

        # Track operations for logging and potential rollback
        self.operations_log: List[Dict] = []
        self.conflicts_log: List[Dict] = []

        # Cache for existing files to avoid repeated filesystem checks
        self._destination_cache: Dict[str, Set[str]] = {}

    def move_item(
        self,
        source: Path,
        destination_dir: Path,
        theme_name: str,
        preserve_structure: bool = True
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """
        Safely move or copy file or folder with conflict handling.

        Args:
            source: Source path to move or copy
            destination_dir: Base destination directory
            theme_name: Name of the theme directory
            preserve_structure: Whether to preserve folder structures

        Returns:
            Tuple of (success, final_destination_path, error_message)
        """
        try:
            # Create theme directory if it doesn't exist
            theme_dir = self.create_theme_folder(destination_dir, theme_name)

            # Determine final destination path
            if source.is_dir() and preserve_structure:
                # For directories, move the entire folder
                destination_path = theme_dir / source.name
            else:
                # For files, place directly in theme directory
                destination_path = theme_dir / source.name

            # Handle conflicts
            if destination_path.exists():
                resolved_path, conflict_action = self.handle_conflict(
                    source, destination_path, self.conflict_resolution
                )

                if resolved_path is None:
                    # Conflict resolution decided to skip
                    logger.info(f"Skipped moving {source} due to conflict resolution")
                    return False, None, "Skipped due to conflict"

                destination_path = resolved_path

            # Validate the move operation
            validation_error = self.validate_move_operation(source, destination_path)
            if validation_error:
                logger.error(f"Move validation failed: {validation_error}")
                return False, None, validation_error

            # Perform the operation (move or copy)
            success = self._execute_operation(source, destination_path)

            if success:
                # Log the operation
                self.operations_log.append({
                    'action': 'copy' if self.copy_files else 'move',
                    'source': str(source),
                    'destination': str(destination_path),
                    'theme': theme_name,
                    'timestamp': datetime.now().isoformat(),
                    'dry_run': self.dry_run
                })

                operation = "Copied" if self.copy_files else "Moved"
                logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}{operation} {source} to {destination_path}")
                return True, destination_path, None
            else:
                operation = "Copy" if self.copy_files else "Move"
                return False, None, f"{operation} operation failed"

        except Exception as e:
            operation = "copying" if self.copy_files else "moving"
            error_msg = f"Error {operation} {source}: {e}"
            logger.error(error_msg)
            return False, None, error_msg

    def handle_conflict(
        self,
        source: Path,
        destination: Path,
        mode: str
    ) -> Tuple[Optional[Path], str]:
        """
        Resolve naming conflicts based on the specified mode.

        Args:
            source: Source path
            destination: Conflicting destination path
            mode: Conflict resolution mode

        Returns:
            Tuple of (resolved_destination_path, action_taken)
        """
        if not destination.exists():
            return destination, "no_conflict"

        conflict_info = {
            'source': str(source),
            'destination': str(destination),
            'mode': mode,
            'timestamp': datetime.now().isoformat()
        }

        if mode == "rename":
            new_path = self.generate_unique_filename(destination.parent, destination.name)
            conflict_info.update({
                'action': 'renamed',
                'new_name': new_path.name
            })
            self.conflicts_log.append(conflict_info)
            logger.info(f"Conflict resolved by renaming: {destination.name} -> {new_path.name}")
            return new_path, "renamed"

        elif mode == "skip":
            conflict_info['action'] = 'skipped'
            self.conflicts_log.append(conflict_info)
            logger.info(f"Skipping due to conflict: {destination}")
            return None, "skipped"

        elif mode == "overwrite":
            conflict_info['action'] = 'overwritten'
            self.conflicts_log.append(conflict_info)
            logger.warning(f"Overwriting existing file: {destination}")
            return destination, "overwritten"

        elif mode == "timestamp":
            new_path = self.apply_timestamp_suffix(destination)
            conflict_info.update({
                'action': 'timestamp_renamed',
                'new_name': new_path.name
            })
            self.conflicts_log.append(conflict_info)
            logger.info(f"Conflict resolved with timestamp: {destination.name} -> {new_path.name}")
            return new_path, "timestamp_renamed"

        elif mode == "prompt":
            # Interactive resolution
            return self._interactive_conflict_resolution(source, destination, conflict_info)

        else:
            logger.error(f"Unknown conflict resolution mode: {mode}")
            return self.generate_unique_filename(destination.parent, destination.name), "renamed"

    def generate_unique_filename(self, destination_dir: Path, original_name: str) -> Path:
        """
        Create numbered filename that doesn't conflict.

        Args:
            destination_dir: Directory where file will be placed
            original_name: Original filename

        Returns:
            Path with unique filename
        """
        base_path = destination_dir / original_name

        if not base_path.exists():
            return base_path

        # Parse filename
        if '.' in original_name and not original_name.startswith('.'):
            # File with extension
            name_parts = original_name.rsplit('.', 1)
            base_name = name_parts[0]
            extension = '.' + name_parts[1]
        else:
            # File without extension or directory
            base_name = original_name
            extension = ''

        # Find the next available number
        counter = self.get_next_available_number(destination_dir, base_name, extension)

        new_name = f"{base_name}_{counter}{extension}"
        return destination_dir / new_name

    def get_next_available_number(self, base_path: Path, base_name: str, extension: str) -> int:
        """
        Find the next available counter number for a filename.

        Args:
            base_path: Base directory path
            base_name: Base filename without extension
            extension: File extension (including dot)

        Returns:
            Next available counter number
        """
        existing_numbers = set()

        try:
            # Get cached filenames or scan directory
            cache_key = str(base_path)
            if cache_key not in self._destination_cache:
                self._destination_cache[cache_key] = {f.name for f in base_path.iterdir() if f.is_file() or f.is_dir()}

            existing_files = self._destination_cache[cache_key]

            # Look for existing numbered files
            import re
            pattern = re.compile(rf"^{re.escape(base_name)}_(\d+){re.escape(extension)}$")

            for filename in existing_files:
                match = pattern.match(filename)
                if match:
                    existing_numbers.add(int(match.group(1)))

            # Also check for the base file without number
            if f"{base_name}{extension}" in existing_files:
                existing_numbers.add(0)

        except Exception as e:
            logger.warning(f"Error scanning for existing numbers: {e}")

        # Find next available number
        counter = 1
        while counter in existing_numbers:
            counter += 1

        return counter

    def apply_timestamp_suffix(self, destination: Path) -> Path:
        """
        Generate timestamp-based unique filename.

        Args:
            destination: Original destination path

        Returns:
            Path with timestamp suffix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if destination.suffix:
            # File with extension
            base_name = destination.stem
            extension = destination.suffix
            new_name = f"{base_name}_{timestamp}{extension}"
        else:
            # Directory or file without extension
            new_name = f"{destination.name}_{timestamp}"

        return destination.parent / new_name

    def _interactive_conflict_resolution(
        self,
        source: Path,
        destination: Path,
        conflict_info: Dict
    ) -> Tuple[Optional[Path], str]:
        """
        Handle interactive conflict resolution.

        Args:
            source: Source path
            destination: Conflicting destination path
            conflict_info: Conflict information dictionary

        Returns:
            Tuple of (resolved_path, action_taken)
        """
        try:
            # Get file information for comparison
            source_stat = source.stat()
            dest_stat = destination.stat()

            print(f"\nConflict detected:")
            print(f"  Source: {source}")
            print(f"    Size: {source_stat.st_size} bytes")
            print(f"    Modified: {datetime.fromtimestamp(source_stat.st_mtime)}")
            print(f"  Destination: {destination}")
            print(f"    Size: {dest_stat.st_size} bytes")
            print(f"    Modified: {datetime.fromtimestamp(dest_stat.st_mtime)}")

            while True:
                choice = input("\nChoose action: (r)ename, (s)kip, (o)verwrite, (a)uto-rename remaining: ").lower().strip()

                if choice == 'r':
                    new_path = self.generate_unique_filename(destination.parent, destination.name)
                    conflict_info.update({'action': 'interactively_renamed', 'new_name': new_path.name})
                    self.conflicts_log.append(conflict_info)
                    return new_path, "interactively_renamed"

                elif choice == 's':
                    conflict_info['action'] = 'interactively_skipped'
                    self.conflicts_log.append(conflict_info)
                    return None, "interactively_skipped"

                elif choice == 'o':
                    conflict_info['action'] = 'interactively_overwritten'
                    self.conflicts_log.append(conflict_info)
                    return destination, "interactively_overwritten"

                elif choice == 'a':
                    # Switch to auto-rename mode for remaining files
                    self.conflict_resolution = "rename"
                    new_path = self.generate_unique_filename(destination.parent, destination.name)
                    conflict_info.update({'action': 'auto_renamed_remaining', 'new_name': new_path.name})
                    self.conflicts_log.append(conflict_info)
                    print("Switched to auto-rename mode for remaining conflicts.")
                    return new_path, "auto_renamed_remaining"

                else:
                    print("Invalid choice. Please enter 'r', 's', 'o', or 'a'.")

        except Exception as e:
            logger.error(f"Error in interactive conflict resolution: {e}")
            # Fallback to rename
            new_path = self.generate_unique_filename(destination.parent, destination.name)
            return new_path, "fallback_renamed"

    def check_case_insensitive_conflict(self, destination: Path) -> bool:
        """
        Handle case-insensitive filesystem issues.

        Args:
            destination: Path to check

        Returns:
            True if there's a case conflict
        """
        if not destination.parent.exists():
            return False

        try:
            # Get all existing files in the directory
            existing_files = {f.name.lower(): f.name for f in destination.parent.iterdir()}
            return destination.name.lower() in existing_files and existing_files[destination.name.lower()] != destination.name

        except Exception as e:
            logger.warning(f"Error checking case-insensitive conflicts: {e}")
            return False

    def create_theme_folder(self, output_dir: Path, theme_name: str) -> Path:
        """
        Create new theme directory if it doesn't exist.

        Args:
            output_dir: Base output directory
            theme_name: Name of the theme

        Returns:
            Path to the theme directory
        """
        # Ensure theme name is safe for filesystem
        safe_theme_name = safe_filename(theme_name)
        theme_dir = output_dir / safe_theme_name

        if not self.dry_run and not theme_dir.exists():
            try:
                theme_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created theme directory: {theme_dir}")
            except Exception as e:
                logger.error(f"Error creating theme directory {theme_dir}: {e}")
                raise
        elif self.dry_run:
            logger.info(f"[DRY RUN] Would create theme directory: {theme_dir}")

        return theme_dir

    def preserve_structure(self, folder_path: Path) -> bool:
        """
        Ensure folder hierarchy is maintained during move.

        Args:
            folder_path: Path to the folder

        Returns:
            True if structure should be preserved
        """
        # Always preserve structure for folders - this is a core requirement
        return folder_path.is_dir()

    def validate_move_operation(self, source: Path, destination: Path) -> Optional[str]:
        """
        Pre-check for issues before moving.

        Args:
            source: Source path
            destination: Destination path

        Returns:
            Error message if validation fails, None if valid
        """
        try:
            # Check source exists
            if not source.exists():
                return f"Source does not exist: {source}"

            # Check source is readable
            if not os.access(source, os.R_OK):
                return f"Source is not readable: {source}"

            # Check destination parent directory is writable
            destination_parent = destination.parent
            if not destination_parent.exists():
                if self.dry_run:
                    # In dry run mode, just validate that we could create it
                    try:
                        # Check if we have permission to create in the parent's parent
                        grandparent = destination_parent.parent
                        if grandparent.exists() and not os.access(grandparent, os.W_OK):
                            return f"Cannot create destination directory (no write permission): {grandparent}"
                    except Exception as e:
                        return f"Cannot validate destination directory creation: {e}"
                else:
                    # Actually create the directory
                    try:
                        destination_parent.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        return f"Cannot create destination directory: {e}"

            if not self.dry_run and destination_parent.exists() and not os.access(destination_parent, os.W_OK):
                return f"Destination directory is not writable: {destination_parent}"

            # Check we're not moving a directory into itself
            if source.is_dir():
                try:
                    destination.resolve().relative_to(source.resolve())
                    return f"Cannot move directory into itself: {source} -> {destination}"
                except ValueError:
                    # This is expected when destination is not under source
                    pass

            # Check available disk space (basic check)
            if source.is_file():
                source_size = source.stat().st_size
                if hasattr(shutil, 'disk_usage'):
                    free_space = shutil.disk_usage(destination_parent).free
                    if source_size > free_space:
                        return f"Insufficient disk space for move operation"

            return None

        except Exception as e:
            return f"Validation error: {e}"

    def _execute_operation(self, source: Path, destination: Path) -> bool:
        """
        Execute the actual move or copy operation.

        Args:
            source: Source path
            destination: Destination path

        Returns:
            True if successful
        """
        try:
            operation = "copy" if self.copy_files else "move"

            if self.dry_run:
                logger.info(f"[DRY RUN] Would {operation} {source} -> {destination}")
                return True

            # Update destination cache
            cache_key = str(destination.parent)
            if cache_key in self._destination_cache:
                self._destination_cache[cache_key].add(destination.name)

            # Perform the operation
            if self.copy_files:
                if source.is_dir():
                    shutil.copytree(str(source), str(destination), dirs_exist_ok=False)
                else:
                    shutil.copy2(str(source), str(destination))

                # Verify the copy was successful
                if destination.exists():
                    logger.info(f"Copied {source} to {destination}")
                    return True
                else:
                    logger.error(f"Copy verification failed: {source} -> {destination}")
                    return False
            else:
                # Move operation (original behavior)
                shutil.move(str(source), str(destination))

                # Verify the move was successful
                if destination.exists() and not source.exists():
                    logger.info(f"Moved {source} to {destination}")
                    return True
                else:
                    logger.error(f"Move verification failed: {source} -> {destination}")
                    return False

        except Exception as e:
            operation = "copy" if self.copy_files else "move"
            logger.error(f"Error executing {operation} {source} -> {destination}: {e}")
            return False

    def get_operations_log(self) -> List[Dict]:
        """Get the log of all operations performed."""
        return self.operations_log.copy()

    def get_conflicts_log(self) -> List[Dict]:
        """Get the log of all conflicts encountered."""
        return self.conflicts_log.copy()

    def generate_report(self) -> Dict:
        """
        Generate a comprehensive report of the organization operation.

        Returns:
            Dictionary containing operation statistics and details
        """
        total_operations = len(self.operations_log)
        successful_moves = len([op for op in self.operations_log if op.get('action') == 'move'])
        total_conflicts = len(self.conflicts_log)

        # Count conflict resolution actions
        conflict_actions = {}
        for conflict in self.conflicts_log:
            action = conflict.get('action', 'unknown')
            conflict_actions[action] = conflict_actions.get(action, 0) + 1

        report = {
            'total_operations': total_operations,
            'successful_moves': successful_moves,
            'total_conflicts': total_conflicts,
            'conflict_resolutions': conflict_actions,
            'dry_run': self.dry_run,
            'operations_log': self.operations_log,
            'conflicts_log': self.conflicts_log
        }

        return report

    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self._destination_cache.clear()
        logger.debug("Cleared destination cache")