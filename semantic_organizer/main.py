"""
Main entry point for the Semantic File Organizer.

This module provides the command-line interface and orchestrates the entire
file organization process using semantic similarity analysis.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .analyzer import SemanticAnalyzer
from .clusterer import Clusterer
from .file_extractors import FileExtractor
from .organizer import FileOrganizer
from .theme_matcher import ThemeMatcher
from .utils import setup_logging, validate_directory, ProgressTracker, format_bytes


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Automatically organize files and folders by semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m semantic_organizer input_folder output_folder

  # Copy files instead of moving them
  python -m semantic_organizer input/ output/ --copy

  # With custom similarity threshold
  python -m semantic_organizer input/ output/ --similarity-threshold 0.8

  # Dry run to preview organization
  python -m semantic_organizer input/ output/ --dry-run

  # Skip conflicts instead of renaming
  python -m semantic_organizer input/ output/ --conflict-resolution skip

  # Verbose logging
  python -m semantic_organizer input/ output/ --verbose
        """
    )

    # Directory arguments (required unless --list-models is used)
    parser.add_argument(
        'input_directory',
        type=str,
        nargs='?',  # Make optional
        help='Source directory containing files/folders to organize'
    )

    parser.add_argument(
        'output_directory',
        type=str,
        nargs='?',  # Make optional
        help='Destination directory where themed subdirectories will be created'
    )

    # Optional arguments
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.7,
        help='Minimum similarity for theme matching (0.0-1.0, default: 0.7)'
    )

    parser.add_argument(
        '--conflict-resolution',
        choices=['rename', 'skip', 'overwrite', 'timestamp', 'prompt'],
        default='rename',
        help='How to handle naming conflicts (default: rename)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview organization without moving files'
    )

    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them (preserves originals in input directory)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )

    parser.add_argument(
        '--embedding-model',
        type=str,
        default='intfloat/e5-base',
        help='Sentence transformer model to use. Popular options: intfloat/e5-base (default, quality), '
             'all-MiniLM-L6-v2 (fast), sentence-transformers/paraphrase-mpnet-base-v2 (SBERT quality), '
             'intfloat/e5-small (balanced). Use --list-models to see all available models.'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available embedding models and exit'
    )

    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=2,
        help='Minimum items needed to form a new theme (default: 2)'
    )


    parser.add_argument(
        '--max-file-size',
        type=int,
        default=100,
        help='Maximum file size to process in MB (default: 100)'
    )

    parser.add_argument(
        '--sample-limit',
        type=int,
        default=50000,
        help='Maximum characters to extract from large files (default: 50000)'
    )

    parser.add_argument(
        '--skip-extensions',
        type=str,
        nargs='+',
        help='File extensions to skip entirely (e.g., .exe .dll)'
    )

    parser.add_argument(
        '--content-weight',
        type=float,
        default=0.8,
        help='Weight of content vs filename in similarity (default: 0.8)'
    )

    parser.add_argument(
        '--filename-weight',
        type=float,
        default=0.2,
        help='Weight of filename vs content in similarity (default: 0.2)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts (use with overwrite mode)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        help='Optional file to write logs to'
    )

    return parser


class SemanticFileOrganizer:
    """Main orchestrator class for the semantic file organization process."""

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the organizer with command-line arguments.

        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.logger = setup_logging(args.verbose, args.log_file)

        # Validate weights sum to 1.0
        if abs((args.content_weight + args.filename_weight) - 1.0) > 0.01:
            self.logger.warning("Content weight and filename weight should sum to 1.0")

        # Initialize components
        self.analyzer = None
        self.theme_matcher = None
        self.clusterer = None
        self.organizer = None

        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_items': 0,
            'files_processed': 0,
            'folders_processed': 0,
            'existing_themes_used': 0,
            'new_themes_created': 0,
            'conflicts_encountered': 0,
            'items_moved': 0,
            'items_skipped': 0,
            'errors_encountered': 0
        }

    def run(self) -> int:
        """
        Execute the complete file organization process.

        Returns:
            Exit code (0 for success, 1 for partial success, 2 for failure)
        """
        try:
            self.stats['start_time'] = time.time()

            # Phase 1: Setup & Validation
            if not self._setup_and_validate():
                return 2

            # Phase 2: Discovery
            input_items, existing_themes = self._discovery_phase()
            if not input_items:
                self.logger.info("No items found to organize")
                return 0

            # Phase 3: Analysis
            analyzed_items = self._analysis_phase(input_items)

            # Phase 4: Theme Matching
            matched_items, unmatched_items = self._theme_matching_phase(analyzed_items, existing_themes)

            # Phase 5: Clustering & Theme Creation
            clustered_items = self._clustering_phase(unmatched_items)

            # Phase 6: Organization
            self._organization_phase(matched_items, clustered_items)

            # Phase 7: Reporting
            exit_code = self._generate_final_report()

            return exit_code

        except KeyboardInterrupt:
            self.logger.info("Operation interrupted by user")
            return 1
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.stats['end_time'] = time.time()
            return 2
        finally:
            if self.stats['end_time'] is None:
                self.stats['end_time'] = time.time()

    def _setup_and_validate(self) -> bool:
        """
        Setup and validation phase.

        Returns:
            True if setup successful
        """
        self.logger.info("=== Phase 1: Setup & Validation ===")

        try:
            # Validate directories
            self.input_dir = validate_directory(self.args.input_directory, must_exist=True)
            self.output_dir = validate_directory(
                self.args.output_directory,
                must_exist=False,
                create_if_missing=not self.args.dry_run
            )

            self.logger.info(f"Input directory: {self.input_dir}")
            self.logger.info(f"Output directory: {self.output_dir}")

            if self.args.dry_run:
                self.logger.info("DRY RUN MODE - No files will be moved")

            # Initialize components
            self.logger.info("Initializing semantic analyzer...")
            self.analyzer = SemanticAnalyzer(
                embedding_model=self.args.embedding_model,
                max_file_size_mb=self.args.max_file_size,
                sample_limit=self.args.sample_limit,
                content_weight=self.args.content_weight,
                filename_weight=self.args.filename_weight
            )

            self.theme_matcher = ThemeMatcher(
                analyzer=self.analyzer,
                similarity_threshold=self.args.similarity_threshold,
                min_cluster_size=self.args.min_cluster_size
            )

            self.clusterer = Clusterer(
                min_cluster_size=self.args.min_cluster_size
            )

            self.organizer = FileOrganizer(
                conflict_resolution=self.args.conflict_resolution,
                dry_run=self.args.dry_run,
                copy_files=self.args.copy
            )

            self.logger.info("Setup completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            return False

    def _discovery_phase(self) -> Tuple[List[Path], Dict]:
        """
        Discovery phase - scan input and output directories.

        Returns:
            Tuple of (input_items, existing_themes)
        """
        self.logger.info("=== Phase 2: Discovery ===")

        # Scan input directory
        input_items = []
        try:
            self.logger.info(f"Scanning input directory: {self.input_dir}")
            for item in self.input_dir.iterdir():
                if item.name.startswith('.'):
                    continue  # Skip hidden files/folders

                if self.args.skip_extensions and item.is_file():
                    if item.suffix.lower() in [ext.lower() for ext in self.args.skip_extensions]:
                        self.logger.debug(f"Skipping {item} due to extension filter")
                        continue

                input_items.append(item)

            self.stats['total_items'] = len(input_items)
            self.stats['files_processed'] = len([item for item in input_items if item.is_file()])
            self.stats['folders_processed'] = len([item for item in input_items if item.is_dir()])

            self.logger.info(f"Found {len(input_items)} items ({self.stats['files_processed']} files, {self.stats['folders_processed']} folders)")

        except Exception as e:
            self.logger.error(f"Error scanning input directory: {e}")
            input_items = []

        # Scan output directory for existing themes
        existing_themes = self.theme_matcher.load_existing_themes(self.output_dir)
        self.logger.info(f"Found {len(existing_themes)} existing themes")

        return input_items, existing_themes

    def _analysis_phase(self, input_items: List[Path]) -> List[Tuple[Path, Dict, Dict]]:
        """
        Analysis phase - generate embeddings for all items.

        Args:
            input_items: List of paths to analyze

        Returns:
            List of tuples containing (path, embedding, metadata)
        """
        self.logger.info("=== Phase 3: Analysis ===")

        analyzed_items = []
        progress = ProgressTracker(len(input_items), "Analyzing items")

        for item in input_items:
            try:
                embedding, metadata = self.analyzer.analyze_item(item)
                analyzed_items.append((item, embedding, metadata))
                progress.update()

            except Exception as e:
                self.logger.error(f"Error analyzing {item}: {e}")
                self.stats['errors_encountered'] += 1
                progress.update()

        progress.finish()
        self.logger.info(f"Successfully analyzed {len(analyzed_items)} items")

        return analyzed_items

    def _theme_matching_phase(
        self,
        analyzed_items: List[Tuple[Path, Dict, Dict]],
        existing_themes: Dict
    ) -> Tuple[List[Tuple[Path, str, float]], List[Tuple[Path, Dict, Dict]]]:
        """
        Theme matching phase - match items to existing themes.

        Args:
            analyzed_items: List of analyzed item tuples
            existing_themes: Dictionary of existing themes

        Returns:
            Tuple of (matched_items, unmatched_items)
        """
        self.logger.info("=== Phase 4: Theme Matching ===")

        matched_items = []
        unmatched_items = []

        progress = ProgressTracker(len(analyzed_items), "Matching to themes")

        for item_path, embedding, metadata in analyzed_items:
            try:
                match_result = self.theme_matcher.match_to_theme(embedding, item_path)

                if match_result:
                    theme_name, similarity = match_result
                    matched_items.append((item_path, theme_name, similarity))
                    self.stats['existing_themes_used'] += 1
                else:
                    unmatched_items.append((item_path, embedding, metadata))

                progress.update()

            except Exception as e:
                self.logger.error(f"Error matching {item_path}: {e}")
                unmatched_items.append((item_path, embedding, metadata))
                self.stats['errors_encountered'] += 1
                progress.update()

        progress.finish()
        self.logger.info(f"Matched {len(matched_items)} items to existing themes")
        self.logger.info(f"{len(unmatched_items)} items need new themes")

        return matched_items, unmatched_items

    def _clustering_phase(
        self,
        unmatched_items: List[Tuple[Path, Dict, Dict]]
    ) -> List[Tuple[List[Path], str]]:
        """
        Clustering phase - group unmatched items and create themes.

        Args:
            unmatched_items: List of unmatched item tuples

        Returns:
            List of tuples containing (item_paths, theme_name)
        """
        self.logger.info("=== Phase 5: Clustering & Theme Creation ===")

        if not unmatched_items:
            self.logger.info("No unmatched items to cluster")
            return []

        try:
            # Extract embeddings and metadata
            embeddings = [item[1] for item in unmatched_items]
            items_info = [(item[0], item[2]) for item in unmatched_items]

            # Perform clustering
            self.logger.info(f"Clustering {len(unmatched_items)} unmatched items")
            clusters = self.clusterer.cluster_items(embeddings, items_info)

            # Create themes for clusters
            clustered_items = []
            for cluster_id, item_indices in clusters.items():
                cluster_items = [unmatched_items[i] for i in item_indices]

                # Generate theme name
                theme_name = self.theme_matcher.create_new_theme(cluster_items)

                # Get paths for this cluster
                cluster_paths = [item[0] for item in cluster_items]
                clustered_items.append((cluster_paths, theme_name))

                self.stats['new_themes_created'] += 1
                self.logger.info(f"Created theme '{theme_name}' for {len(cluster_paths)} items")

            return clustered_items

        except Exception as e:
            self.logger.error(f"Error during clustering: {e}")
            # Fallback: create a single miscellaneous theme
            all_paths = [item[0] for item in unmatched_items]
            return [(all_paths, "Miscellaneous_Files")]

    def _organization_phase(
        self,
        matched_items: List[Tuple[Path, str, float]],
        clustered_items: List[Tuple[List[Path], str]]
    ) -> None:
        """
        Organization phase - move files to their designated themes.

        Args:
            matched_items: Items matched to existing themes
            clustered_items: Items grouped into new themes
        """
        self.logger.info("=== Phase 6: Organization ===")

        total_moves = len(matched_items) + sum(len(paths) for paths, _ in clustered_items)
        if total_moves == 0:
            self.logger.info("No items to move")
            return

        progress = ProgressTracker(total_moves, "Moving items")

        # Move matched items to existing themes
        for item_path, theme_name, similarity in matched_items:
            success, final_path, error = self.organizer.move_item(
                item_path, self.output_dir, theme_name
            )

            if success:
                self.stats['items_moved'] += 1
            else:
                self.stats['items_skipped'] += 1
                self.logger.warning(f"Failed to move {item_path}: {error}")

            progress.update()

        # Move clustered items to new themes
        for cluster_paths, theme_name in clustered_items:
            for item_path in cluster_paths:
                success, final_path, error = self.organizer.move_item(
                    item_path, self.output_dir, theme_name
                )

                if success:
                    self.stats['items_moved'] += 1
                else:
                    self.stats['items_skipped'] += 1
                    self.logger.warning(f"Failed to move {item_path}: {error}")

                progress.update()

        progress.finish()

        # Update conflict statistics
        conflicts = self.organizer.get_conflicts_log()
        self.stats['conflicts_encountered'] = len(conflicts)

        self.logger.info(f"Organization phase completed")
        operation = "copied" if self.args.copy else "moved"
        self.logger.info(f"Items {operation}: {self.stats['items_moved']}")
        self.logger.info(f"Items skipped: {self.stats['items_skipped']}")
        self.logger.info(f"Conflicts resolved: {self.stats['conflicts_encountered']}")

    def _generate_final_report(self) -> int:
        """
        Generate and display final report.

        Returns:
            Appropriate exit code
        """
        self.logger.info("=== Phase 7: Final Report ===")

        # Ensure end time is set
        if self.stats['end_time'] is None:
            self.stats['end_time'] = time.time()

        # Calculate timing
        duration = self.stats['end_time'] - self.stats['start_time']

        # Generate report
        report = self.organizer.generate_report()

        # Print summary
        print("\n" + "="*50)
        print("SEMANTIC FILE ORGANIZATION REPORT")
        print("="*50)
        print(f"Input Directory: {self.input_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Processing Time: {duration:.2f} seconds")
        print(f"Dry Run Mode: {'Yes' if self.args.dry_run else 'No'}")
        print(f"Copy Mode: {'Yes' if self.args.copy else 'No'}")
        print()

        print("STATISTICS:")
        print(f"  Total Items Found: {self.stats['total_items']}")
        print(f"  Files: {self.stats['files_processed']}")
        print(f"  Folders: {self.stats['folders_processed']}")
        operation = "Copied" if self.args.copy else "Moved"
        print(f"  Items {operation}: {self.stats['items_moved']}")
        print(f"  Items Skipped: {self.stats['items_skipped']}")
        print(f"  Existing Themes Used: {self.stats['existing_themes_used']}")
        print(f"  New Themes Created: {self.stats['new_themes_created']}")
        print(f"  Conflicts Encountered: {self.stats['conflicts_encountered']}")
        print(f"  Errors: {self.stats['errors_encountered']}")
        print()

        # Show conflict details if any
        if self.stats['conflicts_encountered'] > 0:
            print("CONFLICT RESOLUTION SUMMARY:")
            conflicts = self.organizer.get_conflicts_log()
            conflict_actions = {}
            for conflict in conflicts:
                action = conflict.get('action', 'unknown')
                conflict_actions[action] = conflict_actions.get(action, 0) + 1

            for action, count in conflict_actions.items():
                print(f"  {action}: {count}")
            print()

        # Calculate success rate
        if self.stats['total_items'] > 0:
            success_rate = (self.stats['items_moved'] / self.stats['total_items']) * 100
            print(f"SUCCESS RATE: {success_rate:.1f}%")
        else:
            success_rate = 0

        print("="*50)

        # Save detailed log if requested
        if self.args.log_file:
            self.logger.info(f"Detailed logs saved to: {self.args.log_file}")

        # Determine exit code
        if self.stats['errors_encountered'] == 0 and self.stats['items_skipped'] == 0:
            return 0  # Complete success
        elif self.stats['items_moved'] > 0:
            return 1  # Partial success
        else:
            return 2  # Failure


def main() -> int:
    """
    Main entry point for the semantic file organizer.

    Returns:
        Exit code
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle --list-models flag
    if args.list_models:
        from .analyzer import SemanticAnalyzer
        SemanticAnalyzer.list_available_models()
        return 0

    # Validate arguments
    # Check directories are provided when not listing models
    if not args.list_models:
        if not args.input_directory or not args.output_directory:
            parser.error("input_directory and output_directory are required unless using --list-models")

    if args.similarity_threshold < 0.0 or args.similarity_threshold > 1.0:
        print("Error: Similarity threshold must be between 0.0 and 1.0")
        return 2

    if args.min_cluster_size < 1:
        print("Error: Minimum cluster size must be at least 1")
        return 2

    if args.max_file_size < 1:
        print("Error: Maximum file size must be at least 1 MB")
        return 2

    # Confirm destructive operations
    if args.conflict_resolution == 'overwrite' and not args.force and not args.dry_run:
        response = input("WARNING: Overwrite mode will replace existing files. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled by user")
            return 0

    try:
        organizer = SemanticFileOrganizer(args)
        return organizer.run()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 1
    except Exception as e:
        print(f"Fatal error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())