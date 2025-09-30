"""
Theme matching module for identifying existing themes and matching items to them.

This module handles the logic for scanning existing output directories,
matching items to existing themes, and creating new themes when needed.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from .analyzer import SemanticAnalyzer

logger = logging.getLogger('semantic_organizer.theme_matcher')


class ThemeMatcher:
    """Class for matching items to existing themes and creating new ones."""

    def __init__(
        self,
        analyzer: SemanticAnalyzer,
        similarity_threshold: float = 0.7,
        min_cluster_size: int = 2
    ):
        """
        Initialize the ThemeMatcher.

        Args:
            analyzer: SemanticAnalyzer instance for generating embeddings
            similarity_threshold: Minimum similarity for theme matching (0-1)
            min_cluster_size: Minimum items needed to form a new theme
        """
        self.analyzer = analyzer
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

        # Storage for existing themes
        self.existing_themes: Dict[str, Dict] = {}
        self.theme_embeddings: Dict[str, np.ndarray] = {}

    def load_existing_themes(self, output_dir: Path) -> Dict[str, Dict]:
        """
        Scan output directory for existing theme folders.

        Args:
            output_dir: Path to the output directory

        Returns:
            Dictionary mapping theme names to theme information
        """
        logger.info(f"Scanning output directory for existing themes: {output_dir}")

        themes = {}
        theme_embeddings = {}

        try:
            if not output_dir.exists():
                logger.info("Output directory does not exist, no existing themes found")
                return themes

            # Scan for subdirectories (these are theme folders)
            for item in output_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    theme_name = item.name

                    # Generate embedding for the theme folder
                    try:
                        embedding, metadata = self.analyzer.analyze_folder(item)

                        themes[theme_name] = {
                            'path': item,
                            'name': theme_name,
                            'embedding': embedding,
                            'file_count': len(list(item.glob('**/*'))) if item.exists() else 0,
                            'metadata': metadata
                        }

                        theme_embeddings[theme_name] = embedding
                        logger.debug(f"Loaded existing theme: {theme_name} ({themes[theme_name]['file_count']} items)")

                    except Exception as e:
                        logger.warning(f"Could not analyze existing theme folder {theme_name}: {e}")
                        # Create a basic theme entry with just the name
                        theme_embedding = self.analyzer.generate_embedding(self._clean_theme_name(theme_name))
                        themes[theme_name] = {
                            'path': item,
                            'name': theme_name,
                            'embedding': theme_embedding,
                            'file_count': 0,
                            'metadata': {'analysis_method': 'name_only', 'error': str(e)}
                        }
                        theme_embeddings[theme_name] = theme_embedding

            self.existing_themes = themes
            self.theme_embeddings = theme_embeddings

            logger.info(f"Found {len(themes)} existing themes")
            return themes

        except Exception as e:
            logger.error(f"Error scanning output directory {output_dir}: {e}")
            return {}

    def match_to_theme(self, item_embedding: np.ndarray, item_path: Path) -> Optional[Tuple[str, float]]:
        """
        Find the best matching existing theme for an item.

        Args:
            item_embedding: Semantic embedding of the item
            item_path: Path to the item (for logging)

        Returns:
            Tuple of (theme_name, similarity_score) if match found, None otherwise
        """
        if not self.existing_themes:
            logger.debug(f"No existing themes to match against for {item_path}")
            return None

        best_theme = None
        best_similarity = 0.0

        for theme_name, theme_info in self.existing_themes.items():
            try:
                theme_embedding = theme_info['embedding']
                similarity = self.calculate_similarity(item_embedding, theme_embedding)

                logger.debug(f"Similarity between {item_path.name} and theme '{theme_name}': {similarity:.3f}")

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_theme = theme_name

            except Exception as e:
                logger.warning(f"Error calculating similarity with theme {theme_name}: {e}")
                continue

        # Check if best match meets threshold
        if best_theme and best_similarity >= self.similarity_threshold:
            logger.debug(f"Matched {item_path.name} to existing theme '{best_theme}' (similarity: {best_similarity:.3f})")
            return best_theme, best_similarity
        else:
            logger.debug(f"No theme match found for {item_path.name} (best: {best_similarity:.3f}, threshold: {self.similarity_threshold})")
            return None

    def create_new_theme(self, items: List[Tuple[Path, np.ndarray, Dict]], base_name: Optional[str] = None) -> str:
        """
        Generate a new theme name from clustered items.

        Args:
            items: List of tuples containing (path, embedding, metadata) for items in the cluster
            base_name: Optional base name for the theme

        Returns:
            Generated theme name
        """
        if not items:
            return base_name or "Miscellaneous_Files"

        try:
            # Extract common patterns from the items
            paths = [item[0] for item in items]
            theme_name = self._generate_theme_name_from_items(paths, base_name)

            # Ensure the name doesn't conflict with existing themes
            theme_name = self._ensure_unique_theme_name(theme_name)

            logger.info(f"Created new theme: '{theme_name}' for {len(items)} items")
            return theme_name

        except Exception as e:
            logger.error(f"Error creating new theme name: {e}")
            fallback_name = base_name or f"Theme_{len(self.existing_themes) + 1}"
            return self._ensure_unique_theme_name(fallback_name)

    def _generate_theme_name_from_items(self, paths: List[Path], base_name: Optional[str] = None) -> str:
        """
        Generate a descriptive theme name based on the items in the cluster.

        Args:
            paths: List of paths in the cluster
            base_name: Optional base name to use

        Returns:
            Generated theme name
        """
        if base_name:
            return self._clean_theme_name(base_name)

        # Analyze file extensions
        extensions = []
        for path in paths:
            if path.is_file() and path.suffix:
                ext = path.suffix.lower()
                extensions.append(ext)

        # Analyze path components for common keywords (only filename, not full path)
        all_parts = []
        for path in paths:
            # Only use the filename stem, not the full path
            name_parts = path.stem.replace('_', ' ').replace('-', ' ').split()
            all_parts.extend(name_parts)

        # Count common words and extensions
        word_counts = {}
        for part in all_parts:
            words = part.lower().split()
            for word in words:
                if len(word) > 2:  # Skip very short words
                    word_counts[word] = word_counts.get(word, 0) + 1

        ext_counts = {}
        for ext in extensions:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        # Generate theme name based on patterns - prioritize semantic content over file extensions
        theme_parts = []

        # First, add most common meaningful words (semantic content is priority)
        if word_counts:
            # Filter out common words that aren't useful for themes
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below', 'between'}
            meaningful_words = {word: count for word, count in word_counts.items() if word not in stop_words and len(word) > 2}

            if meaningful_words:
                # Get top 3 most common words for better theme description
                top_words = sorted(meaningful_words.items(), key=lambda x: x[1], reverse=True)[:3]
                for word, count in top_words:
                    if count >= 2:  # Word appears in at least 2 items
                        theme_parts.append(word.title())

        # Only add file extension if no meaningful semantic words found
        if not theme_parts and ext_counts:
            # Only use extension for non-text files or when files are truly mixed
            most_common_ext = max(ext_counts, key=ext_counts.get)
            # Skip generic text extensions in favor of semantic content
            if most_common_ext not in ['.txt', '.md']:
                ext_name = self._extension_to_name(most_common_ext)
                if ext_name:
                    theme_parts.append(ext_name)

        # Create theme name
        if theme_parts:
            theme_name = '_'.join(theme_parts)
        else:
            # Fallback based on file types or general category
            if extensions:
                dominant_ext = max(ext_counts, key=ext_counts.get)
                ext_name = self._extension_to_name(dominant_ext)
                theme_name = ext_name or "Mixed_Files"
            else:
                # Must be folders
                theme_name = "Folders"

        return self._clean_theme_name(theme_name)

    def _extension_to_name(self, extension: str) -> Optional[str]:
        """
        Convert file extension to a descriptive name.

        Args:
            extension: File extension (including the dot)

        Returns:
            Descriptive name or None
        """
        ext_mapping = {
            '.txt': 'Text_Documents',
            '.md': 'Markdown_Files',
            '.doc': 'Word_Documents',
            '.docx': 'Word_Documents',
            '.pdf': 'PDF_Documents',
            '.xlsx': 'Spreadsheets',
            '.xls': 'Spreadsheets',
            '.pptx': 'Presentations',
            '.ppt': 'Presentations',
            '.jpg': 'Images',
            '.jpeg': 'Images',
            '.png': 'Images',
            '.gif': 'Images',
            '.mp4': 'Videos',
            '.avi': 'Videos',
            '.mov': 'Videos',
            '.mp3': 'Audio',
            '.wav': 'Audio',
            '.zip': 'Archives',
            '.rar': 'Archives',
            '.tar': 'Archives',
            '.py': 'Python_Scripts',
            '.js': 'JavaScript_Files',
            '.html': 'Web_Files',
            '.css': 'Web_Files',
            '.json': 'Data_Files',
            '.xml': 'Data_Files',
            '.csv': 'Data_Files'
        }
        return ext_mapping.get(extension.lower())

    def _clean_theme_name(self, name: str) -> str:
        """
        Clean and normalize a theme name.

        Args:
            name: Raw theme name

        Returns:
            Cleaned theme name suitable for use as a directory name
        """
        if not name:
            return "Untitled_Theme"

        # Replace problematic characters including forward slash
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', name)

        # Replace spaces and multiple underscores with single underscores
        clean_name = re.sub(r'[\s_]+', '_', clean_name)

        # Remove leading/trailing underscores and dots
        clean_name = clean_name.strip('_.')

        # Split on underscores and filter out empty parts and very short parts
        parts = [part for part in clean_name.split('_') if part and len(part) > 1]

        # If we have parts, use them; otherwise use a default
        if parts:
            clean_name = '_'.join(part.capitalize() for part in parts[:4])  # Limit to 4 parts max
        else:
            clean_name = "Untitled_Theme"

        # Ensure it's not empty and limit length
        if not clean_name or len(clean_name) < 2:
            clean_name = "Untitled_Theme"
        elif len(clean_name) > 50:
            clean_name = clean_name[:50].rstrip('_')

        return clean_name

    def _ensure_unique_theme_name(self, name: str) -> str:
        """
        Ensure theme name doesn't conflict with existing themes.

        Args:
            name: Proposed theme name

        Returns:
            Unique theme name
        """
        original_name = name
        counter = 1

        while name in self.existing_themes:
            name = f"{original_name}_{counter}"
            counter += 1

        return name

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Ensure embeddings are numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)

            # Handle zero vectors
            if np.allclose(emb1, 0) or np.allclose(emb2, 0):
                return 0.0

            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Ensure result is in valid range [0, 1]
            similarity = max(0.0, min(1.0, (similarity + 1) / 2))  # Convert from [-1,1] to [0,1]

            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def add_theme(self, theme_name: str, theme_path: Path) -> None:
        """
        Add a new theme to the existing themes collection.

        Args:
            theme_name: Name of the theme
            theme_path: Path to the theme directory
        """
        try:
            embedding, metadata = self.analyzer.analyze_folder(theme_path)

            self.existing_themes[theme_name] = {
                'path': theme_path,
                'name': theme_name,
                'embedding': embedding,
                'file_count': 0,
                'metadata': metadata
            }

            self.theme_embeddings[theme_name] = embedding
            logger.debug(f"Added new theme to collection: {theme_name}")

        except Exception as e:
            logger.error(f"Error adding theme {theme_name}: {e}")

    def get_theme_info(self, theme_name: str) -> Optional[Dict]:
        """
        Get information about a specific theme.

        Args:
            theme_name: Name of the theme

        Returns:
            Theme information dictionary or None if not found
        """
        return self.existing_themes.get(theme_name)

    def list_themes(self) -> List[str]:
        """
        Get list of all existing theme names.

        Returns:
            List of theme names
        """
        return list(self.existing_themes.keys())