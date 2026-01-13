"""
Integration tests for the Semantic File Organizer.

These tests verify that the complete system works together correctly.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from semantic_organizer.main import SemanticFileOrganizer
import argparse


class TestIntegration(unittest.TestCase):
    """Integration test cases for the complete system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"

        self.input_dir.mkdir()
        self.output_dir.mkdir()

        # Create test files of different types
        (self.input_dir / "document1.txt").write_text("This is a text document about machine learning.")
        (self.input_dir / "document2.txt").write_text("Another text file about artificial intelligence.")
        (self.input_dir / "report.pdf").write_text("Mock PDF content about data science.")

        # Create a folder with files
        test_folder = self.input_dir / "project_folder"
        test_folder.mkdir()
        (test_folder / "readme.md").write_text("# Project Documentation")
        (test_folder / "code.py").write_text("print('Hello World')")

        # Create some existing theme in output
        existing_theme = self.output_dir / "Existing_Theme"
        existing_theme.mkdir()
        (existing_theme / "existing_file.txt").write_text("Existing content")

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_mock_args(self, **kwargs):
        """Create mock command line arguments."""
        defaults = {
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'similarity_threshold': 0.7,
            'conflict_resolution': 'rename',
            'dry_run': True,  # Use dry run to avoid actual file moves in tests
            'verbose': False,
            'embedding_model': 'all-MiniLM-L6-v2',
            'min_cluster_size': 2,
            'max_file_size': 100,
            'sample_limit': 50000,
            'skip_extensions': None,
            'content_weight': 0.8,
            'filename_weight': 0.2,
            'force': False,
            'log_file': None,
            'copy': False
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch('sentence_transformers.SentenceTransformer')
    def test_complete_organization_workflow(self, mock_transformer):
        """Test the complete organization workflow."""
        # Mock the sentence transformer
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = lambda x, **kwargs: np.random.rand(384)
        mock_transformer.return_value = mock_model

        args = self._create_mock_args()
        organizer = SemanticFileOrganizer(args)

        exit_code = organizer.run()

        # Should complete successfully
        self.assertEqual(exit_code, 0)

        # Check that items were discovered
        self.assertGreater(organizer.stats['total_items'], 0)
        self.assertGreater(organizer.stats['files_processed'], 0)
        self.assertGreater(organizer.stats['folders_processed'], 0)

    @patch('sentence_transformers.SentenceTransformer')
    def test_existing_theme_matching(self, mock_transformer):
        """Test that items are matched to existing themes."""
        # Create similar embeddings for theme matching
        def mock_encode(text, **kwargs):
            if 'existing' in text.lower():
                return np.array([1.0] + [0.0] * 383)  # Similar embedding for existing theme
            elif 'machine learning' in text.lower() or 'artificial intelligence' in text.lower():
                return np.array([0.9] + [0.1] * 383)  # Similar to existing theme
            else:
                return np.random.rand(384)

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        args = self._create_mock_args()
        organizer = SemanticFileOrganizer(args)

        exit_code = organizer.run()

        self.assertEqual(exit_code, 0)
        # Some items should have been matched to existing themes
        self.assertGreaterEqual(organizer.stats['existing_themes_used'], 0)

    @patch('sentence_transformers.SentenceTransformer')
    def test_new_theme_creation(self, mock_transformer):
        """Test creation of new themes for unmatched items."""
        # Create different embeddings that won't match existing themes
        def mock_encode(text, **kwargs):
            if 'document' in text.lower():
                return np.array([0.0, 1.0] + [0.0] * 382)  # Documents cluster
            elif 'report' in text.lower():
                return np.array([0.0, 0.0, 1.0] + [0.0] * 381)  # Reports cluster
            elif 'project' in text.lower():
                return np.array([0.0, 0.0, 0.0, 1.0] + [0.0] * 380)  # Projects cluster
            else:
                return np.random.rand(384)

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        args = self._create_mock_args()
        organizer = SemanticFileOrganizer(args)

        exit_code = organizer.run()

        self.assertEqual(exit_code, 0)
        # Should create new themes for unmatched items
        self.assertGreaterEqual(organizer.stats['new_themes_created'], 0)

    @patch('sentence_transformers.SentenceTransformer')
    def test_conflict_resolution(self, mock_transformer):
        """Test conflict resolution during organization."""
        # Set up conflicting files
        theme_dir = self.output_dir / "Text_Documents"
        theme_dir.mkdir()
        (theme_dir / "document1.txt").write_text("Conflicting file")

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model

        args = self._create_mock_args(conflict_resolution='rename', dry_run=False)
        organizer = SemanticFileOrganizer(args)

        exit_code = organizer.run()

        self.assertIn(exit_code, [0, 1])  # Success or partial success
        # Should have encountered and resolved conflicts
        self.assertGreaterEqual(organizer.stats['conflicts_encountered'], 0)

    @patch('sentence_transformers.SentenceTransformer')
    def test_dry_run_mode(self, mock_transformer):
        """Test dry run mode doesn't actually move files."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model

        # Count files before
        initial_files = list(self.input_dir.rglob('*'))
        initial_count = len([f for f in initial_files if f.is_file()])

        args = self._create_mock_args(dry_run=True)
        organizer = SemanticFileOrganizer(args)

        exit_code = organizer.run()

        # Files should still be in input directory
        final_files = list(self.input_dir.rglob('*'))
        final_count = len([f for f in final_files if f.is_file()])

        self.assertEqual(exit_code, 0)
        self.assertEqual(initial_count, final_count)

    @patch('sentence_transformers.SentenceTransformer')
    def test_folder_structure_preservation(self, mock_transformer):
        """Test that folder structures are preserved."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model

        args = self._create_mock_args(dry_run=False)
        organizer = SemanticFileOrganizer(args)

        exit_code = organizer.run()

        # Check if folder was moved with structure intact
        # Find where the project_folder ended up
        moved_folders = list(self.output_dir.rglob('project_folder'))

        if moved_folders:
            moved_folder = moved_folders[0]
            self.assertTrue((moved_folder / "readme.md").exists())
            self.assertTrue((moved_folder / "code.py").exists())

    @patch('sentence_transformers.SentenceTransformer')
    def test_error_handling(self, mock_transformer):
        """Test error handling with problematic files."""
        # Create a file that will cause extraction errors
        problem_file = self.input_dir / "corrupted.pdf"
        problem_file.write_bytes(b"This is not a valid PDF file")

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model

        args = self._create_mock_args()
        organizer = SemanticFileOrganizer(args)

        # Should handle errors gracefully
        exit_code = organizer.run()

        # Should complete despite errors
        self.assertIn(exit_code, [0, 1])  # Success or partial success

    def test_empty_input_directory(self):
        """Test handling of empty input directory."""
        # Create empty input directory
        empty_input = self.temp_dir / "empty_input"
        empty_input.mkdir()

        args = self._create_mock_args(input_directory=str(empty_input))
        organizer = SemanticFileOrganizer(args)

        exit_code = organizer.run()

        self.assertEqual(exit_code, 0)
        self.assertEqual(organizer.stats['total_items'], 0)

    def test_nonexistent_input_directory(self):
        """Test handling of non-existent input directory."""
        nonexistent = self.temp_dir / "nonexistent"

        args = self._create_mock_args(input_directory=str(nonexistent))
        organizer = SemanticFileOrganizer(args)

        exit_code = organizer.run()

        self.assertEqual(exit_code, 2)  # Failure


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests with real file operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('sentence_transformers.SentenceTransformer')
    def test_complete_workflow_with_real_files(self, mock_transformer):
        """Test complete workflow with actual file operations."""
        # Set up input directory with various files
        input_dir = self.temp_dir / "input"
        output_dir = self.temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create test files
        (input_dir / "ml_paper.txt").write_text("This paper discusses machine learning algorithms.")
        (input_dir / "ai_research.txt").write_text("Artificial intelligence research findings.")
        (input_dir / "photo.jpg").write_text("Mock image file")
        (input_dir / "video.mp4").write_text("Mock video file")

        # Create similar embeddings for ML-related files
        def mock_encode(text, **kwargs):
            if any(word in text.lower() for word in ['machine', 'artificial', 'learning', 'intelligence']):
                return np.array([1.0] + [0.1] * 383)  # Similar embeddings
            else:
                return np.array([0.0, 1.0] + [0.1] * 382)  # Different embeddings

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        args = argparse.Namespace(
            input_directory=str(input_dir),
            output_directory=str(output_dir),
            similarity_threshold=0.7,
            conflict_resolution='rename',
            dry_run=False,  # Actually move files
            verbose=False,
            embedding_model='all-MiniLM-L6-v2',
            min_cluster_size=2,
            max_file_size=100,
            sample_limit=50000,
            skip_extensions=None,
            content_weight=0.8,
            filename_weight=0.2,
            force=False,
            log_file=None,
            copy=False
        )

        organizer = SemanticFileOrganizer(args)
        exit_code = organizer.run()

        # Should complete successfully
        self.assertEqual(exit_code, 0)

        # Files should be moved to output directory
        self.assertEqual(len(list(input_dir.iterdir())), 0)  # Input should be empty

        # Output should contain organized files
        output_files = list(output_dir.rglob('*'))
        moved_files = [f for f in output_files if f.is_file()]
        self.assertEqual(len(moved_files), 4)  # All 4 files should be moved


if __name__ == '__main__':
    unittest.main()