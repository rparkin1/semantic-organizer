"""
Tests for the SemanticAnalyzer module.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from semantic_organizer.analyzer import SemanticAnalyzer


class TestSemanticAnalyzer(unittest.TestCase):
    """Test cases for SemanticAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the sentence transformer to avoid downloading models in tests
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            # Create a mock model that returns predictable embeddings
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.random.rand(384)
            mock_transformer.return_value = mock_model

            self.analyzer = SemanticAnalyzer()

        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_preprocess_text(self):
        """Test text preprocessing."""
        raw_text = "This is    a   test with\n\nexcessive   whitespace.\n\n\n"
        processed = self.analyzer.preprocess_text(raw_text)

        # Should normalize whitespace
        self.assertNotIn("    ", processed)
        self.assertNotIn("\n\n", processed)

    def test_preprocess_text_with_urls(self):
        """Test URL replacement in text preprocessing."""
        text_with_url = "Visit our website at https://example.com for more info."
        processed = self.analyzer.preprocess_text(text_with_url)

        self.assertIn("[URL]", processed)
        self.assertNotIn("https://example.com", processed)

    def test_preprocess_text_with_dates(self):
        """Test date replacement in preprocessing."""
        text_with_date = "The meeting is on 2023-12-25 at 3 PM."
        processed = self.analyzer.preprocess_text(text_with_date)

        self.assertIn("[DATE]", processed)
        self.assertNotIn("2023-12-25", processed)

    def test_generate_embedding(self):
        """Test embedding generation."""
        test_text = "This is a test document about machine learning."
        embedding = self.analyzer.generate_embedding(test_text)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 384)  # Mock model dimension

    def test_generate_embedding_empty_text(self):
        """Test embedding generation for empty text."""
        embedding = self.analyzer.generate_embedding("")

        self.assertIsInstance(embedding, np.ndarray)
        self.assertTrue(np.allclose(embedding, 0))  # Should be zero vector

    def test_analyze_file(self):
        """Test file analysis."""
        # Create a test file
        test_file = self.temp_dir / "test.txt"
        with open(test_file, 'w') as f:
            f.write("This is a test file about artificial intelligence.")

        embedding, metadata = self.analyzer.analyze_file(test_file)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(metadata['analysis_method'], 'file_content')
        self.assertIn('text_length', metadata)

    def test_analyze_folder(self):
        """Test folder analysis."""
        # Create a test folder with files
        test_folder = self.temp_dir / "test_folder"
        test_folder.mkdir()

        (test_folder / "file1.txt").write_text("Document about machine learning")
        (test_folder / "file2.txt").write_text("Paper on neural networks")

        embedding, metadata = self.analyzer.analyze_folder(test_folder)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(metadata['analysis_method'], 'folder_sampling')
        self.assertIn('sampled_files', metadata)
        self.assertEqual(metadata['total_files'], 2)

    def test_analyze_empty_folder(self):
        """Test analysis of empty folder."""
        empty_folder = self.temp_dir / "empty_folder"
        empty_folder.mkdir()

        embedding, metadata = self.analyzer.analyze_folder(empty_folder)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(metadata['total_files'], 0)

    def test_fallback_text_generation(self):
        """Test fallback text generation from path."""
        test_path = Path("machine_learning_paper.pdf")
        fallback_text = self.analyzer._get_fallback_text(test_path)

        self.assertIn("machine learning paper", fallback_text.lower())
        self.assertIn("pdf file", fallback_text.lower())

    def test_caching(self):
        """Test embedding caching."""
        test_file = self.temp_dir / "cached_test.txt"
        with open(test_file, 'w') as f:
            f.write("Test content for caching.")

        # First analysis should cache the result
        embedding1, metadata1 = self.analyzer.analyze_file(test_file)

        # Second analysis should use cache
        embedding2, metadata2 = self.analyzer.analyze_file(test_file)

        np.testing.assert_array_equal(embedding1, embedding2)
        self.assertEqual(metadata2['analysis_method'], 'cached')

    def test_cache_management(self):
        """Test cache size and clearing."""
        initial_size = self.analyzer.get_cache_size()

        test_file = self.temp_dir / "cache_test.txt"
        with open(test_file, 'w') as f:
            f.write("Test content")

        self.analyzer.analyze_file(test_file)
        self.assertEqual(self.analyzer.get_cache_size(), initial_size + 1)

        self.analyzer.clear_cache()
        self.assertEqual(self.analyzer.get_cache_size(), 0)


if __name__ == '__main__':
    unittest.main()