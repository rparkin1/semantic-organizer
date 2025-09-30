"""
Tests for the FileExtractor module.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from semantic_organizer.file_extractors import FileExtractor


class TestFileExtractor(unittest.TestCase):
    """Test cases for FileExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FileExtractor()
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_get_file_type(self):
        """Test file type detection."""
        self.assertEqual(self.extractor.get_file_type(Path("test.txt")), "txt")
        self.assertEqual(self.extractor.get_file_type(Path("test.pdf")), "pdf")
        self.assertEqual(self.extractor.get_file_type(Path("test.docx")), "docx")
        self.assertEqual(self.extractor.get_file_type(Path("test.xlsx")), "xlsx")
        self.assertEqual(self.extractor.get_file_type(Path("test.pptx")), "pptx")
        self.assertEqual(self.extractor.get_file_type(Path("test.unknown")), "unknown")

    def test_extract_from_text_file(self):
        """Test text file extraction."""
        # Create a test text file
        test_file = self.temp_dir / "test.txt"
        test_content = "This is a test text file with some content."

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        text, metadata = self.extractor.extract_text(test_file)

        self.assertIn(test_content, text)
        self.assertEqual(metadata['file_type'], 'txt')
        self.assertEqual(metadata['extraction_method'], 'text_reader')
        self.assertTrue(metadata['content_available'])

    def test_extract_from_nonexistent_file(self):
        """Test extraction from non-existent file."""
        non_existent = self.temp_dir / "nonexistent.txt"

        text, metadata = self.extractor.extract_text(non_existent)

        # Should fall back to filename-based extraction
        self.assertIn("nonexistent", text.lower())
        self.assertEqual(metadata['extraction_method'], 'filename_only')
        self.assertFalse(metadata['content_available'])

    def test_file_size_limit(self):
        """Test file size limit enforcement."""
        # Create a large test file
        large_file = self.temp_dir / "large.txt"

        with open(large_file, 'w') as f:
            # Write content larger than the limit (simulate by setting a small limit)
            f.write("x" * 1000)

        # Create extractor with small size limit
        small_limit_extractor = FileExtractor(max_file_size_mb=0.001)  # Very small limit

        text, metadata = small_limit_extractor.extract_text(large_file)

        # Should fall back to filename extraction
        self.assertIn("large", text.lower())
        self.assertEqual(metadata['extraction_method'], 'filename_only')

    @patch('semantic_organizer.file_extractors.chardet.detect')
    def test_encoding_detection(self, mock_detect):
        """Test text encoding detection."""
        mock_detect.return_value = {'encoding': 'utf-8', 'confidence': 0.99}

        test_file = self.temp_dir / "encoded.txt"
        test_content = "Test content with special characters: åäö"

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        text, metadata = self.extractor.extract_text(test_file)

        self.assertIn("special characters", text)
        mock_detect.assert_called_once()

    def test_fallback_extraction(self):
        """Test fallback extraction method."""
        test_path = Path("test_file.pdf")

        text, metadata = self.extractor._fallback_extraction(test_path)

        self.assertIn("test file", text.lower())
        self.assertEqual(metadata['extraction_method'], 'filename_only')
        self.assertFalse(metadata['content_available'])

    def test_sample_limit(self):
        """Test content sampling for large files."""
        extractor = FileExtractor(sample_limit=50)  # Very small limit

        test_file = self.temp_dir / "long.txt"
        long_content = "This is a very long text content. " * 10

        with open(test_file, 'w') as f:
            f.write(long_content)

        text, metadata = extractor.extract_text(test_file)

        # Text should be truncated
        self.assertTrue(len(text) <= 60)  # Account for filename prefix
        self.assertTrue(metadata.get('truncated', False))


if __name__ == '__main__':
    unittest.main()