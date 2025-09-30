"""
Tests for the FileOrganizer module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from semantic_organizer.organizer import FileOrganizer


class TestFileOrganizer(unittest.TestCase):
    """Test cases for FileOrganizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.source_dir = self.temp_dir / "source"
        self.dest_dir = self.temp_dir / "destination"

        self.source_dir.mkdir()
        self.dest_dir.mkdir()

        # Create test files
        (self.source_dir / "test1.txt").write_text("Test file 1")
        (self.source_dir / "test2.txt").write_text("Test file 2")
        (self.source_dir / "test_folder").mkdir()
        (self.source_dir / "test_folder" / "nested.txt").write_text("Nested file")

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_create_theme_folder(self):
        """Test theme folder creation."""
        organizer = FileOrganizer()

        theme_dir = organizer.create_theme_folder(self.dest_dir, "Test_Theme")

        self.assertTrue(theme_dir.exists())
        self.assertEqual(theme_dir.name, "Test_Theme")

    def test_create_theme_folder_dry_run(self):
        """Test theme folder creation in dry run mode."""
        organizer = FileOrganizer(dry_run=True)

        theme_dir = organizer.create_theme_folder(self.dest_dir, "Test_Theme")

        # Directory should not actually be created
        self.assertFalse(theme_dir.exists())

    def test_generate_unique_filename_no_conflict(self):
        """Test unique filename generation with no conflicts."""
        organizer = FileOrganizer()

        unique_path = organizer.generate_unique_filename(self.dest_dir, "test.txt")

        self.assertEqual(unique_path.name, "test.txt")

    def test_generate_unique_filename_with_conflict(self):
        """Test unique filename generation with conflicts."""
        organizer = FileOrganizer()

        # Create conflicting file
        (self.dest_dir / "test.txt").write_text("Existing file")

        unique_path = organizer.generate_unique_filename(self.dest_dir, "test.txt")

        self.assertEqual(unique_path.name, "test_1.txt")

    def test_generate_unique_filename_multiple_conflicts(self):
        """Test unique filename generation with multiple conflicts."""
        organizer = FileOrganizer()

        # Create multiple conflicting files
        (self.dest_dir / "test.txt").write_text("Existing file")
        (self.dest_dir / "test_1.txt").write_text("Existing file 1")
        (self.dest_dir / "test_3.txt").write_text("Existing file 3")

        unique_path = organizer.generate_unique_filename(self.dest_dir, "test.txt")

        # Should find the next available number (2 is missing, but we take the next after max)
        self.assertEqual(unique_path.name, "test_4.txt")

    def test_get_next_available_number(self):
        """Test finding next available counter number."""
        organizer = FileOrganizer()

        # Create files with various numbers
        (self.dest_dir / "doc.pdf").write_text("Original")
        (self.dest_dir / "doc_1.pdf").write_text("Copy 1")
        (self.dest_dir / "doc_3.pdf").write_text("Copy 3")
        (self.dest_dir / "doc_10.pdf").write_text("Copy 10")

        next_num = organizer.get_next_available_number(self.dest_dir, "doc", ".pdf")

        # Should return 11 (next after highest existing number)
        self.assertEqual(next_num, 11)

    def test_apply_timestamp_suffix(self):
        """Test timestamp suffix application."""
        organizer = FileOrganizer()

        test_path = self.dest_dir / "document.pdf"
        timestamped_path = organizer.apply_timestamp_suffix(test_path)

        # Should contain timestamp in the name
        self.assertTrue("_20" in timestamped_path.name)  # Year should be in name
        self.assertTrue(timestamped_path.name.endswith(".pdf"))

    def test_handle_conflict_rename_mode(self):
        """Test conflict handling in rename mode."""
        organizer = FileOrganizer(conflict_resolution="rename")

        source = self.source_dir / "test1.txt"
        destination = self.dest_dir / "test1.txt"

        # Create conflicting destination
        destination.write_text("Existing content")

        resolved_path, action = organizer.handle_conflict(source, destination, "rename")

        self.assertEqual(action, "renamed")
        self.assertEqual(resolved_path.name, "test1_1.txt")

    def test_handle_conflict_skip_mode(self):
        """Test conflict handling in skip mode."""
        organizer = FileOrganizer(conflict_resolution="skip")

        source = self.source_dir / "test1.txt"
        destination = self.dest_dir / "test1.txt"

        # Create conflicting destination
        destination.write_text("Existing content")

        resolved_path, action = organizer.handle_conflict(source, destination, "skip")

        self.assertEqual(action, "skipped")
        self.assertIsNone(resolved_path)

    def test_handle_conflict_overwrite_mode(self):
        """Test conflict handling in overwrite mode."""
        organizer = FileOrganizer(conflict_resolution="overwrite")

        source = self.source_dir / "test1.txt"
        destination = self.dest_dir / "test1.txt"

        # Create conflicting destination
        destination.write_text("Existing content")

        resolved_path, action = organizer.handle_conflict(source, destination, "overwrite")

        self.assertEqual(action, "overwritten")
        self.assertEqual(resolved_path, destination)

    def test_validate_move_operation_valid(self):
        """Test move validation for valid operation."""
        organizer = FileOrganizer()

        source = self.source_dir / "test1.txt"
        destination = self.dest_dir / "moved_test1.txt"

        error = organizer.validate_move_operation(source, destination)

        self.assertIsNone(error)

    def test_validate_move_operation_nonexistent_source(self):
        """Test move validation for non-existent source."""
        organizer = FileOrganizer()

        source = self.source_dir / "nonexistent.txt"
        destination = self.dest_dir / "moved.txt"

        error = organizer.validate_move_operation(source, destination)

        self.assertIsNotNone(error)
        self.assertIn("does not exist", error)

    def test_move_item_success(self):
        """Test successful item moving."""
        organizer = FileOrganizer()

        source = self.source_dir / "test1.txt"

        success, final_path, error = organizer.move_item(source, self.dest_dir, "Text_Files")

        self.assertTrue(success)
        self.assertIsNotNone(final_path)
        self.assertIsNone(error)
        self.assertTrue(final_path.exists())
        self.assertFalse(source.exists())

    def test_move_item_dry_run(self):
        """Test item moving in dry run mode."""
        organizer = FileOrganizer(dry_run=True)

        source = self.source_dir / "test1.txt"
        original_content = source.read_text()

        success, final_path, error = organizer.move_item(source, self.dest_dir, "Text_Files")

        self.assertTrue(success)
        # Source should still exist in dry run mode
        self.assertTrue(source.exists())
        self.assertEqual(source.read_text(), original_content)

    def test_move_folder_preserve_structure(self):
        """Test folder moving with structure preservation."""
        organizer = FileOrganizer()

        source_folder = self.source_dir / "test_folder"

        success, final_path, error = organizer.move_item(source_folder, self.dest_dir, "Folders")

        self.assertTrue(success)
        # Check that nested structure is preserved
        nested_file = final_path / "nested.txt"
        self.assertTrue(nested_file.exists())

    def test_operations_logging(self):
        """Test that operations are properly logged."""
        organizer = FileOrganizer()

        source = self.source_dir / "test1.txt"
        organizer.move_item(source, self.dest_dir, "Text_Files")

        operations_log = organizer.get_operations_log()

        self.assertEqual(len(operations_log), 1)
        self.assertEqual(operations_log[0]['action'], 'move')
        self.assertIn('timestamp', operations_log[0])

    def test_conflicts_logging(self):
        """Test that conflicts are properly logged."""
        organizer = FileOrganizer(conflict_resolution="rename")

        # Create theme directory and conflicting file
        theme_dir = self.dest_dir / "Text_Files"
        theme_dir.mkdir()
        (theme_dir / "test1.txt").write_text("Existing file")

        source = self.source_dir / "test1.txt"
        organizer.move_item(source, self.dest_dir, "Text_Files")

        conflicts_log = organizer.get_conflicts_log()

        self.assertEqual(len(conflicts_log), 1)
        self.assertEqual(conflicts_log[0]['action'], 'renamed')

    def test_generate_report(self):
        """Test report generation."""
        organizer = FileOrganizer()

        source = self.source_dir / "test1.txt"
        organizer.move_item(source, self.dest_dir, "Text_Files")

        report = organizer.generate_report()

        self.assertIn('total_operations', report)
        self.assertIn('successful_moves', report)
        self.assertIn('operations_log', report)
        self.assertEqual(report['successful_moves'], 1)


if __name__ == '__main__':
    unittest.main()