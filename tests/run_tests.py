"""
Test runner for the Semantic File Organizer test suite.
"""

import unittest
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the semantic_organizer package
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == '__main__':
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(str(start_dir), pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(not result.wasSuccessful())