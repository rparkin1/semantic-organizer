"""
Entry point for running the semantic organizer as a module.

This allows the package to be run with: python -m semantic_organizer
"""

from .main import main

if __name__ == "__main__":
    exit(main())