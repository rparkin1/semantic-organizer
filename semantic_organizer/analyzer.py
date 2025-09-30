"""
Semantic analysis module for generating embeddings and processing text content.

This module provides the core semantic analysis functionality using embeddings
to understand and compare the semantic meaning of files and folders.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .file_extractors import FileExtractor

logger = logging.getLogger('semantic_organizer.analyzer')


class SemanticAnalyzer:
    """Main semantic analysis class for generating embeddings and processing text."""

    # Available embedding models with their characteristics
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "description": "Lightweight, fast model (23MB, 384 dimensions)",
            "size": "small",
            "speed": "fast",
            "quality": "good"
        },
        "all-mpnet-base-v2": {
            "description": "High quality model (438MB, 768 dimensions)",
            "size": "large",
            "speed": "medium",
            "quality": "excellent"
        },
        "intfloat/e5-base": {
            "description": "E5 base model (435MB, 768 dimensions)",
            "size": "large",
            "speed": "medium",
            "quality": "excellent"
        },
        "intfloat/e5-small": {
            "description": "E5 small model (134MB, 384 dimensions)",
            "size": "medium",
            "speed": "fast",
            "quality": "very good"
        },
        "intfloat/e5-large": {
            "description": "E5 large model (1.3GB, 1024 dimensions)",
            "size": "very large",
            "speed": "slow",
            "quality": "outstanding"
        },
        "sentence-transformers/all-roberta-large-v1": {
            "description": "RoBERTa large model (1.4GB, 1024 dimensions)",
            "size": "very large",
            "speed": "slow",
            "quality": "outstanding"
        },
        "sentence-transformers/all-MiniLM-L12-v2": {
            "description": "SBERT MiniLM L12 model (134MB, 384 dimensions)",
            "size": "medium",
            "speed": "fast",
            "quality": "very good"
        },
        "sentence-transformers/all-distilroberta-v1": {
            "description": "SBERT DistilRoBERTa model (290MB, 768 dimensions)",
            "size": "large",
            "speed": "medium",
            "quality": "very good"
        },
        "sentence-transformers/paraphrase-MiniLM-L6-v2": {
            "description": "SBERT paraphrase MiniLM model (23MB, 384 dimensions)",
            "size": "small",
            "speed": "fast",
            "quality": "good"
        },
        "sentence-transformers/paraphrase-mpnet-base-v2": {
            "description": "SBERT paraphrase MPNet model (438MB, 768 dimensions)",
            "size": "large",
            "speed": "medium",
            "quality": "excellent"
        },
        "sentence-transformers/all-MiniLM-L6-v1": {
            "description": "SBERT MiniLM L6 v1 model (23MB, 384 dimensions)",
            "size": "small",
            "speed": "fast",
            "quality": "good"
        },
        "sentence-transformers/msmarco-distilbert-base-v4": {
            "description": "SBERT MS MARCO DistilBERT model (260MB, 768 dimensions)",
            "size": "large",
            "speed": "medium",
            "quality": "very good"
        }
    }

    def __init__(
        self,
        embedding_model: str = "intfloat/e5-base",
        max_file_size_mb: int = 100,
        sample_limit: int = 50000,
        content_weight: float = 0.8,
        filename_weight: float = 0.2
    ):
        """
        Initialize the SemanticAnalyzer.

        Args:
            embedding_model: Name of the sentence-transformers model to use.
                           Available models: all-MiniLM-L6-v2 (default, fast),
                           all-mpnet-base-v2 (high quality), intfloat/e5-base,
                           intfloat/e5-small, intfloat/e5-large, etc.
            max_file_size_mb: Maximum file size to process in MB
            sample_limit: Maximum characters to extract from large files
            content_weight: Weight for content in final similarity (0-1)
            filename_weight: Weight for filename in final similarity (0-1)
        """
        self.embedding_model_name = embedding_model
        self.max_file_size_mb = max_file_size_mb
        self.sample_limit = sample_limit
        self.content_weight = content_weight
        self.filename_weight = filename_weight

        # Initialize components
        self.file_extractor = FileExtractor(max_file_size_mb, sample_limit)
        self.embedding_model = None
        self._initialize_model()

        # Cache for embeddings to avoid recomputation
        self._embedding_cache: Dict[str, np.ndarray] = {}

    @classmethod
    def list_available_models(cls) -> None:
        """Print information about available embedding models."""
        print("\n🤖 Available Embedding Models:")
        print("=" * 70)

        # Group models by type
        miniml_models = []
        e5_models = []
        sbert_models = []
        other_models = []

        for model_name, info in cls.AVAILABLE_MODELS.items():
            if "e5" in model_name.lower():
                e5_models.append((model_name, info))
            elif model_name.startswith("sentence-transformers/"):
                sbert_models.append((model_name, info))
            elif "miniml" in model_name.lower():
                miniml_models.append((model_name, info))
            else:
                other_models.append((model_name, info))

        # Print grouped models
        if miniml_models:
            print("\n🚀 MiniLM Models (Fast & Lightweight):")
            for model_name, info in miniml_models:
                cls._print_model_info(model_name, info)

        if e5_models:
            print("\n⚡ E5 Models (High Quality):")
            for model_name, info in e5_models:
                cls._print_model_info(model_name, info)

        if sbert_models:
            print("\n🎯 SBERT Models (Sentence-BERT):")
            for model_name, info in sbert_models:
                cls._print_model_info(model_name, info)

        if other_models:
            print("\n🔧 Other Models:")
            for model_name, info in other_models:
                cls._print_model_info(model_name, info)

        # Print recommendations
        print("\n💡 Recommendations:")
        recommendations = cls.get_model_recommendations()
        for use_case, model in recommendations.items():
            print(f"   {use_case}: {model}")

        print("\n💡 Usage: --embedding-model MODEL_NAME")
        print("💡 Example: --embedding-model intfloat/e5-base")
        print("=" * 70)

    @classmethod
    def _print_model_info(cls, model_name: str, info: Dict) -> None:
        """Print formatted model information."""
        print(f"  📝 {model_name}")
        print(f"     {info['description']}")
        print(f"     Size: {info['size']}, Speed: {info['speed']}, Quality: {info['quality']}")
        print()

    @classmethod
    def get_model_recommendations(cls) -> Dict[str, str]:
        """Get model recommendations for different use cases."""
        return {
            "fastest": "all-MiniLM-L6-v2",
            "fast": "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "balanced": "intfloat/e5-small",
            "quality": "intfloat/e5-base",
            "sbert_quality": "sentence-transformers/paraphrase-mpnet-base-v2",
            "best": "intfloat/e5-large"
        }

    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            # Log model information if available
            if self.embedding_model_name in self.AVAILABLE_MODELS:
                model_info = self.AVAILABLE_MODELS[self.embedding_model_name]
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                logger.info(f"Model info: {model_info['description']}")
            else:
                logger.info(f"Loading custom embedding model: {self.embedding_model_name}")
                logger.warning(f"Model not in predefined list. Available models: {list(self.AVAILABLE_MODELS.keys())}")

            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")

            # Log actual embedding dimension
            try:
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Model embedding dimension: {embedding_dim}")
            except Exception:
                logger.debug("Could not determine embedding dimension")

        except ImportError:
            logger.error("sentence-transformers not available. Please install it: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading embedding model {self.embedding_model_name}: {e}")
            # Try fallback models in order of preference
            fallback_models = ["intfloat/e5-base", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]

            for fallback_model in fallback_models:
                if fallback_model != self.embedding_model_name:
                    try:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        self.embedding_model = SentenceTransformer(fallback_model)
                        self.embedding_model_name = fallback_model  # Update model name
                        logger.info(f"Fallback model {fallback_model} loaded successfully")
                        return
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
                        continue

            logger.error("All fallback models failed")
            raise

    def analyze_item(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Generate semantic embedding for a file or folder.

        Args:
            path: Path to the file or folder

        Returns:
            Tuple of (embedding_vector, metadata_dict)
        """
        try:
            if path.is_dir():
                return self.analyze_folder(path)
            else:
                return self.analyze_file(path)
        except Exception as e:
            logger.error(f"Error analyzing item {path}: {e}")
            # Return fallback embedding based on path name
            fallback_text = self._get_fallback_text(path)
            embedding = self.generate_embedding(fallback_text)
            metadata = {
                'analysis_method': 'fallback',
                'error': str(e),
                'content_available': False
            }
            return embedding, metadata

    def analyze_file(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Analyze a single file and generate its semantic embedding.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (embedding_vector, metadata_dict)
        """
        # Check cache first
        cache_key = str(file_path.resolve())
        if cache_key in self._embedding_cache:
            logger.debug(f"Using cached embedding for {file_path}")
            cached_embedding = self._embedding_cache[cache_key]
            return cached_embedding, {'analysis_method': 'cached'}

        # Extract text content
        text_content, extraction_metadata = self.file_extractor.extract_text(file_path)

        # Preprocess text
        processed_text = self.preprocess_text(text_content)

        # Generate embedding
        embedding = self.generate_embedding(processed_text)

        # Cache the embedding
        self._embedding_cache[cache_key] = embedding

        # Prepare metadata
        metadata = {
            'analysis_method': 'file_content',
            'text_length': len(processed_text),
            'extraction_metadata': extraction_metadata
        }

        logger.debug(f"Generated embedding for file {file_path} (text length: {len(processed_text)})")
        return embedding, metadata

    def analyze_folder(self, folder_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Analyze a folder by sampling its contents and folder name.

        Args:
            folder_path: Path to the folder

        Returns:
            Tuple of (embedding_vector, metadata_dict)
        """
        # Check cache first
        cache_key = str(folder_path.resolve())
        if cache_key in self._embedding_cache:
            logger.debug(f"Using cached embedding for folder {folder_path}")
            cached_embedding = self._embedding_cache[cache_key]
            return cached_embedding, {'analysis_method': 'cached'}

        text_parts = []
        sampled_files = []
        sample_limit = 10  # Sample up to 10 files from the folder

        # Add folder name as context
        folder_name = folder_path.name.replace('_', ' ').replace('-', ' ')
        text_parts.append(f"Folder: {folder_name}")

        try:
            # Get list of files in the folder
            files = [f for f in folder_path.iterdir() if f.is_file()]

            # Sample files for analysis (prioritize certain types)
            priority_extensions = {'.txt', '.md', '.pdf', '.docx', '.xlsx', '.pptx'}
            priority_files = [f for f in files if f.suffix.lower() in priority_extensions]
            other_files = [f for f in files if f.suffix.lower() not in priority_extensions]

            # Sample files with priority for supported types
            files_to_sample = (priority_files + other_files)[:sample_limit]

            for file_path in files_to_sample:
                try:
                    file_text, _ = self.file_extractor.extract_text(file_path)
                    if file_text.strip():
                        # Limit text per file to avoid overwhelming the embedding
                        limited_text = file_text[:1000] if len(file_text) > 1000 else file_text
                        text_parts.append(limited_text)
                        sampled_files.append(str(file_path.name))
                except Exception as e:
                    logger.debug(f"Could not extract text from {file_path}: {e}")
                    # Still include filename
                    filename_text = file_path.stem.replace('_', ' ').replace('-', ' ')
                    text_parts.append(filename_text)
                    sampled_files.append(str(file_path.name))

        except Exception as e:
            logger.warning(f"Error sampling folder contents {folder_path}: {e}")

        # Combine all text
        combined_text = '\n'.join(text_parts)
        processed_text = self.preprocess_text(combined_text)

        # Generate embedding
        embedding = self.generate_embedding(processed_text)

        # Cache the embedding
        self._embedding_cache[cache_key] = embedding

        # Prepare metadata
        metadata = {
            'analysis_method': 'folder_sampling',
            'sampled_files': sampled_files,
            'total_files': len([f for f in folder_path.iterdir() if f.is_file()]),
            'text_length': len(processed_text)
        }

        logger.debug(f"Generated embedding for folder {folder_path} (sampled {len(sampled_files)} files)")
        return embedding, metadata

    def get_text_content(self, file_path: Path) -> str:
        """
        Extract text content using FileExtractor.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content
        """
        text_content, _ = self.file_extractor.extract_text(file_path)
        return text_content

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to semantic vector embedding.

        Args:
            text: Text to embed

        Returns:
            Numpy array representing the semantic embedding
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())

        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text before embedding generation.

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove non-printable characters except common ones
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)

        # Normalize some common patterns
        text = re.sub(r'https?://\S+', '[URL]', text)  # Replace URLs
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]', text)  # Replace dates
        text = re.sub(r'\b\d+\.\d+\b', '[NUMBER]', text)  # Replace decimals

        # Strip and limit length
        text = text.strip()
        if len(text) > self.sample_limit:
            text = text[:self.sample_limit]

        return text

    def _get_fallback_text(self, path: Path) -> str:
        """
        Generate fallback text from path information when content extraction fails.

        Args:
            path: Path to generate fallback text for

        Returns:
            Fallback text based on path information
        """
        parts = []

        # Add filename/foldername
        name = path.name.replace('_', ' ').replace('-', ' ')
        parts.append(name)

        # Add file extension context if it's a file
        if path.is_file() and path.suffix:
            extension = path.suffix[1:].lower()  # Remove the dot
            parts.append(f"{extension} file")

        # Add parent directory context
        if path.parent.name != path.root:
            parent_name = path.parent.name.replace('_', ' ').replace('-', ' ')
            parts.append(f"from {parent_name}")

        return ' '.join(parts)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_size(self) -> int:
        """Get the current number of cached embeddings."""
        return len(self._embedding_cache)