"""
Clustering module for grouping similar items and detecting themes.

This module provides clustering algorithms to group similar items together
when they don't match existing themes, enabling automatic theme creation.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path

logger = logging.getLogger('semantic_organizer.clusterer')


class Clusterer:
    """Class for clustering similar items and determining optimal groupings."""

    def __init__(self, min_cluster_size: int = 2, max_clusters: int = 20):
        """
        Initialize the Clusterer.

        Args:
            min_cluster_size: Minimum items needed to form a cluster
            max_clusters: Maximum number of clusters to create
        """
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters

        # Initialize clustering algorithms
        self._init_clustering_algorithms()

    def _init_clustering_algorithms(self) -> None:
        """Initialize clustering algorithms from scikit-learn and optional HDBSCAN."""
        try:
            from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            from sklearn.neighbors import NearestNeighbors
            self._sklearn_available = True
            self.dbscan = DBSCAN
            self.kmeans = KMeans
            self.agglomerative = AgglomerativeClustering
            self.silhouette_score = silhouette_score
            self.nearest_neighbors = NearestNeighbors

            # Try to import HDBSCAN for better density-based clustering
            try:
                import hdbscan
                self.hdbscan = hdbscan.HDBSCAN
                self._hdbscan_available = True
                logger.debug("HDBSCAN available for advanced clustering")
            except ImportError:
                logger.debug("HDBSCAN not available, using fallback clustering methods")
                self._hdbscan_available = False

        except ImportError:
            logger.error("scikit-learn not available. Clustering functionality will be limited.")
            self._sklearn_available = False
            self._hdbscan_available = False

    def cluster_items(
        self,
        embeddings: List[np.ndarray],
        items: List[Tuple[Path, Dict]],
        method: str = "auto"
    ) -> Dict[int, List[int]]:
        """
        Group similar items into clusters.

        Args:
            embeddings: List of embedding vectors for each item
            items: List of tuples containing (path, metadata) for each item
            method: Clustering method ("auto", "dbscan", "kmeans", "hierarchical")

        Returns:
            Dictionary mapping cluster_id to list of item indices
        """
        if not embeddings or len(embeddings) < 2:
            logger.info("Not enough items for clustering")
            return {0: list(range(len(embeddings)))} if embeddings else {}

        if not self._sklearn_available:
            return self._simple_distance_clustering(embeddings, items)

        try:
            # Convert embeddings to numpy array
            X = np.array(embeddings)

            if method == "auto":
                method = self._choose_clustering_method(X)

            logger.info(f"Clustering {len(embeddings)} items using {method} method")

            if method == "hdbscan_knn":
                clusters = self._cluster_hdbscan_knn(X)
            elif method == "dbscan":
                clusters = self._cluster_dbscan(X)
            elif method == "kmeans":
                clusters = self._cluster_kmeans(X)
            elif method == "hierarchical":
                clusters = self._cluster_hierarchical(X)
            else:
                logger.warning(f"Unknown clustering method: {method}, falling back to DBSCAN")
                clusters = self._cluster_dbscan(X)

            # Filter out clusters that are too small
            filtered_clusters = {}
            cluster_id = 0

            for original_id, indices in clusters.items():
                if len(indices) >= self.min_cluster_size:
                    filtered_clusters[cluster_id] = indices
                    cluster_id += 1
                else:
                    # Add small clusters to a miscellaneous cluster
                    if -1 not in filtered_clusters:
                        filtered_clusters[-1] = []
                    filtered_clusters[-1].extend(indices)

            logger.info(f"Created {len(filtered_clusters)} clusters")
            return filtered_clusters

        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            return self._fallback_clustering(embeddings, items)

    def _choose_clustering_method(self, X: np.ndarray) -> str:
        """
        Automatically choose the best clustering method based on data characteristics.

        Args:
            X: Embedding matrix

        Returns:
            Best clustering method name
        """
        n_samples = X.shape[0]

        # Prefer HDBSCAN+KNN hybrid if available for natural theme discovery
        if self._hdbscan_available and n_samples >= 5:
            return "hdbscan_knn"
        elif n_samples < 10:
            return "hierarchical"  # Works well for small datasets
        elif n_samples < 100:
            return "kmeans"  # K-means generally better for theme detection
        else:
            return "kmeans"  # Efficient for larger datasets

    def _cluster_dbscan(self, X: np.ndarray) -> Dict[int, List[int]]:
        """
        Cluster using DBSCAN algorithm.

        Args:
            X: Embedding matrix

        Returns:
            Dictionary mapping cluster_id to list of item indices
        """
        # Determine appropriate eps value based on data
        from sklearn.neighbors import NearestNeighbors

        # Find optimal eps using k-distance graph
        k = min(4, X.shape[0] - 1)  # Use min(4, n-1) for k-distance
        if k <= 0:
            k = 1

        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Use 90th percentile of k-distances as eps
        eps = np.percentile(distances[:, -1], 90)
        eps = max(eps, 0.1)  # Ensure eps is not too small

        # Run DBSCAN
        dbscan = self.dbscan(eps=eps, min_samples=self.min_cluster_size)
        cluster_labels = dbscan.fit_predict(X)

        # Convert labels to clusters dictionary
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        logger.debug(f"DBSCAN found {len(clusters)} clusters with eps={eps:.3f}")
        return clusters

    def _cluster_kmeans(self, X: np.ndarray) -> Dict[int, List[int]]:
        """
        Cluster using K-means algorithm.

        Args:
            X: Embedding matrix

        Returns:
            Dictionary mapping cluster_id to list of item indices
        """
        n_samples = X.shape[0]
        optimal_k = self.determine_optimal_clusters(X)

        kmeans = self.kmeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Convert labels to clusters dictionary
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        logger.debug(f"K-means found {len(clusters)} clusters with k={optimal_k}")
        return clusters

    def _cluster_hierarchical(self, X: np.ndarray) -> Dict[int, List[int]]:
        """
        Cluster using Agglomerative Hierarchical clustering.

        Args:
            X: Embedding matrix

        Returns:
            Dictionary mapping cluster_id to list of item indices
        """
        n_samples = X.shape[0]
        optimal_k = self.determine_optimal_clusters(X)

        agg = self.agglomerative(n_clusters=optimal_k, linkage='ward')
        cluster_labels = agg.fit_predict(X)

        # Convert labels to clusters dictionary
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        logger.debug(f"Hierarchical clustering found {len(clusters)} clusters with k={optimal_k}")
        return clusters

    def _cluster_hdbscan_knn(self, X: np.ndarray) -> Dict[int, List[int]]:
        """
        Hybrid HDBSCAN+KNN clustering for natural theme discovery.

        Uses HDBSCAN for initial clustering, then KNN analysis to validate
        and potentially split or merge clusters based on local density patterns.

        Args:
            X: Embedding matrix

        Returns:
            Dictionary mapping cluster_id to list of item indices
        """
        try:
            n_samples = X.shape[0]

            # Step 1: Initial HDBSCAN clustering with adaptive parameters
            min_cluster_size = max(2, min(self.min_cluster_size, n_samples // 4))
            min_samples = max(1, min_cluster_size - 1)

            # Use smaller min_cluster_size for better theme separation
            hdbscan_clusterer = self.hdbscan(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',  # HDBSCAN doesn't support cosine directly
                cluster_selection_epsilon=0.1,  # Allow more granular clusters
                alpha=1.0,  # Standard alpha for consistent clusters
                cluster_selection_method='eom'  # Excess of Mass for natural boundaries
            )

            cluster_labels = hdbscan_clusterer.fit_predict(X)
            logger.debug(f"HDBSCAN initial clustering: {len(set(cluster_labels))} clusters found")

            # Convert HDBSCAN labels to clusters dict
            initial_clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in initial_clusters:
                    initial_clusters[label] = []
                initial_clusters[label].append(idx)

            # Step 2: KNN-based density analysis and cluster refinement
            refined_clusters = self._refine_clusters_with_knn(X, initial_clusters, hdbscan_clusterer)

            logger.debug(f"HDBSCAN+KNN hybrid found {len(refined_clusters)} final clusters")
            return refined_clusters

        except Exception as e:
            logger.error(f"Error in HDBSCAN+KNN clustering: {e}")
            # Fallback to hierarchical clustering
            return self._cluster_hierarchical(X)

    def _refine_clusters_with_knn(
        self,
        X: np.ndarray,
        initial_clusters: Dict[int, List[int]],
        hdbscan_clusterer
    ) -> Dict[int, List[int]]:
        """
        Refine clusters using KNN density analysis.

        Args:
            X: Embedding matrix
            initial_clusters: Initial clusters from HDBSCAN
            hdbscan_clusterer: Fitted HDBSCAN clusterer

        Returns:
            Refined clusters dictionary
        """
        from sklearn.neighbors import NearestNeighbors

        refined_clusters = {}
        cluster_id = 0

        # Analyze each cluster for potential splitting or validation
        for original_id, indices in initial_clusters.items():
            if original_id == -1:  # Noise points
                continue

            if len(indices) < self.min_cluster_size:
                continue

            cluster_points = X[indices]

            # Step 1: KNN analysis for internal cluster density
            k = min(5, len(indices) - 1)  # Use smaller k for cluster analysis
            if k <= 0:
                refined_clusters[cluster_id] = indices
                cluster_id += 1
                continue

            nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
            nbrs.fit(cluster_points)
            distances, _ = nbrs.kneighbors(cluster_points)

            # Calculate density metrics
            avg_distances = np.mean(distances[:, 1:], axis=1)  # Skip self-distance
            density_threshold = np.percentile(avg_distances, 75)  # Top 75% density

            # Step 2: Check if cluster should be split
            high_density_mask = avg_distances <= density_threshold
            low_density_mask = ~high_density_mask

            # If we have significant low-density points, consider sub-clustering
            if np.sum(low_density_mask) >= self.min_cluster_size and np.sum(high_density_mask) >= self.min_cluster_size:
                # Try to split the cluster
                sub_clusters = self._attempt_cluster_split(cluster_points, indices, high_density_mask, low_density_mask)

                for sub_cluster in sub_clusters:
                    if len(sub_cluster) >= self.min_cluster_size:
                        refined_clusters[cluster_id] = sub_cluster
                        cluster_id += 1
            else:
                # Keep original cluster
                refined_clusters[cluster_id] = indices
                cluster_id += 1

        # Handle noise points by attempting to assign them to nearest clusters
        if -1 in initial_clusters:
            noise_indices = initial_clusters[-1]
            refined_clusters = self._reassign_noise_points(X, noise_indices, refined_clusters)

        return refined_clusters

    def _attempt_cluster_split(
        self,
        cluster_points: np.ndarray,
        original_indices: List[int],
        high_density_mask: np.ndarray,
        low_density_mask: np.ndarray
    ) -> List[List[int]]:
        """
        Attempt to split a cluster based on density analysis.

        Returns:
            List of sub-clusters (as lists of original indices)
        """
        sub_clusters = []

        # High density core
        high_density_indices = [original_indices[i] for i, is_high in enumerate(high_density_mask) if is_high]
        if len(high_density_indices) >= self.min_cluster_size:
            sub_clusters.append(high_density_indices)

        # Low density points - try to form coherent sub-clusters
        low_density_indices = [original_indices[i] for i, is_low in enumerate(low_density_mask) if is_low]
        if len(low_density_indices) >= self.min_cluster_size:
            # Use simple distance-based clustering for low-density points
            low_density_points = cluster_points[low_density_mask]

            # Try hierarchical clustering on low-density points
            if len(low_density_points) >= 2:
                from sklearn.cluster import AgglomerativeClustering
                try:
                    n_clusters = max(1, min(2, len(low_density_indices) // self.min_cluster_size))
                    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    sub_labels = agg.fit_predict(low_density_points)

                    # Group by sub-labels
                    sub_groups = {}
                    for i, label in enumerate(sub_labels):
                        if label not in sub_groups:
                            sub_groups[label] = []
                        sub_groups[label].append(low_density_indices[i])

                    for sub_group in sub_groups.values():
                        if len(sub_group) >= self.min_cluster_size:
                            sub_clusters.append(sub_group)

                except Exception:
                    # Fallback: treat all low-density points as one cluster
                    sub_clusters.append(low_density_indices)
            else:
                sub_clusters.append(low_density_indices)

        # If no valid sub-clusters were created, return original
        if not sub_clusters:
            sub_clusters = [original_indices]

        return sub_clusters

    def _reassign_noise_points(
        self,
        X: np.ndarray,
        noise_indices: List[int],
        clusters: Dict[int, List[int]]
    ) -> Dict[int, List[int]]:
        """
        Try to reassign noise points to the nearest appropriate cluster.
        """
        if not noise_indices or not clusters:
            return clusters

        from sklearn.neighbors import NearestNeighbors

        # Create a mapping from point indices to cluster IDs
        point_to_cluster = {}
        cluster_centers = {}

        for cluster_id, indices in clusters.items():
            cluster_points = X[indices]
            cluster_centers[cluster_id] = np.mean(cluster_points, axis=0)
            for idx in indices:
                point_to_cluster[idx] = cluster_id

        # For each noise point, find the nearest cluster
        for noise_idx in noise_indices:
            noise_point = X[noise_idx].reshape(1, -1)

            best_cluster = None
            best_distance = float('inf')

            for cluster_id, center in cluster_centers.items():
                distance = np.linalg.norm(noise_point - center.reshape(1, -1))
                if distance < best_distance:
                    best_distance = distance
                    best_cluster = cluster_id

            # Only assign if reasonably close (within 1.5x the average intra-cluster distance)
            if best_cluster is not None:
                cluster_indices = clusters[best_cluster]
                cluster_points = X[cluster_indices]
                avg_intra_distance = np.mean([
                    np.linalg.norm(cluster_points[i] - cluster_points[j])
                    for i in range(len(cluster_points))
                    for j in range(i+1, len(cluster_points))
                ])

                if best_distance <= avg_intra_distance * 1.5:
                    clusters[best_cluster].append(noise_idx)

        return clusters

    def determine_optimal_clusters(self, X: np.ndarray) -> int:
        """
        Auto-detect optimal number of clusters for the data.

        Args:
            X: Embedding matrix

        Returns:
            Optimal number of clusters
        """
        n_samples = X.shape[0]

        # Handle edge cases
        if n_samples < 4:
            return max(1, n_samples // 2)

        max_k = min(self.max_clusters, n_samples // self.min_cluster_size)
        if max_k < 2:
            return 1

        best_k = 2
        best_score = -1

        # Try different values of k and use silhouette score
        for k in range(2, max_k + 1):
            try:
                kmeans = self.kmeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)

                # Skip if any cluster would be too small
                unique_labels, counts = np.unique(labels, return_counts=True)
                if np.any(counts < self.min_cluster_size):
                    continue

                score = self.silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception as e:
                logger.debug(f"Error evaluating k={k}: {e}")
                continue

        logger.debug(f"Optimal k={best_k} with silhouette score={best_score:.3f}")
        return best_k

    def _simple_distance_clustering(
        self,
        embeddings: List[np.ndarray],
        items: List[Tuple[Path, Dict]]
    ) -> Dict[int, List[int]]:
        """
        Simple distance-based clustering when scikit-learn is not available.

        Args:
            embeddings: List of embedding vectors
            items: List of item tuples

        Returns:
            Dictionary mapping cluster_id to list of item indices
        """
        logger.info("Using simple distance-based clustering (scikit-learn not available)")

        if len(embeddings) < 2:
            return {0: [0]} if embeddings else {}

        # Use simple threshold-based clustering
        threshold = 0.7  # Similarity threshold
        clusters = {}
        cluster_id = 0
        assigned = set()

        for i, emb1 in enumerate(embeddings):
            if i in assigned:
                continue

            # Start new cluster
            current_cluster = [i]
            assigned.add(i)

            # Find similar items
            for j, emb2 in enumerate(embeddings):
                if j in assigned or j <= i:
                    continue

                similarity = self._cosine_similarity(emb1, emb2)
                if similarity >= threshold:
                    current_cluster.append(j)
                    assigned.add(j)

            if len(current_cluster) >= self.min_cluster_size:
                clusters[cluster_id] = current_cluster
                cluster_id += 1

        # Handle unassigned items
        unassigned = []
        for i in range(len(embeddings)):
            if i not in assigned:
                unassigned.append(i)

        if unassigned:
            if len(unassigned) >= self.min_cluster_size:
                clusters[cluster_id] = unassigned
            else:
                # Add to miscellaneous cluster
                clusters[-1] = clusters.get(-1, []) + unassigned

        return clusters

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            # Convert from [-1,1] to [0,1]
            return (similarity + 1) / 2

        except Exception:
            return 0.0

    def _fallback_clustering(
        self,
        embeddings: List[np.ndarray],
        items: List[Tuple[Path, Dict]]
    ) -> Dict[int, List[int]]:
        """
        Fallback clustering method when other methods fail.

        Args:
            embeddings: List of embedding vectors
            items: List of item tuples

        Returns:
            Dictionary mapping cluster_id to list of item indices
        """
        logger.warning("Using fallback clustering method")

        # Group by file extension as a basic fallback
        extension_groups = {}

        for idx, (path, metadata) in enumerate(items):
            if path.is_file():
                ext = path.suffix.lower()
                if ext not in extension_groups:
                    extension_groups[ext] = []
                extension_groups[ext].append(idx)
            else:
                # Folders go into their own group
                if 'folders' not in extension_groups:
                    extension_groups['folders'] = []
                extension_groups['folders'].append(idx)

        # Convert to cluster format
        clusters = {}
        cluster_id = 0

        for ext, indices in extension_groups.items():
            if len(indices) >= self.min_cluster_size:
                clusters[cluster_id] = indices
                cluster_id += 1
            else:
                # Add to miscellaneous
                if -1 not in clusters:
                    clusters[-1] = []
                clusters[-1].extend(indices)

        return clusters

    def assign_theme_names(self, clusters: Dict[int, List[int]], items: List[Tuple[Path, Dict]]) -> Dict[int, str]:
        """
        Generate descriptive names for clusters.

        Args:
            clusters: Dictionary mapping cluster_id to list of item indices
            items: List of item tuples

        Returns:
            Dictionary mapping cluster_id to theme name
        """
        theme_names = {}

        for cluster_id, indices in clusters.items():
            cluster_items = [items[i] for i in indices]
            cluster_paths = [item[0] for item in cluster_items]

            if cluster_id == -1:
                theme_names[cluster_id] = "Miscellaneous_Files"
            else:
                theme_name = self._generate_cluster_name(cluster_paths)
                theme_names[cluster_id] = theme_name

        return theme_names

    def _generate_cluster_name(self, paths: List[Path]) -> str:
        """
        Generate a descriptive name for a cluster based on the paths it contains.

        Args:
            paths: List of paths in the cluster

        Returns:
            Generated cluster name
        """
        # This is a simplified version - the ThemeMatcher has more sophisticated logic
        extensions = []
        for path in paths:
            if path.is_file() and path.suffix:
                extensions.append(path.suffix.lower())

        if extensions:
            # Use most common extension
            from collections import Counter
            ext_counts = Counter(extensions)
            most_common_ext = ext_counts.most_common(1)[0][0]

            ext_mapping = {
                '.txt': 'Text_Files',
                '.pdf': 'PDF_Documents',
                '.docx': 'Word_Documents',
                '.xlsx': 'Spreadsheets',
                '.jpg': 'Images',
                '.png': 'Images',
                '.mp4': 'Videos',
                '.mp3': 'Audio_Files'
            }

            return ext_mapping.get(most_common_ext, f'Files_{most_common_ext[1:].upper()}')
        else:
            return 'Folders'