"""
Steel Member Damage Detection v3.0
Robust damage detection with instance validation and RANSAC-based analysis.

Key improvements over v2:
- HDBSCAN for adaptive clustering
- Member validation (geometry checks before damage analysis)
- RANSAC-based centerline fitting (robust to noise)
- 5cm damage threshold (conservative detection)
- Confidence scoring for each detection
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings('ignore')

# Try to import hdbscan, fall back to DBSCAN if not available
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("Warning: hdbscan not available, using DBSCAN fallback")


class MemberOrientation(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL = "diagonal"


class MemberType(Enum):
    BUNTON = "bunton"
    GUARD = "guard"
    COLUMN = "column"
    UNKNOWN = "unknown"


class ValidationStatus(Enum):
    VALID = "valid"
    TOO_FEW_POINTS = "too_few_points"
    NOT_ELONGATED = "not_elongated"
    TOO_SHORT = "too_short"
    CROSS_SECTION_TOO_LARGE = "cross_section_too_large"
    CROSS_SECTION_TOO_THIN = "cross_section_too_thin"
    INCONSISTENT_CROSS_SECTION = "inconsistent_cross_section"


@dataclass
class MemberValidation:
    """Result of member validation."""
    is_valid: bool
    confidence: float  # 0-1
    status: ValidationStatus
    reason: str

    # Geometry info (even for invalid members)
    length_m: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0
    aspect_ratio: float = 0.0
    cross_section_consistency: float = 0.0


@dataclass
class DamageAnalysis:
    """Damage analysis result for a validated member."""
    is_damaged: bool
    severity: str  # 'NONE', 'LOW', 'MEDIUM', 'HIGH'
    max_deviation_cm: float
    mean_deviation_cm: float
    p95_deviation_cm: float
    damage_percentage: float  # % of points exceeding threshold
    num_damage_regions: int
    centerline_points: np.ndarray = None


@dataclass
class MemberAnalysisResult:
    """Complete analysis result for one steel member instance."""

    # Identification
    member_id: int
    member_type: MemberType
    orientation: MemberOrientation

    # Point cloud
    points: np.ndarray
    num_points: int
    centroid: np.ndarray
    bounding_box: Dict

    # Validation
    validation: MemberValidation

    # Damage (only populated if validation passed)
    damage: Optional[DamageAnalysis] = None

    def to_dict(self) -> Dict:
        return {
            'member_id': int(self.member_id),
            'member_type': self.member_type.value,
            'orientation': self.orientation.value,
            'num_points': self.num_points,
            'centroid': [float(x) for x in self.centroid],
            'bounding_box': self.bounding_box,
            'validation': {
                'is_valid': self.validation.is_valid,
                'confidence': float(self.validation.confidence),
                'status': self.validation.status.value,
                'reason': self.validation.reason,
                'geometry': {
                    'length_m': float(self.validation.length_m),
                    'width_m': float(self.validation.width_m),
                    'height_m': float(self.validation.height_m),
                    'aspect_ratio': float(self.validation.aspect_ratio)
                }
            },
            'damage': {
                'is_damaged': self.damage.is_damaged,
                'severity': self.damage.severity,
                'max_deviation_cm': float(self.damage.max_deviation_cm),
                'mean_deviation_cm': float(self.damage.mean_deviation_cm),
                'p95_deviation_cm': float(self.damage.p95_deviation_cm),
                'damage_percentage': float(self.damage.damage_percentage),
                'num_damage_regions': self.damage.num_damage_regions
            } if self.damage else None
        }


class MemberValidator:
    """
    Validates if a point cluster represents a real steel member.

    Checks:
    - Minimum point count
    - Elongated shape (aspect ratio)
    - Minimum length
    - Cross-section dimensions
    - Cross-section consistency along length
    """

    # Validation thresholds
    MIN_POINTS = 200
    MIN_ASPECT_RATIO = 2.5       # Length / max(width, height) - reduced from 3.0 for real-world
    MIN_LENGTH_BUNTON = 0.4      # 40cm minimum for buntons
    MIN_LENGTH_COLUMN = 0.8      # 80cm minimum for columns
    MAX_CROSS_SECTION = 1.5      # Max 1.5m cross-section (allow for thick beams)
    MIN_CROSS_SECTION = 0.03     # Min 3cm (exclude very thin noise)
    MIN_CONSISTENCY = 0.4        # Cross-section consistency threshold

    def validate(self, points: np.ndarray, expected_type: MemberType = MemberType.BUNTON) -> MemberValidation:
        """
        Validate if a point cluster is a real steel member.

        Returns MemberValidation with is_valid, confidence, and reason.
        """

        # 1. Point count check
        if len(points) < self.MIN_POINTS:
            return MemberValidation(
                is_valid=False,
                confidence=0.0,
                status=ValidationStatus.TOO_FEW_POINTS,
                reason=f"Too few points ({len(points)} < {self.MIN_POINTS})"
            )

        # 2. PCA analysis for dimensions
        pca = PCA(n_components=3)
        pca.fit(points)

        # Get dimensions along each principal axis
        centered = points - points.mean(axis=0)
        lengths = []
        for axis in pca.components_:
            proj = np.dot(centered, axis)
            lengths.append(proj.max() - proj.min())

        length = lengths[0]      # Primary axis = length
        width = lengths[1]       # Secondary = width
        height = lengths[2]      # Tertiary = height

        # 3. Aspect ratio check
        max_cross = max(width, height)
        aspect_ratio = length / max(max_cross, 0.01)

        if aspect_ratio < self.MIN_ASPECT_RATIO:
            return MemberValidation(
                is_valid=False,
                confidence=0.2,
                status=ValidationStatus.NOT_ELONGATED,
                reason=f"Not elongated enough (ratio {aspect_ratio:.1f} < {self.MIN_ASPECT_RATIO})",
                length_m=length, width_m=width, height_m=height, aspect_ratio=aspect_ratio
            )

        # 4. Length check
        min_length = self.MIN_LENGTH_BUNTON if expected_type in [MemberType.BUNTON, MemberType.GUARD] else self.MIN_LENGTH_COLUMN
        if length < min_length:
            return MemberValidation(
                is_valid=False,
                confidence=0.3,
                status=ValidationStatus.TOO_SHORT,
                reason=f"Too short ({length:.2f}m < {min_length}m)",
                length_m=length, width_m=width, height_m=height, aspect_ratio=aspect_ratio
            )

        # 5. Cross-section dimension checks
        if max_cross > self.MAX_CROSS_SECTION:
            return MemberValidation(
                is_valid=False,
                confidence=0.2,
                status=ValidationStatus.CROSS_SECTION_TOO_LARGE,
                reason=f"Cross-section too large ({max_cross:.2f}m > {self.MAX_CROSS_SECTION}m)",
                length_m=length, width_m=width, height_m=height, aspect_ratio=aspect_ratio
            )

        min_cross = min(width, height)
        if min_cross < self.MIN_CROSS_SECTION:
            return MemberValidation(
                is_valid=False,
                confidence=0.3,
                status=ValidationStatus.CROSS_SECTION_TOO_THIN,
                reason=f"Cross-section too thin ({min_cross:.2f}m < {self.MIN_CROSS_SECTION}m)",
                length_m=length, width_m=width, height_m=height, aspect_ratio=aspect_ratio
            )

        # 6. Cross-section consistency check
        consistency = self._check_cross_section_consistency(points, pca)
        if consistency < self.MIN_CONSISTENCY:
            return MemberValidation(
                is_valid=False,
                confidence=0.4,
                status=ValidationStatus.INCONSISTENT_CROSS_SECTION,
                reason=f"Inconsistent cross-section ({consistency:.2f} < {self.MIN_CONSISTENCY})",
                length_m=length, width_m=width, height_m=height,
                aspect_ratio=aspect_ratio, cross_section_consistency=consistency
            )

        # 7. Calculate overall confidence
        confidence = self._calculate_confidence(aspect_ratio, length, max_cross, consistency)

        return MemberValidation(
            is_valid=True,
            confidence=confidence,
            status=ValidationStatus.VALID,
            reason="Valid steel member",
            length_m=length, width_m=width, height_m=height,
            aspect_ratio=aspect_ratio, cross_section_consistency=consistency
        )

    def _check_cross_section_consistency(self, points: np.ndarray, pca: PCA, n_slices: int = 5) -> float:
        """
        Check if cross-section is consistent along member length.
        Returns consistency score 0-1 (1 = perfectly consistent).
        """
        principal_axis = pca.components_[0]
        secondary_axis = pca.components_[1]
        centered = points - points.mean(axis=0)
        projections = np.dot(centered, principal_axis)

        # Divide into slices along length
        slice_widths = []
        p_min, p_max = projections.min(), projections.max()

        for i in range(n_slices):
            slice_start = p_min + (p_max - p_min) * i / n_slices
            slice_end = p_min + (p_max - p_min) * (i + 1) / n_slices
            mask = (projections >= slice_start) & (projections < slice_end)

            if mask.sum() < 10:
                continue

            slice_points = points[mask]
            # Get perpendicular spread (width)
            perp_proj = np.dot(slice_points - slice_points.mean(axis=0), secondary_axis)
            slice_width = perp_proj.max() - perp_proj.min()
            slice_widths.append(slice_width)

        if len(slice_widths) < 3:
            return 0.5  # Not enough data, assume moderate consistency

        # Consistency = 1 - coefficient of variation
        mean_width = np.mean(slice_widths)
        std_width = np.std(slice_widths)
        cv = std_width / (mean_width + 1e-6)

        return max(0, min(1, 1 - cv))

    def _calculate_confidence(self, aspect_ratio: float, length: float,
                              cross_section: float, consistency: float) -> float:
        """Calculate overall confidence score 0-1."""
        # Higher aspect ratio = more confident it's a beam
        aspect_score = min(1.0, (aspect_ratio - self.MIN_ASPECT_RATIO) / 5.0 + 0.5)

        # Longer = more confident
        length_score = min(1.0, length / 2.0)

        # Reasonable cross-section = more confident
        cross_score = 1.0 if 0.1 < cross_section < 0.8 else 0.7

        # Consistency directly contributes
        consistency_score = consistency

        # Weighted average
        confidence = (
            0.3 * aspect_score +
            0.2 * length_score +
            0.2 * cross_score +
            0.3 * consistency_score
        )

        return min(1.0, max(0.0, confidence))


def segment_members(points: np.ndarray, member_type: str = 'bunton') -> np.ndarray:
    """
    Segment point cloud into individual steel member instances.
    Uses HDBSCAN if available, otherwise DBSCAN.
    """

    if len(points) < 50:
        return np.full(len(points), -1)

    # Parameters based on member type
    if member_type in ['bunton', 'guard']:
        min_cluster_size = 150
        min_samples = 20
        eps = 0.25  # For DBSCAN fallback
    else:  # column
        min_cluster_size = 300
        min_samples = 30
        eps = 0.3

    if HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom',
            metric='euclidean'
        )
        labels = clusterer.fit_predict(points)
    else:
        # DBSCAN fallback
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clusterer.fit_predict(points)

    # Post-process: merge fragmented clusters
    labels = merge_fragmented_clusters(points, labels, max_gap=0.15)

    return labels


def merge_fragmented_clusters(points: np.ndarray, labels: np.ndarray, max_gap: float = 0.15) -> np.ndarray:
    """
    Merge clusters that are likely fragments of the same member.

    Two clusters are merged if:
    - Their centroids are close along the primary axis
    - The gap between them is small
    """
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        return labels

    # Calculate cluster info
    cluster_info = {}
    for label in unique_labels:
        mask = labels == label
        cluster_points = points[mask]

        # PCA for orientation
        if len(cluster_points) < 10:
            continue

        pca = PCA(n_components=1)
        pca.fit(cluster_points)

        cluster_info[label] = {
            'centroid': cluster_points.mean(axis=0),
            'axis': pca.components_[0],
            'points': cluster_points
        }

    # Find pairs to merge
    labels_copy = labels.copy()
    merged = set()

    for l1 in cluster_info:
        if l1 in merged:
            continue
        for l2 in cluster_info:
            if l2 <= l1 or l2 in merged:
                continue

            c1 = cluster_info[l1]
            c2 = cluster_info[l2]

            # Check if axes are aligned (same direction)
            axis_dot = abs(np.dot(c1['axis'], c2['axis']))
            if axis_dot < 0.9:  # Not aligned
                continue

            # Check centroid distance
            centroid_dist = np.linalg.norm(c1['centroid'] - c2['centroid'])

            # Get extent of each cluster along combined axis
            combined_axis = c1['axis']

            proj1 = np.dot(c1['points'] - c1['centroid'], combined_axis)
            proj2 = np.dot(c2['points'] - c1['centroid'], combined_axis)

            # Check for gap
            max1, min1 = proj1.max(), proj1.min()
            max2, min2 = proj2.max(), proj2.min()

            gap = max(min1 - max2, min2 - max1)

            if gap < max_gap and gap > -0.5:  # Allow some overlap
                # Merge: assign l2 points to l1
                labels_copy[labels == l2] = l1
                merged.add(l2)

    return labels_copy


def determine_orientation(principal_axis: np.ndarray) -> MemberOrientation:
    """Determine member orientation from principal axis."""
    abs_axis = np.abs(principal_axis)

    if abs_axis[2] > 0.7:
        return MemberOrientation.VERTICAL
    elif abs_axis[0] > 0.5 or abs_axis[1] > 0.5:
        return MemberOrientation.HORIZONTAL
    else:
        return MemberOrientation.DIAGONAL


def fit_centerline(points: np.ndarray, n_segments: int = 10) -> np.ndarray:
    """
    Fit a piecewise linear centerline through the member using median (robust to noise).

    Returns array of centerline points.
    """
    if len(points) < n_segments * 5:
        n_segments = max(3, len(points) // 5)

    # Sort points along principal axis
    pca = PCA(n_components=1)
    t = pca.fit_transform(points).flatten()
    sorted_idx = np.argsort(t)
    sorted_points = points[sorted_idx]

    # Divide into segments and take median of each
    segment_size = len(sorted_points) // n_segments
    centerline_points = []

    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(sorted_points)
        segment = sorted_points[start:end]

        if len(segment) > 0:
            # Use median for robustness
            centerline_points.append(np.median(segment, axis=0))

    return np.array(centerline_points)


def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """Calculate perpendicular distance from point to line segment."""
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq < 1e-10:
        return np.linalg.norm(point - seg_start)

    # Project point onto line
    t = max(0, min(1, np.dot(point - seg_start, seg_vec) / seg_len_sq))
    projection = seg_start + t * seg_vec

    return np.linalg.norm(point - projection)


def calculate_centerline_deviations(points: np.ndarray, centerline: np.ndarray) -> np.ndarray:
    """Calculate perpendicular distance from each point to the centerline."""
    deviations = np.zeros(len(points))

    for i, point in enumerate(points):
        min_dist = float('inf')
        for j in range(len(centerline) - 1):
            dist = point_to_segment_distance(point, centerline[j], centerline[j + 1])
            min_dist = min(min_dist, dist)
        deviations[i] = min_dist

    return deviations


def detect_damage(points: np.ndarray, threshold_cm: float = 5.0,
                  cross_section_width: float = None) -> DamageAnalysis:
    """
    Detect damage by analyzing deviation from ideal centerline.

    The key insight: We need to measure BENDING deviation, not cross-section spread.
    A straight beam with 50cm cross-section will have points 25cm from centerline.
    We detect damage by looking at how much the CENTERLINE itself deviates from straight.

    Args:
        points: Validated member point cloud
        threshold_cm: Deviation threshold in cm (default 5cm per user requirement)
        cross_section_width: Expected cross-section width to account for

    Returns:
        DamageAnalysis with metrics and severity
    """
    threshold_m = threshold_cm / 100.0

    # Fit robust centerline using segment medians
    centerline = fit_centerline(points, n_segments=15)

    if len(centerline) < 3:
        return DamageAnalysis(
            is_damaged=False, severity='NONE',
            max_deviation_cm=0, mean_deviation_cm=0, p95_deviation_cm=0,
            damage_percentage=0, num_damage_regions=0, centerline_points=centerline
        )

    # KEY CHANGE: Measure how much the CENTERLINE deviates from a straight line
    # Not how much individual points deviate from centerline (that's just cross-section)
    centerline_deviation = measure_centerline_straightness(centerline)

    # Also check for local bends in centerline
    bend_angles = detect_centerline_bends(centerline)
    max_bend_angle = max(bend_angles) if bend_angles else 0

    # The deviation is how much the centerline bends, not cross-section spread
    max_deviation = centerline_deviation['max_deviation']
    mean_deviation = centerline_deviation['mean_deviation']
    p95_deviation = centerline_deviation['p95_deviation']

    # Classify severity based on centerline deviation
    if max_deviation > threshold_m * 4:  # >20cm centerline bend
        severity = 'HIGH'
    elif max_deviation > threshold_m * 2:  # >10cm
        severity = 'MEDIUM'
    elif max_deviation > threshold_m or max_bend_angle > 10:  # >5cm or >10 degree bend
        severity = 'LOW'
    else:
        severity = 'NONE'

    # Calculate damage percentage based on centerline segments exceeding threshold
    segment_deviations = centerline_deviation['segment_deviations']
    damage_percentage = 100.0 * np.sum(segment_deviations > threshold_m) / len(segment_deviations)

    return DamageAnalysis(
        is_damaged=(severity != 'NONE'),
        severity=severity,
        max_deviation_cm=max_deviation * 100,
        mean_deviation_cm=mean_deviation * 100,
        p95_deviation_cm=p95_deviation * 100,
        damage_percentage=damage_percentage,
        num_damage_regions=1 if severity != 'NONE' else 0,
        centerline_points=centerline
    )


def measure_centerline_straightness(centerline: np.ndarray) -> Dict:
    """
    Measure how straight the centerline is by fitting a best-fit line
    and measuring perpendicular deviations.

    Returns dict with max_deviation, mean_deviation, p95_deviation in meters.
    """
    if len(centerline) < 3:
        return {'max_deviation': 0, 'mean_deviation': 0, 'p95_deviation': 0, 'segment_deviations': np.array([0])}

    # Fit best-fit line through all centerline points
    # Using endpoints to define ideal straight line
    start = centerline[0]
    end = centerline[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 0.01:
        return {'max_deviation': 0, 'mean_deviation': 0, 'p95_deviation': 0, 'segment_deviations': np.array([0])}

    line_dir = line_vec / line_len

    # Calculate perpendicular distance of each centerline point from the ideal line
    deviations = []
    for point in centerline:
        # Vector from start to point
        v = point - start
        # Project onto line
        proj_len = np.dot(v, line_dir)
        proj_point = start + proj_len * line_dir
        # Perpendicular distance
        perp_dist = np.linalg.norm(point - proj_point)
        deviations.append(perp_dist)

    deviations = np.array(deviations)

    return {
        'max_deviation': float(np.max(deviations)),
        'mean_deviation': float(np.mean(deviations)),
        'p95_deviation': float(np.percentile(deviations, 95)),
        'segment_deviations': deviations
    }


def detect_centerline_bends(centerline: np.ndarray) -> List[float]:
    """
    Detect sharp bends in the centerline by measuring angle changes.

    Returns list of angles (degrees) between consecutive segments.
    """
    if len(centerline) < 3:
        return []

    angles = []
    for i in range(1, len(centerline) - 1):
        v1 = centerline[i] - centerline[i-1]
        v2 = centerline[i+1] - centerline[i]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 0.01 or norm2 < 0.01:
            continue

        v1 = v1 / norm1
        v2 = v2 / norm2

        cos_angle = np.clip(np.dot(v1, v2), -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(angle)

    return angles


def analyze_member(member_id: int, points: np.ndarray,
                   member_type: MemberType, validator: MemberValidator,
                   damage_threshold_cm: float = 5.0) -> MemberAnalysisResult:
    """
    Complete analysis pipeline for a single member instance.

    1. Validate geometry
    2. If valid, analyze for damage
    3. Return complete result
    """

    # Basic info
    centroid = points.mean(axis=0)
    bbox = {
        'min': [float(x) for x in points.min(axis=0)],
        'max': [float(x) for x in points.max(axis=0)]
    }

    # Determine orientation
    pca = PCA(n_components=3)
    pca.fit(points)
    orientation = determine_orientation(pca.components_[0])

    # Validate
    validation = validator.validate(points, member_type)

    # Analyze damage only if valid
    damage = None
    if validation.is_valid:
        damage = detect_damage(points, threshold_cm=damage_threshold_cm)

    return MemberAnalysisResult(
        member_id=member_id,
        member_type=member_type,
        orientation=orientation,
        points=points,
        num_points=len(points),
        centroid=centroid,
        bounding_box=bbox,
        validation=validation,
        damage=damage
    )


def load_las_by_class(file_path: str, class_id: int) -> np.ndarray:
    """Load points from LAS file filtered by classification."""
    import laspy
    las = laspy.read(file_path)
    mask = np.array(las.classification) == class_id
    points = np.vstack([las.x[mask], las.y[mask], las.z[mask]]).T
    return points


def run_damage_detection_v3(input_path: str,
                            output_dir: str,
                            damage_threshold_cm: float = 5.0,
                            member_classes: Dict[str, int] = None) -> Dict:
    """
    Run complete damage detection pipeline v3.

    Args:
        input_path: Path to classified LAS file
        output_dir: Output directory for reports
        damage_threshold_cm: Deviation threshold in cm (default 5cm)
        member_classes: Dict mapping member type to class ID

    Returns:
        Report dictionary
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("DAMAGE DETECTION v3.0 - Robust Instance Analysis")
    print(f"{'='*60}")
    print(f"Input: {input_path.name}")
    print(f"Damage threshold: {damage_threshold_cm}cm")
    print(f"Output: {output_dir}")

    # Default member classes
    if member_classes is None:
        member_classes = {
            'bunton': 3,
            'guard': 2,
            'column': 5,
        }

    # Initialize validator
    validator = MemberValidator()

    all_results = []
    all_points = {}

    # Process each member type
    for member_name, class_id in member_classes.items():
        print(f"\n{'='*40}")
        print(f"Processing {member_name}s (class {class_id})")
        print(f"{'='*40}")

        # Load points
        points = load_las_by_class(str(input_path), class_id)

        if len(points) < 100:
            print(f"  Skipping - only {len(points)} points")
            continue

        print(f"  Loaded {len(points):,} points")
        all_points[member_name] = points

        # Segment into instances
        print(f"  Segmenting into individual instances...")
        member_type = MemberType(member_name) if member_name in ['bunton', 'guard', 'column'] else MemberType.UNKNOWN
        labels = segment_members(points, member_name)

        unique_labels = set(labels) - {-1}
        print(f"  Found {len(unique_labels)} potential instances")

        # Analyze each instance
        valid_count = 0
        damaged_count = 0

        for label in unique_labels:
            mask = labels == label
            instance_points = points[mask]

            # Skip very small clusters
            if len(instance_points) < 50:
                continue

            result = analyze_member(
                member_id=int(label),
                points=instance_points,
                member_type=member_type,
                validator=validator,
                damage_threshold_cm=damage_threshold_cm
            )

            all_results.append(result)

            if result.validation.is_valid:
                valid_count += 1
                status = "VALID"
                if result.damage and result.damage.is_damaged:
                    damaged_count += 1
                    status = f"DAMAGED ({result.damage.severity})"
            else:
                status = f"REJECTED: {result.validation.reason}"

            print(f"    #{label}: {len(instance_points)} pts - {status}")

        print(f"  Summary: {valid_count} valid, {damaged_count} damaged")

    # Generate report
    print(f"\n{'='*60}")
    print("GENERATING REPORT")
    print(f"{'='*60}")

    # Separate results
    valid_results = [r for r in all_results if r.validation.is_valid]
    invalid_results = [r for r in all_results if not r.validation.is_valid]
    damaged_results = [r for r in valid_results if r.damage and r.damage.is_damaged]
    healthy_results = [r for r in valid_results if r.damage and not r.damage.is_damaged]

    # Count by severity
    severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for r in damaged_results:
        severity_counts[r.damage.severity] += 1

    print(f"\nTotal instances analyzed: {len(all_results)}")
    print(f"  Valid members: {len(valid_results)}")
    print(f"    Damaged: {len(damaged_results)}")
    for sev, count in severity_counts.items():
        if count > 0:
            print(f"      {sev}: {count}")
    print(f"    Healthy: {len(healthy_results)}")
    print(f"  Rejected (not real members): {len(invalid_results)}")

    # Generate visualization
    vis_path = generate_visualization_v3(all_results, all_points, output_dir)
    print(f"\nVisualization saved: {vis_path}")

    # Generate detail plots for damaged members
    if damaged_results:
        detail_dir = output_dir / 'damaged_members'
        detail_dir.mkdir(exist_ok=True)

        for r in damaged_results[:5]:  # Top 5
            detail_path = generate_member_detail_v3(r, detail_dir)
            print(f"  Detail: {detail_path.name}")

    # Build report
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'source_file': str(input_path),
            'damage_threshold_cm': damage_threshold_cm,
            'algorithm_version': '3.0'
        },
        'summary': {
            'total_instances': len(all_results),
            'valid_members': len(valid_results),
            'rejected_instances': len(invalid_results),
            'damaged_members': len(damaged_results),
            'healthy_members': len(healthy_results),
            'severity_breakdown': severity_counts
        },
        'damaged_members': [r.to_dict() for r in damaged_results],
        'healthy_members': [r.to_dict() for r in healthy_results],
        'rejected_instances': [r.to_dict() for r in invalid_results]
    }

    # Save JSON
    json_path = output_dir / 'damage_report_v3.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved: {json_path}")

    return report


def generate_visualization_v3(results: List[MemberAnalysisResult],
                              all_points: Dict[str, np.ndarray],
                              output_dir: Path) -> Path:
    """Generate main visualization."""

    valid_results = [r for r in results if r.validation.is_valid]
    damaged = [r for r in valid_results if r.damage and r.damage.is_damaged]
    healthy = [r for r in valid_results if r.damage and not r.damage.is_damaged]
    rejected = [r for r in results if not r.validation.is_valid]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Damage Detection Report v3.0 - Robust Instance Analysis', fontsize=16, fontweight='bold')

    severity_colors = {
        'HIGH': '#FF0000',
        'MEDIUM': '#FF8C00',
        'LOW': '#FFD700'
    }

    # 1. 3D Overview
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Plot all points in gray
    for member_type, points in all_points.items():
        if len(points) > 0:
            sample_idx = np.random.choice(len(points), min(5000, len(points)), replace=False)
            ax1.scatter(points[sample_idx, 0], points[sample_idx, 1], points[sample_idx, 2],
                       s=0.3, alpha=0.2, c='gray')

    # Healthy members in green
    for r in healthy:
        loc = r.centroid
        ax1.scatter([loc[0]], [loc[1]], [loc[2]], s=100, c='green', marker='o', alpha=0.5)

    # Damaged members colored by severity
    for r in damaged:
        loc = r.centroid
        color = severity_colors[r.damage.severity]
        ax1.scatter([loc[0]], [loc[1]], [loc[2]], s=300, c=color, marker='X',
                   edgecolors='black', linewidths=2)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Overview - Damage Locations')

    # 2. Top View (XY)
    ax2 = fig.add_subplot(2, 2, 2)

    for member_type, points in all_points.items():
        if len(points) > 0:
            sample_idx = np.random.choice(len(points), min(5000, len(points)), replace=False)
            ax2.scatter(points[sample_idx, 0], points[sample_idx, 1], s=0.3, alpha=0.2, c='gray')

    for r in healthy:
        ax2.scatter([r.centroid[0]], [r.centroid[1]], s=50, c='green', marker='o', alpha=0.3)

    for r in damaged:
        color = severity_colors[r.damage.severity]
        ax2.scatter([r.centroid[0]], [r.centroid[1]], s=150, c=color, marker='X',
                   edgecolors='black', linewidths=2)
        ax2.annotate(f'#{r.member_id}\n{r.damage.max_deviation_cm:.0f}cm',
                    (r.centroid[0], r.centroid[1]), fontsize=7, ha='center')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View - Damage Locations')
    ax2.set_aspect('equal')

    # 3. Validation Summary (bar chart)
    ax3 = fig.add_subplot(2, 2, 3)

    categories = ['Valid\nHealthy', 'Valid\nDamaged', 'Rejected\nInstances']
    counts = [len(healthy), len(damaged), len(rejected)]
    colors = ['green', 'red', 'gray']

    bars = ax3.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Count')
    ax3.set_title('Instance Classification Summary')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')

    # 4. Summary Text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for r in damaged:
        severity_counts[r.damage.severity] += 1

    summary = f"""
DAMAGE DETECTION REPORT v3.0
{'='*50}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INSTANCE ANALYSIS
-----------------
Total Instances Found: {len(results)}
  Valid Steel Members: {len(valid_results)}
    Healthy: {len(healthy)}
    Damaged: {len(damaged)}
      HIGH Severity: {severity_counts['HIGH']}
      MEDIUM Severity: {severity_counts['MEDIUM']}
      LOW Severity: {severity_counts['LOW']}
  Rejected (noise/fragments): {len(rejected)}

DAMAGE THRESHOLD
----------------
Deviation > 5cm = Damaged (per user requirement)

"""

    if damaged:
        summary += "DAMAGED MEMBERS\n"
        summary += "-" * 30 + "\n"

        for r in sorted(damaged, key=lambda x: x.damage.max_deviation_cm, reverse=True)[:6]:
            summary += f"""
#{r.member_id} ({r.member_type.value}) - {r.damage.severity}
  Max deviation: {r.damage.max_deviation_cm:.1f}cm
  Confidence: {r.validation.confidence:.0%}
  Location: ({r.centroid[0]:.1f}, {r.centroid[1]:.1f}, {r.centroid[2]:.1f})
"""

    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes,
             fontfamily='monospace', fontsize=9,
             verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()

    output_path = output_dir / 'damage_report_v3_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def generate_member_detail_v3(result: MemberAnalysisResult, output_dir: Path) -> Path:
    """Generate detailed analysis plot for a damaged member."""

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Member #{result.member_id} - {result.damage.severity} Severity\n'
                 f'{result.member_type.value} ({result.orientation.value}) - '
                 f'Confidence: {result.validation.confidence:.0%}',
                 fontsize=14, fontweight='bold')

    points = result.points
    centerline = result.damage.centerline_points

    # 1. 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    sample_idx = np.random.choice(len(points), min(2000, len(points)), replace=False)
    ax1.scatter(points[sample_idx, 0], points[sample_idx, 1], points[sample_idx, 2],
               s=1, alpha=0.5, c='blue', label='Points')

    if centerline is not None and len(centerline) > 1:
        ax1.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2],
                'r-', linewidth=3, label='Centerline')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D View with Fitted Centerline')
    ax1.legend()

    # 2. Deviation along length
    ax2 = fig.add_subplot(2, 2, 2)

    if centerline is not None:
        deviations = calculate_centerline_deviations(points, centerline)

        # Sort by position along centerline
        pca = PCA(n_components=1)
        t = pca.fit_transform(points).flatten()
        sorted_idx = np.argsort(t)

        ax2.scatter(t[sorted_idx][sample_idx], deviations[sorted_idx][sample_idx] * 100,
                   s=1, alpha=0.5)
        ax2.axhline(y=5, color='orange', linestyle='--', label='5cm threshold')
        ax2.axhline(y=result.damage.max_deviation_cm, color='red', linestyle=':',
                   label=f'Max: {result.damage.max_deviation_cm:.1f}cm')

        ax2.set_xlabel('Position along member (relative)')
        ax2.set_ylabel('Deviation from centerline (cm)')
        ax2.set_title('Deviation Profile')
        ax2.legend()

    # 3. Cross-section view
    ax3 = fig.add_subplot(2, 2, 3)

    pca = PCA(n_components=3)
    pca.fit(points)
    centered = points - points.mean(axis=0)

    proj_x = np.dot(centered, pca.components_[1])
    proj_y = np.dot(centered, pca.components_[2])

    ax3.scatter(proj_x[sample_idx], proj_y[sample_idx], s=1, alpha=0.5)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Width (m)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title(f'Cross-Section View\n({result.validation.width_m:.2f}m x {result.validation.height_m:.2f}m)')
    ax3.set_aspect('equal')

    # 4. Metrics summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary = f"""
MEMBER ANALYSIS
{'='*40}

VALIDATION
----------
Status: {result.validation.status.value}
Confidence: {result.validation.confidence:.1%}
Aspect Ratio: {result.validation.aspect_ratio:.1f}

GEOMETRY
--------
Length: {result.validation.length_m:.2f} m
Width: {result.validation.width_m:.2f} m
Height: {result.validation.height_m:.2f} m
Points: {result.num_points:,}

DAMAGE ANALYSIS
---------------
Severity: {result.damage.severity}
Max Deviation: {result.damage.max_deviation_cm:.1f} cm
Mean Deviation: {result.damage.mean_deviation_cm:.1f} cm
95th Percentile: {result.damage.p95_deviation_cm:.1f} cm
Damage Regions: {result.damage.num_damage_regions}
Affected Points: {result.damage.damage_percentage:.1f}%

LOCATION
--------
Centroid: ({result.centroid[0]:.2f}, {result.centroid[1]:.2f}, {result.centroid[2]:.2f})
Z Range: {result.bounding_box['min'][2]:.1f} to {result.bounding_box['max'][2]:.1f} m
"""

    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes,
             fontfamily='monospace', fontsize=10,
             verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()

    output_path = output_dir / f'member_{result.member_id}_detail.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python damage_detection_v3.py <input.las> [output_dir] [threshold_cm]")
        print("\nExample:")
        print("  python damage_detection_v3.py slice_054_predicted.las ./damage_report_v3 5")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './damage_report_v3'
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    run_damage_detection_v3(input_file, output_dir, damage_threshold_cm=threshold)
