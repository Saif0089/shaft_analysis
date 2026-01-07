#!/usr/bin/env python3
"""
Steel Member Damage Detection v4.0

Detects bent/damaged steel members from segmented PLY family files.
Designed to work with output from shaft_segmentation_cli.py

Input: classified_families/*.ply (buntons.ply, columns.ply, guards.ply, etc.)
Output: JSON damage report with visualization

Usage:
    python damage_detection_v4.py --input classified_families/ --output damage_report/
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.decomposition import PCA
import hdbscan

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PLY file reading
try:
    from plyfile import PlyData
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False
    print("Warning: plyfile not installed. Install with: pip install plyfile")


@dataclass
class MemberAnalysis:
    """Analysis result for a single steel member instance."""
    member_id: int
    member_type: str  # bunton, column, guard, etc.
    n_points: int

    # Geometry
    length_m: float
    width_m: float
    height_m: float
    centroid: Tuple[float, float, float]
    is_horizontal: bool

    # Damage assessment
    max_deviation_cm: float
    mean_deviation_cm: float
    is_damaged: bool
    damage_severity: str  # NONE, LOW, MEDIUM, HIGH, SEVERE

    # Validation
    is_valid_member: bool
    rejection_reason: Optional[str]


def load_ply_points(ply_path: str) -> np.ndarray:
    """Load points from a PLY file."""
    if not HAS_PLYFILE:
        raise ImportError("plyfile library required")

    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    return points


def cluster_steel_members(points: np.ndarray, member_type: str = 'bunton') -> np.ndarray:
    """
    Cluster point cloud into individual steel member instances using HDBSCAN.

    Args:
        points: (N, 3) array of points
        member_type: Type of member for parameter tuning

    Returns:
        labels: (N,) array of cluster labels (-1 = noise)
    """
    if len(points) < 100:
        return np.full(len(points), -1)

    # HDBSCAN parameters based on member type
    if member_type in ['bunton', 'guard']:
        min_cluster_size = 100
        min_samples = 20
    elif member_type == 'column':
        min_cluster_size = 200
        min_samples = 30
    else:
        min_cluster_size = 100
        min_samples = 20

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom'
    )

    labels = clusterer.fit_predict(points)
    return labels


def validate_member(points: np.ndarray, member_type: str = 'bunton') -> Tuple[bool, str]:
    """
    Validate if a cluster is a real steel member based on geometry.

    More strict validation to avoid false positives on fragments.

    Returns:
        (is_valid, rejection_reason)
    """
    # Stricter requirements for buntons
    if member_type == 'buntons':
        MIN_POINTS = 500  # Need substantial points for reliable measurement
        MIN_LENGTH = 0.8  # 80cm minimum - real buntons span significant distance
    else:
        MIN_POINTS = 300
        MIN_LENGTH = 0.5

    MAX_CROSS_SECTION = 2.0  # 2m max cross-section

    # Point count
    if len(points) < MIN_POINTS:
        return False, f"Too few points ({len(points)} < {MIN_POINTS})"

    # Get dimensions
    x_extent = points[:,0].max() - points[:,0].min()
    y_extent = points[:,1].max() - points[:,1].min()
    z_extent = points[:,2].max() - points[:,2].min()

    # For buntons - must be horizontal and long enough
    if member_type == 'buntons':
        horiz_extent = max(x_extent, y_extent)

        # Must be predominantly horizontal
        if z_extent > horiz_extent * 0.5:
            return False, f"Not horizontal (Z={z_extent:.2f}m vs horiz={horiz_extent:.2f}m)"

        # Must have significant horizontal span
        if horiz_extent < MIN_LENGTH:
            return False, f"Too short ({horiz_extent:.2f}m < {MIN_LENGTH}m)"
    else:
        # For columns/guards - check overall length
        max_extent = max(x_extent, y_extent, z_extent)
        if max_extent < MIN_LENGTH:
            return False, f"Too short ({max_extent:.2f}m < {MIN_LENGTH}m)"

    return True, "Valid"


def measure_centerline_deviation(points: np.ndarray, n_segments: int = 20) -> Dict:
    """
    Measure how bent a steel member is by fitting a centerline and
    calculating deviation from an ideal straight line.

    Returns:
        dict with max_deviation, mean_deviation, centerline points
    """
    if len(points) < 50:
        return {
            'max_deviation': 0,
            'mean_deviation': 0,
            'centerline': points.mean(axis=0).reshape(1, -1)
        }

    # Use PCA to find the principal axis
    pca = PCA(n_components=3)
    pca.fit(points)

    # Project points onto principal axis and sort
    t = pca.transform(points)[:, 0]
    sorted_idx = np.argsort(t)
    sorted_pts = points[sorted_idx]

    # Build centerline using segment medians (robust to noise)
    n_segments = min(n_segments, max(3, len(sorted_pts) // 30))
    segment_size = len(sorted_pts) // n_segments

    centerline = []
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(sorted_pts)
        segment = sorted_pts[start:end]
        centerline.append(np.median(segment, axis=0))
    centerline = np.array(centerline)

    # Measure deviation from ideal straight line (endpoints)
    if len(centerline) < 2:
        return {
            'max_deviation': 0,
            'mean_deviation': 0,
            'centerline': centerline
        }

    start_pt = centerline[0]
    end_pt = centerline[-1]
    line_vec = end_pt - start_pt
    line_len = np.linalg.norm(line_vec)

    if line_len < 0.1:  # Too short to measure
        return {
            'max_deviation': 0,
            'mean_deviation': 0,
            'centerline': centerline
        }

    line_dir = line_vec / line_len

    # Calculate perpendicular distance from each centerline point to ideal line
    deviations = []
    for pt in centerline:
        v = pt - start_pt
        proj_len = np.dot(v, line_dir)
        proj_pt = start_pt + proj_len * line_dir
        perp_dist = np.linalg.norm(pt - proj_pt)
        deviations.append(perp_dist)

    return {
        'max_deviation': float(max(deviations)),
        'mean_deviation': float(np.mean(deviations)),
        'centerline': centerline,
        'deviations': deviations,
        'length': line_len
    }


def classify_damage_severity(max_deviation_cm: float, threshold_cm: float = 10.0) -> str:
    """
    Classify damage severity based on deviation.

    Default threshold is 10cm - visible structural deformation.
    Below threshold is considered normal variation/scan noise.
    """
    if max_deviation_cm < threshold_cm:
        return 'NONE'
    elif max_deviation_cm < threshold_cm * 2:  # 10-20cm
        return 'LOW'
    elif max_deviation_cm < threshold_cm * 3:  # 20-30cm
        return 'MEDIUM'
    elif max_deviation_cm < threshold_cm * 5:  # 30-50cm
        return 'HIGH'
    else:  # >50cm
        return 'SEVERE'


def analyze_member(points: np.ndarray, member_id: int, member_type: str) -> MemberAnalysis:
    """
    Perform complete damage analysis on a single steel member instance.
    """
    # Validate member
    is_valid, rejection_reason = validate_member(points, member_type)

    # Get dimensions
    x_extent = points[:,0].max() - points[:,0].min()
    y_extent = points[:,1].max() - points[:,1].min()
    z_extent = points[:,2].max() - points[:,2].min()

    centroid = tuple(points.mean(axis=0).tolist())

    # Determine orientation
    horiz_extent = max(x_extent, y_extent)
    is_horizontal = horiz_extent > z_extent * 1.2

    # Measure deviation
    if is_valid:
        deviation_result = measure_centerline_deviation(points)
        max_dev_m = deviation_result['max_deviation']
        mean_dev_m = deviation_result['mean_deviation']
        length = deviation_result.get('length', horiz_extent if is_horizontal else z_extent)
    else:
        max_dev_m = 0
        mean_dev_m = 0
        length = max(x_extent, y_extent, z_extent)

    max_dev_cm = max_dev_m * 100
    mean_dev_cm = mean_dev_m * 100

    # Classify damage
    severity = classify_damage_severity(max_dev_cm) if is_valid else 'UNKNOWN'
    is_damaged = severity not in ['NONE', 'UNKNOWN']

    return MemberAnalysis(
        member_id=member_id,
        member_type=member_type,
        n_points=len(points),
        length_m=length,
        width_m=min(x_extent, y_extent) if is_horizontal else min(x_extent, y_extent),
        height_m=z_extent if is_horizontal else max(x_extent, y_extent),
        centroid=centroid,
        is_horizontal=is_horizontal,
        max_deviation_cm=max_dev_cm,
        mean_deviation_cm=mean_dev_cm,
        is_damaged=is_damaged,
        damage_severity=severity,
        is_valid_member=is_valid,
        rejection_reason=None if is_valid else rejection_reason
    )


def analyze_family_file(ply_path: str, member_type: str) -> List[MemberAnalysis]:
    """
    Analyze a single family PLY file (e.g., buntons.ply).

    Returns list of MemberAnalysis for each instance found.
    """
    points = load_ply_points(ply_path)

    if len(points) == 0:
        return []

    # Cluster into individual instances
    labels = cluster_steel_members(points, member_type)

    results = []
    for label in sorted(set(labels)):
        if label == -1:  # Noise
            continue

        mask = labels == label
        member_points = points[mask]

        analysis = analyze_member(member_points, label, member_type)
        results.append(analysis)

    return results


def create_visualization(all_results: Dict[str, List[MemberAnalysis]],
                        output_dir: Path,
                        family_points: Dict[str, np.ndarray] = None,
                        family_labels: Dict[str, np.ndarray] = None):
    """Create visualization of damage detection results."""

    # Summary figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Count statistics
    total_members = 0
    damaged_count = 0
    valid_count = 0
    severity_counts = {'NONE': 0, 'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'SEVERE': 0}

    for member_type, results in all_results.items():
        for r in results:
            total_members += 1
            if r.is_valid_member:
                valid_count += 1
                if r.is_damaged:
                    damaged_count += 1
                severity_counts[r.damage_severity] = severity_counts.get(r.damage_severity, 0) + 1

    # Plot 1: Summary bar chart
    ax = axes[0, 0]
    categories = ['Total\nInstances', 'Valid\nMembers', 'Damaged', 'Healthy']
    healthy_count = valid_count - damaged_count
    values = [total_members, valid_count, damaged_count, healthy_count]
    colors = ['#3498db', '#9b59b6', '#e74c3c', '#2ecc71']
    bars = ax.bar(categories, values, color=colors, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Damage Detection Summary')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(val), ha='center', fontweight='bold')

    # Plot 2: Severity breakdown
    ax = axes[0, 1]
    sev_labels = ['NONE', 'LOW', 'MEDIUM', 'HIGH', 'SEVERE']
    sev_values = [severity_counts.get(s, 0) for s in sev_labels]
    sev_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
    bars = ax.bar(sev_labels, sev_values, color=sev_colors, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Damage Severity Distribution')
    for bar, val in zip(bars, sev_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   str(val), ha='center', fontweight='bold')

    # Plot 3: Deviation histogram
    ax = axes[1, 0]
    all_deviations = []
    for results in all_results.values():
        for r in results:
            if r.is_valid_member:
                all_deviations.append(r.max_deviation_cm)

    if all_deviations:
        bins = np.linspace(0, min(max(all_deviations) + 5, 100), 20)
        ax.hist(all_deviations, bins=bins, color='#3498db', edgecolor='black', alpha=0.7)
        ax.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='5cm threshold')
        ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='10cm threshold')
    ax.set_xlabel('Max Deviation (cm)')
    ax.set_ylabel('Count')
    ax.set_title('Deviation Distribution')
    ax.legend()

    # Plot 4: By member type
    ax = axes[1, 1]
    type_data = {}
    for member_type, results in all_results.items():
        damaged = sum(1 for r in results if r.is_valid_member and r.is_damaged)
        healthy = sum(1 for r in results if r.is_valid_member and not r.is_damaged)
        type_data[member_type] = {'damaged': damaged, 'healthy': healthy}

    types = list(type_data.keys())
    x = np.arange(len(types))
    width = 0.35

    damaged_vals = [type_data[t]['damaged'] for t in types]
    healthy_vals = [type_data[t]['healthy'] for t in types]

    ax.bar(x - width/2, damaged_vals, width, label='Damaged', color='#e74c3c')
    ax.bar(x + width/2, healthy_vals, width, label='Healthy', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylabel('Count')
    ax.set_title('Damage by Member Type')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'damage_summary.png', dpi=150)
    plt.close()

    print(f"Saved damage_summary.png")


def generate_report(all_results: Dict[str, List[MemberAnalysis]],
                   output_dir: Path,
                   input_dir: str) -> Dict:
    """Generate JSON damage report."""

    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'source_directory': str(input_dir),
            'algorithm_version': '4.0',
            'damage_threshold_cm': 5.0
        },
        'summary': {
            'total_instances': 0,
            'valid_members': 0,
            'damaged_members': 0,
            'healthy_members': 0,
            'rejected_instances': 0,
            'severity_breakdown': {
                'NONE': 0,
                'LOW': 0,
                'MEDIUM': 0,
                'HIGH': 0,
                'SEVERE': 0
            }
        },
        'damaged_members': [],
        'healthy_members': [],
        'rejected_instances': []
    }

    for member_type, results in all_results.items():
        for r in results:
            report['summary']['total_instances'] += 1

            member_data = {
                'member_id': int(r.member_id),
                'member_type': member_type,
                'n_points': int(r.n_points),
                'centroid': [float(c) for c in r.centroid],
                'dimensions': {
                    'length_m': float(round(r.length_m, 3)),
                    'width_m': float(round(r.width_m, 3)),
                    'height_m': float(round(r.height_m, 3))
                },
                'is_horizontal': bool(r.is_horizontal),
                'damage': {
                    'max_deviation_cm': float(round(r.max_deviation_cm, 2)),
                    'mean_deviation_cm': float(round(r.mean_deviation_cm, 2)),
                    'severity': r.damage_severity,
                    'is_damaged': r.is_damaged
                }
            }

            if not r.is_valid_member:
                member_data['rejection_reason'] = r.rejection_reason
                report['rejected_instances'].append(member_data)
                report['summary']['rejected_instances'] += 1
            elif r.is_damaged:
                report['damaged_members'].append(member_data)
                report['summary']['damaged_members'] += 1
                report['summary']['valid_members'] += 1
                report['summary']['severity_breakdown'][r.damage_severity] += 1
            else:
                report['healthy_members'].append(member_data)
                report['summary']['healthy_members'] += 1
                report['summary']['valid_members'] += 1
                report['summary']['severity_breakdown']['NONE'] += 1

    # Sort damaged members by severity
    severity_order = {'SEVERE': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    report['damaged_members'].sort(
        key=lambda x: (severity_order.get(x['damage']['severity'], 99), -x['damage']['max_deviation_cm'])
    )

    return report


def main():
    parser = argparse.ArgumentParser(description='Steel Member Damage Detection v4.0')
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory containing family PLY files')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for damage report')
    parser.add_argument('--threshold', '-t', type=float, default=5.0,
                       help='Damage threshold in cm (default: 5.0)')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define which family files to analyze
    family_files = {
        'buntons': 'buntons.ply',
        'columns': 'columns.ply',
        'guards': 'guards.ply',
        'columns_secondary': 'columns_secondary.ply'
    }

    print("=" * 60)
    print("STEEL MEMBER DAMAGE DETECTION v4.0")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Threshold: {args.threshold}cm")
    print()

    all_results = {}

    for member_type, filename in family_files.items():
        ply_path = input_dir / filename

        if not ply_path.exists():
            print(f"Skipping {member_type}: {filename} not found")
            continue

        print(f"Analyzing {member_type}...")
        results = analyze_family_file(str(ply_path), member_type)
        all_results[member_type] = results

        # Summary for this type
        valid = [r for r in results if r.is_valid_member]
        damaged = [r for r in valid if r.is_damaged]
        print(f"  Found {len(results)} instances, {len(valid)} valid, {len(damaged)} damaged")

    print()
    print("Generating report and visualizations...")

    # Generate report
    report = generate_report(all_results, output_dir, str(input_dir))

    # Save JSON report
    report_path = output_dir / 'damage_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved {report_path}")

    # Create visualizations
    create_visualization(all_results, output_dir)

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    s = report['summary']
    print(f"Total instances analyzed: {s['total_instances']}")
    print(f"  Valid members: {s['valid_members']}")
    print(f"    Damaged: {s['damaged_members']}")
    print(f"    Healthy: {s['healthy_members']}")
    print(f"  Rejected (noise): {s['rejected_instances']}")
    print()
    print("Severity breakdown:")
    for sev, count in s['severity_breakdown'].items():
        if count > 0:
            print(f"  {sev}: {count}")

    # Print top damaged members
    if report['damaged_members']:
        print()
        print("Top damaged members:")
        for m in report['damaged_members'][:5]:
            print(f"  {m['member_type']} #{m['member_id']}: {m['damage']['max_deviation_cm']:.1f}cm ({m['damage']['severity']})")

    print()
    print("=" * 60)
    print("DAMAGE DETECTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
