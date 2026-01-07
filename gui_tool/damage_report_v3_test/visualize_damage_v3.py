#!/usr/bin/env python3
"""
Visualize damage detection v3 results.
Creates a visualization showing damaged vs healthy members.
"""

import json
import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load the damage report
    report_path = Path(__file__).parent / "damage_report_v3.json"
    with open(report_path) as f:
        report = json.load(f)

    # Load the original point cloud
    las_path = report['metadata']['source_file']
    las = laspy.read(las_path)
    points = np.vstack([las.x, las.y, las.z]).T

    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Class mapping
    class_names = {
        0: 'Background',
        1: 'Buntons',
        2: 'Columns',
        3: 'Guides',
        4: 'Guards',
        5: 'Pipes',
        6: 'Shaft Wall',
        7: 'Cables'
    }

    # Plot 1: Summary statistics
    ax1 = axes[0, 0]
    summary = report['summary']
    categories = ['Valid\nMembers', 'Damaged', 'Healthy', 'Rejected\n(noise)']
    values = [summary['valid_members'], summary['damaged_members'],
              summary['healthy_members'], summary['rejected_instances']]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
    bars = ax1.bar(categories, values, color=colors, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Damage Detection v3 Summary', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(values) * 1.15)

    # Plot 2: Severity breakdown
    ax2 = axes[0, 1]
    severity = summary['severity_breakdown']
    sev_labels = ['HIGH\n(>20cm)', 'MEDIUM\n(10-20cm)', 'LOW\n(5-10cm)']
    sev_values = [severity['HIGH'], severity['MEDIUM'], severity['LOW']]
    sev_colors = ['#c0392b', '#e67e22', '#f1c40f']
    bars2 = ax2.bar(sev_labels, sev_values, color=sev_colors, edgecolor='black')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Damage Severity Distribution', fontsize=14, fontweight='bold')
    for bar, val in zip(bars2, sev_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(sev_values + [1]) * 1.3)

    # Plot 3: Deviation histogram for damaged members
    ax3 = axes[1, 0]
    damaged_deviations = [m['damage']['max_deviation_cm'] for m in report['damaged_members']]
    healthy_deviations = [m['damage']['max_deviation_cm'] for m in report['healthy_members']]

    all_devs = damaged_deviations + healthy_deviations
    bins = np.linspace(0, max(all_devs) + 2, 15)

    ax3.hist(healthy_deviations, bins=bins, alpha=0.7, label='Healthy', color='#2ecc71', edgecolor='black')
    ax3.hist(damaged_deviations, bins=bins, alpha=0.7, label='Damaged', color='#e74c3c', edgecolor='black')
    ax3.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='5cm threshold')
    ax3.set_xlabel('Max Deviation (cm)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Deviation Distribution', fontsize=14, fontweight='bold')
    ax3.legend()

    # Plot 4: Top damaged members
    ax4 = axes[1, 1]

    # Sort by deviation
    damaged_sorted = sorted(report['damaged_members'],
                           key=lambda x: x['damage']['max_deviation_cm'],
                           reverse=True)[:10]

    labels = [f"{m['member_type'][:3]}_{m['member_id']}" for m in damaged_sorted]
    deviations = [m['damage']['max_deviation_cm'] for m in damaged_sorted]
    colors_bar = ['#c0392b' if d > 15 else '#e67e22' if d > 10 else '#f1c40f' for d in deviations]

    y_pos = np.arange(len(labels))
    ax4.barh(y_pos, deviations, color=colors_bar, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels)
    ax4.set_xlabel('Max Deviation (cm)', fontsize=12)
    ax4.set_title('Top 10 Most Damaged Members', fontsize=14, fontweight='bold')
    ax4.axvline(x=5.0, color='red', linestyle='--', linewidth=1.5, label='5cm threshold')

    # Add deviation values
    for i, (dev, label) in enumerate(zip(deviations, labels)):
        ax4.text(dev + 0.3, i, f'{dev:.1f}cm', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'damage_summary_v3.png', dpi=150, bbox_inches='tight')
    print("Saved damage_summary_v3.png")

    # Create detailed text report
    report_text = []
    report_text.append("=" * 70)
    report_text.append("SHAFT STEEL MEMBER DAMAGE DETECTION REPORT v3.0")
    report_text.append("=" * 70)
    report_text.append(f"\nSource: {report['metadata']['source_file']}")
    report_text.append(f"Timestamp: {report['metadata']['timestamp']}")
    report_text.append(f"Damage threshold: {report['metadata']['damage_threshold_cm']}cm")
    report_text.append(f"Algorithm version: {report['metadata']['algorithm_version']}")

    report_text.append("\n" + "-" * 70)
    report_text.append("SUMMARY")
    report_text.append("-" * 70)
    report_text.append(f"Total instances analyzed: {summary['total_instances']}")
    report_text.append(f"  Valid steel members: {summary['valid_members']}")
    report_text.append(f"    Damaged: {summary['damaged_members']}")
    report_text.append(f"    Healthy: {summary['healthy_members']}")
    report_text.append(f"  Rejected (noise/fragments): {summary['rejected_instances']}")
    report_text.append(f"\nSeverity breakdown:")
    report_text.append(f"  HIGH (>20cm): {severity['HIGH']}")
    report_text.append(f"  MEDIUM (10-20cm): {severity['MEDIUM']}")
    report_text.append(f"  LOW (5-10cm): {severity['LOW']}")

    report_text.append("\n" + "-" * 70)
    report_text.append("DAMAGED MEMBERS (sorted by severity)")
    report_text.append("-" * 70)

    for m in damaged_sorted:
        report_text.append(f"\n{m['member_type'].upper()} #{m['member_id']} - {m['damage']['severity']}")
        report_text.append(f"  Max deviation: {m['damage']['max_deviation_cm']:.1f}cm")
        report_text.append(f"  Mean deviation: {m['damage']['mean_deviation_cm']:.1f}cm")
        report_text.append(f"  Damage percentage: {m['damage']['damage_percentage']:.1f}%")
        report_text.append(f"  Confidence: {m['validation']['confidence']:.2f}")
        report_text.append(f"  Length: {m['validation']['geometry']['length_m']:.2f}m")
        report_text.append(f"  Centroid: ({m['centroid'][0]:.2f}, {m['centroid'][1]:.2f}, {m['centroid'][2]:.2f})")

    report_text.append("\n" + "-" * 70)
    report_text.append("HEALTHY MEMBERS")
    report_text.append("-" * 70)

    for m in report['healthy_members']:
        report_text.append(f"\n{m['member_type'].upper()} #{m['member_id']}")
        report_text.append(f"  Max deviation: {m['damage']['max_deviation_cm']:.1f}cm (below threshold)")
        report_text.append(f"  Confidence: {m['validation']['confidence']:.2f}")
        report_text.append(f"  Length: {m['validation']['geometry']['length_m']:.2f}m")

    report_text.append("\n" + "-" * 70)
    report_text.append("REJECTED INSTANCES (not valid steel members)")
    report_text.append("-" * 70)

    rejection_counts = {}
    for m in report['rejected_instances']:
        reason = m['validation']['status']
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for reason, count in sorted(rejection_counts.items(), key=lambda x: -x[1]):
        report_text.append(f"  {reason}: {count}")

    report_text.append("\n" + "=" * 70)
    report_text.append("END OF REPORT")
    report_text.append("=" * 70)

    report_file = Path(__file__).parent / 'damage_report_v3.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_text))
    print(f"Saved {report_file}")

if __name__ == '__main__':
    main()
