#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
import csv

def load_weakness_data():
    """Load the weakness comparison data"""
    try:
        with open('weakness_comparison.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("Error: weakness_comparison.json not found. Please run extract_continuous_weaknesses.py first.")
        return None

def create_dove_weakness_ranking():
    """Create ordered list of most weak paths based on dove score"""
    data = load_weakness_data()
    if not data:
        return None
    
    # Extract dove ranking weak paths
    dove_paths = data['dove_ranking_analysis']['weak_paths']
    
    # Sort by mean score (lowest = weakest)
    dove_paths_sorted = sorted(dove_paths, key=lambda x: x['mean_score'])
    
    # Create simplified structure for JSON output
    dove_weakness_ranking = []
    for i, path in enumerate(dove_paths_sorted, 1):
        # Get the final capability name
        final_capability = path['capability_path'][-1] if path.get('capability_path') else path.get('capability', 'Unknown')
        
        # Truncate long capability names
        if len(final_capability) > 80:
            display_capability = final_capability[:77] + '...'
        else:
            display_capability = final_capability
            
        weakness_entry = {
            'rank': i,
            'capability': display_capability,
            'full_capability_path': path.get('capability_path', []),
            'mean_score': path['mean_score'],
            'threshold_diff': path['threshold_diff'],
            'size': path['size'],
            'weakness_severity': abs(path['threshold_diff'])  # How far below threshold
        }
        dove_weakness_ranking.append(weakness_entry)
    
    # Save to JSON
    with open('dove_weakness_ranking.json', 'w') as f:
        json.dump({
            'methodology': 'Weak paths ordered by dove score (lowest mean score = weakest)',
            'threshold': data['dove_ranking_analysis']['parameters']['global_threshold_tau'],
            'total_weak_paths': len(dove_weakness_ranking),
            'weak_paths_ranked': dove_weakness_ranking
        }, f, indent=2)
    
    return dove_weakness_ranking

def plot_dove_weakness_ranking(dove_ranking):
    """Create a clear plot showing the most weak paths based on dove score"""
    if not dove_ranking:
        return
    
    # Take top 15 weakest for readability
    top_weak = dove_ranking[:15]
    
    # Prepare data for plotting
    capabilities = [item['capability'] for item in top_weak]
    scores = [item['mean_score'] for item in top_weak]
    sizes = [item['size'] for item in top_weak]
    threshold = 0.5  # Approximate threshold for visualization
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(capabilities))
    bars = ax.barh(y_pos, scores, color='#FF6B6B', alpha=0.7)
    
    # Add threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (~{threshold})')
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{i+1}. {cap[:50]}{'...' if len(cap) > 50 else ''}" 
                       for i, cap in enumerate(capabilities)], fontsize=10)
    ax.set_xlabel('Dove Score (Mean)', fontsize=12, fontweight='bold')
    ax.set_title('Top 7 Weakest Capabilities by Dove Score\n(Lower Score = Weaker Performance)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add score labels on bars
    for i, (bar, score, size) in enumerate(zip(bars, scores, sizes)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f} (n={size})', va='center', fontsize=9)
    
    # Add legend and grid
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(scores) * 1.2)
    
    plt.tight_layout()
    plt.savefig('dove_weakness_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Dove weakness ranking plot saved as dove_weakness_ranking.png")

def create_weakness_comparison_csv():
    """Create CSV file with detailed comparison table"""
    data = load_weakness_data()
    if not data:
        return
    
    # Extract weakness data
    dove_paths = data['dove_ranking_analysis']['weak_paths']
    ranking_paths = data['original_ranking_analysis']['weak_paths']
    
    # Create detailed capability info
    dove_capability_info = {}
    ranking_capability_info = {}
    
    for path in dove_paths:
        final_cap = path['capability_path'][-1] if path.get('capability_path') else path.get('capability', 'Unknown')
        dove_capability_info[final_cap] = {
            'mean_score': path['mean_score'],
            'size': path['size'],
            'threshold_diff': path['threshold_diff'],
            'path_id': path['path'],
            'full_capability_path': path.get('capability_path', [])
        }
    
    for path in ranking_paths:
        final_cap = path['capability_path'][-1] if path.get('capability_path') else path.get('capability', 'Unknown')
        ranking_capability_info[final_cap] = {
            'mean_score': path['mean_score'],
            'size': path['size'],
            'threshold_diff': path['threshold_diff'],
            'path_id': path['path'],
            'full_capability_path': path.get('capability_path', [])
        }
    
    # Create comparison categories
    dove_capabilities = set(dove_capability_info.keys())
    ranking_capabilities = set(ranking_capability_info.keys())
    both_weak = dove_capabilities & ranking_capabilities
    dove_only = dove_capabilities - ranking_capabilities
    ranking_only = ranking_capabilities - dove_capabilities
    
    # Prepare CSV data
    csv_data = []
    headers = [
        'Category', 'Capability', 'Agreement_Type',
        'Dove_Score', 'Dove_Size', 'Dove_Threshold_Diff', 'Dove_Path_ID',
        'Ranking_Score', 'Ranking_Size', 'Ranking_Threshold_Diff', 'Ranking_Path_ID',
        'Full_Capability_Path'
    ]
    
    # Add both agree capabilities
    for cap in sorted(both_weak):
        dove_info = dove_capability_info[cap]
        ranking_info = ranking_capability_info[cap]
        csv_data.append([
            'Both Agree', cap, '🟢 Both',
            f"{dove_info['mean_score']:.3f}", dove_info['size'], f"{dove_info['threshold_diff']:.3f}", dove_info['path_id'],
            f"{ranking_info['mean_score']:.3f}", ranking_info['size'], f"{ranking_info['threshold_diff']:.3f}", ranking_info['path_id'],
            ' | '.join(dove_info['full_capability_path'])
        ])
    
    # Add dove only capabilities
    for cap in sorted(dove_only):
        dove_info = dove_capability_info[cap]
        csv_data.append([
            'Dove Only', cap, '🔴 Dove Only',
            f"{dove_info['mean_score']:.3f}", dove_info['size'], f"{dove_info['threshold_diff']:.3f}", dove_info['path_id'],
            'N/A', 'N/A', 'N/A', 'N/A',
            ' | '.join(dove_info['full_capability_path'])
        ])
    
    # Add ranking only capabilities
    for cap in sorted(ranking_only):
        ranking_info = ranking_capability_info[cap]
        csv_data.append([
            'EvalTree Only', cap, '🔵 EvalTree Only',
            'N/A', 'N/A', 'N/A', 'N/A',
            f"{ranking_info['mean_score']:.3f}", ranking_info['size'], f"{ranking_info['threshold_diff']:.3f}", ranking_info['path_id'],
            ' | '.join(ranking_info['full_capability_path'])
        ])
    
    # Save to CSV
    with open('weakness_comparison_table.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_data)
    
    print("✅ Detailed comparison table saved to weakness_comparison_table.csv")
    
    # Also save summary statistics
    summary_data = {
        'summary': {
            'total_unique_capabilities': len(dove_capabilities | ranking_capabilities),
            'agreement_rate': len(both_weak) / len(dove_capabilities | ranking_capabilities),
            'dove_detections': len(dove_capabilities),
            'ranking_detections': len(ranking_capabilities),
            'both_agree': len(both_weak),
            'dove_only': len(dove_only),
            'ranking_only': len(ranking_only)
        },
        'thresholds': {
            'dove_global_threshold': data['dove_ranking_analysis']['parameters']['global_threshold_tau'],
            'ranking_global_threshold': data['original_ranking_analysis']['parameters']['global_threshold_tau']
        }
    }
    
    with open('weakness_comparison_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print("✅ Summary statistics saved to weakness_comparison_summary.json")
    
    return csv_data

def create_weakness_comparison_plot():
    """Create simple comparison plot between dove score and regular ranking weaknesses"""
    data = load_weakness_data()
    if not data:
        return
    
    # Extract weakness data
    dove_paths = data['dove_ranking_analysis']['weak_paths']
    ranking_paths = data['original_ranking_analysis']['weak_paths']
    
    # Extract final capabilities for comparison
    dove_capabilities = set()
    ranking_capabilities = set()
    
    for path in dove_paths:
        final_cap = path['capability_path'][-1] if path.get('capability_path') else path.get('capability', 'Unknown')
        if len(final_cap) > 60:
            final_cap = final_cap[:57] + '...'
        dove_capabilities.add(final_cap)
    
    for path in ranking_paths:
        final_cap = path['capability_path'][-1] if path.get('capability_path') else path.get('capability', 'Unknown')
        if len(final_cap) > 60:
            final_cap = final_cap[:57] + '...'
        ranking_capabilities.add(final_cap)
    
    # Create comparison categories
    both_weak = dove_capabilities & ranking_capabilities
    dove_only = dove_capabilities - ranking_capabilities
    ranking_only = ranking_capabilities - dove_capabilities
    
    # Create visualization with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Venn-style comparison
    categories = ['Both Methods\nAgree', 'Dove Score\nOnly', 'Regular Ranking\nOnly']
    counts = [len(both_weak), len(dove_only), len(ranking_only)]
    colors = ['#9B59B6', '#FF6B6B', '#4ECDC4']
    
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Number of Weak Capabilities', fontsize=12, fontweight='bold')
    ax1.set_title('Weakness Detection Comparison\n(Dove Score vs Regular Ranking)', 
                 fontsize=14, fontweight='bold')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Agreement rate pie chart
    total_unique = len(dove_capabilities | ranking_capabilities)
    agreement_rate = len(both_weak) / total_unique if total_unique > 0 else 0
    
    pie_data = [len(both_weak), len(dove_only), len(ranking_only)]
    pie_labels = [f'Both Agree\n({len(both_weak)})', 
                 f'RobustTree Only\n({len(dove_only)})',
                 f'BinaryScore EvalTree Only\n({len(ranking_only)})']
    
    wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Agreement Distribution\n(Total: {total_unique} capabilities, {agreement_rate:.1%} agreement)', 
                 fontsize=14, fontweight='bold')
    
    # Add summary statistics as text
    summary_text = f"""
    Summary Statistics:
    • Total Weak Capabilities: {total_unique}
    • Agreement Rate: {agreement_rate:.1%}
    • Dove Score Detections: {len(dove_capabilities)}
    • Regular Ranking Detections: {len(ranking_capabilities)}
    • Global Threshold (Dove): {data['dove_ranking_analysis']['parameters']['global_threshold_tau']:.3f}
    • Global Threshold (Ranking): {data['original_ranking_analysis']['parameters']['global_threshold_tau']:.3f}
    """
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('weakness_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Weakness comparison plot saved as weakness_comparison_plot.png")
    
    # Print detailed comparison
    print("\n" + "="*60)
    print("DETAILED WEAKNESS COMPARISON")
    print("="*60)
    
    print(f"\n🟢 BOTH METHODS AGREE ({len(both_weak)} capabilities):")
    for i, cap in enumerate(sorted(both_weak), 1):
        print(f"  {i}. {cap}")
    
    print(f"\n🔴 DOVE SCORE ONLY ({len(dove_only)} capabilities):")
    for i, cap in enumerate(sorted(dove_only), 1):
        print(f"  {i}. {cap}")
    
    print(f"\n🔵 EVALTREE RANKING ONLY ({len(ranking_only)} capabilities):")
    for i, cap in enumerate(sorted(ranking_only), 1):
        print(f"  {i}. {cap}")

def main():
    print("Creating weakness analysis plots...")
    print("="*50)
    
    # Create dove weakness ranking
    print("\n1️⃣ Creating dove weakness ranking...")
    dove_ranking = create_dove_weakness_ranking()
    if dove_ranking:
        print(f"✅ Dove weakness ranking saved to dove_weakness_ranking.json ({len(dove_ranking)} weak paths)")
        plot_dove_weakness_ranking(dove_ranking)
    
    # Create comparison plot
    print("\n2️⃣ Creating weakness comparison plot...")
    create_weakness_comparison_plot()
    
    # Create comparison CSV table
    print("\n3️⃣ Creating comparison CSV table...")
    create_weakness_comparison_csv()
    
    print("\n🎉 All plots and data files created successfully!")
    print("Files generated:")
    print("  • dove_weakness_ranking.json - Ordered list of weakest paths")
    print("  • dove_weakness_ranking.png - Visual ranking of weakest capabilities")
    print("  • weakness_comparison_plot.png - Comparison charts between scoring methods")
    print("  • weakness_comparison_table.csv - Detailed comparison table")
    print("  • weakness_comparison_summary.json - Summary statistics")

if __name__ == "__main__":
    main() 