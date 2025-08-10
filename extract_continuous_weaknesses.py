#!/usr/bin/env python3
import json
import math
import statistics

class ContinuousWeaknessExtractor:
    def __init__(self, min_node_size_sigma1=50, min_child_size_sigma2=30, confidence_level=0.05, use_statistical_test=True):
        self.sigma1 = min_node_size_sigma1  # minimum node size for extraction
        self.sigma2 = min_child_size_sigma2  # minimum child size for consideration
        self.alpha = confidence_level  # confidence level for statistical tests
        self.use_statistical_test = use_statistical_test  # whether to use t-test or simple comparison
        self.tau = None  # threshold (will be set to global mean)
        self.weak_nodes = {}  # stores weakness test results for each node
        self.extracted_paths = []  # final result set of weak path roots
        
    def get_continuous_scores(self, node, score_type='dove_ranking'):
        """Recursively collect all continuous scores under a node"""
        scores = []
        
        def collect_scores(n):
            if isinstance(n, dict):
                if n.get('size') == 1:  # leaf node
                    score = n.get(score_type)
                    if score is not None:
                        scores.append(float(score))
                elif 'subtrees' in n:
                    if isinstance(n['subtrees'], list):
                        for subtree in n['subtrees']:
                            collect_scores(subtree)
                    elif isinstance(n['subtrees'], dict):
                        for subtree in n['subtrees'].values():
                            collect_scores(subtree)
        
        collect_scores(node)
        return scores
    
    def get_node_size(self, node, score_type='dove_ranking'):
        """Get the number of valid scores under a node"""
        return len(self.get_continuous_scores(node, score_type))
    
    def calculate_global_threshold(self, tree_data, score_type='dove_ranking'):
        """Calculate global mean as threshold τ"""
        all_scores = self.get_continuous_scores(tree_data, score_type)
        if not all_scores:
            return 0.5  # fallback
        
        self.tau = statistics.mean(all_scores)
        print(f"Global mean threshold (τ): {self.tau:.4f}")
        print(f"Total valid scores: {len(all_scores)}")
        return self.tau
    
    def perform_weakness_test(self, scores):
        """
        Test if node scores are significantly below threshold τ
        Returns True if the node is considered "weak"
        """
        if not scores:
            return False
        
        node_mean = statistics.mean(scores)
        n = len(scores)
        
        if not self.use_statistical_test:
            # Simple heuristic: weak if mean < τ
            return node_mean < self.tau
        
        # Statistical test: one-sample t-test H0: mean >= τ, H1: mean < τ
        if n == 1:
            return node_mean < self.tau
        
        if n < 3:  # Too few samples for reliable t-test
            return node_mean < (self.tau - 0.05)  # More conservative threshold
        
        # Calculate t-statistic
        sample_std = statistics.stdev(scores)
        if sample_std == 0:
            return node_mean < self.tau
        
        t_stat = (node_mean - self.tau) / (sample_std / math.sqrt(n))
        
        # Critical value for one-tailed test (approximate)
        if self.alpha == 0.05:
            critical_t = -1.645  # For large samples
        elif self.alpha == 0.01:
            critical_t = -2.326
        else:
            critical_t = -2.0  # Conservative default
        
        # Adjust critical value for small samples (rough approximation)
        if n < 30:
            critical_t *= 1.1  # More conservative for small samples
        
        return t_stat < critical_t
    
    def test_node_weakness(self, node, score_type='dove_ranking', node_path="root"):
        """Test if a node and its children are weak"""
        scores = self.get_continuous_scores(node, score_type)
        node_size = len(scores)
        
        # Test if this node is weak
        is_weak = self.perform_weakness_test(scores)
        
        # Store node information
        if not hasattr(self, 'node_info'):
            self.node_info = {}
        
        self.node_info[node_path] = {
            'capability': node.get('capability', 'Unknown'),
            'size': node_size,
            'mean_score': statistics.mean(scores) if scores else 0,
            'std_score': statistics.stdev(scores) if len(scores) > 1 else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'is_weak': is_weak,
            'below_threshold': statistics.mean(scores) < self.tau if scores else False
        }
        
        self.weak_nodes[node_path] = is_weak
        
        # Recursively test children
        if 'subtrees' in node:
            if isinstance(node['subtrees'], list):
                for i, child in enumerate(node['subtrees']):
                    child_path = f"{node_path}.{i}"
                    self.test_node_weakness(child, score_type, child_path)
            elif isinstance(node['subtrees'], dict):
                for i, (key, child) in enumerate(node['subtrees'].items()):
                    child_path = f"{node_path}.{i}"
                    self.test_node_weakness(child, score_type, child_path)
    
    def extract_weak_paths(self, node, node_path="root"):
        """Extract weak path roots following the hierarchical logic"""
        node_size = self.get_node_size(node)
        
        # Check if this node qualifies for extraction
        if node_size >= self.sigma1 and self.weak_nodes.get(node_path, False):
            # Check if all children are also weak
            all_children_weak = True
            has_valid_children = False
            
            if 'subtrees' in node:
                if isinstance(node['subtrees'], list):
                    for i, child in enumerate(node['subtrees']):
                        child_path = f"{node_path}.{i}"
                        child_size = self.get_node_size(child)
                        
                        if child_size >= self.sigma2:
                            has_valid_children = True
                            if not self.weak_nodes.get(child_path, False):
                                all_children_weak = False
                                break
                elif isinstance(node['subtrees'], dict):
                    for i, (key, child) in enumerate(node['subtrees'].items()):
                        child_path = f"{node_path}.{i}"
                        child_size = self.get_node_size(child)
                        
                        if child_size >= self.sigma2:
                            has_valid_children = True
                            if not self.weak_nodes.get(child_path, False):
                                all_children_weak = False
                                break
            
            # If all children are weak (or no valid children), extract this node
            if all_children_weak:
                self.extracted_paths.append({
                    'path': node_path,
                    'capability': node.get('capability', 'Unknown'),
                    'size': node_size,
                    'mean_score': self.node_info[node_path]['mean_score'],
                    'std_score': self.node_info[node_path]['std_score'],
                    'threshold_diff': self.node_info[node_path]['mean_score'] - self.tau,
                    'has_children': has_valid_children,
                    'weakness_type': 'statistical' if self.use_statistical_test else 'simple'
                })
                return  # Don't descend further (avoid overlap)
        
        # Recursively extract from children
        if 'subtrees' in node:
            if isinstance(node['subtrees'], list):
                for i, child in enumerate(node['subtrees']):
                    child_path = f"{node_path}.{i}"
                    self.extract_weak_paths(child, child_path)
            elif isinstance(node['subtrees'], dict):
                for i, (key, child) in enumerate(node['subtrees'].items()):
                    child_path = f"{node_path}.{i}"
                    self.extract_weak_paths(child, child_path)
    
    def find_deeper_starting_nodes(self, tree_data, min_depth=2, score_type='dove_ranking'):
        """Find nodes at a specific depth to start analysis"""
        starting_nodes = []
        
        def traverse_to_depth(node, current_depth=0, path="root"):
            if current_depth >= min_depth:
                node_size = self.get_node_size(node, score_type)
                if node_size >= self.sigma1:
                    starting_nodes.append((node, path))
                return
            
            if 'subtrees' in node:
                if isinstance(node['subtrees'], list):
                    for i, child in enumerate(node['subtrees']):
                        child_path = f"{path}.{i}" if path else str(i)
                        traverse_to_depth(child, current_depth + 1, child_path)
                elif isinstance(node['subtrees'], dict):
                    for i, (key, child) in enumerate(node['subtrees'].items()):
                        child_path = f"{path}.{i}" if path else str(i)
                        traverse_to_depth(child, current_depth + 1, child_path)
        
        traverse_to_depth(tree_data)
        return starting_nodes
    
    def build_capability_path_with_scores(self, tree_data, path_string, score_type='dove_ranking'):
        """Build the full capability path from root to target node with scores for each level"""
        path_parts = path_string.split('.')
        capability_path = []
        path_scores = []
        
        current_node = tree_data
        capability_path.append(current_node.get('capability', 'Root'))
        
        # Calculate score for root node
        root_scores = self.get_continuous_scores(current_node, score_type)
        if root_scores:
            path_scores.append(statistics.mean(root_scores))
        
        # Navigate through the path
        for i, part in enumerate(path_parts[1:], 1):  # Skip 'root'
            if 'subtrees' in current_node:
                try:
                    idx = int(part)
                    if isinstance(current_node['subtrees'], list):
                        if idx < len(current_node['subtrees']):
                            current_node = current_node['subtrees'][idx]
                            capability_path.append(current_node.get('capability', f'Node_{i}'))
                            # Calculate score for this node
                            node_scores = self.get_continuous_scores(current_node, score_type)
                            if node_scores:
                                path_scores.append(statistics.mean(node_scores))
                    elif isinstance(current_node['subtrees'], dict):
                        keys = list(current_node['subtrees'].keys())
                        if idx < len(keys):
                            current_node = current_node['subtrees'][keys[idx]]
                            capability_path.append(current_node.get('capability', f'Node_{i}'))
                            # Calculate score for this node
                            node_scores = self.get_continuous_scores(current_node, score_type)
                            if node_scores:
                                path_scores.append(statistics.mean(node_scores))
                except (ValueError, IndexError, KeyError):
                    capability_path.append(f'Unknown_{part}')
                    break
        
        return capability_path, path_scores
    
    def extract_hierarchical_weaknesses(self, tree_data, score_type='dove_ranking', start_depth=2):
        """Main function to extract hierarchical weakness profiles"""
        # Reset state
        self.weak_nodes = {}
        self.extracted_paths = []
        self.node_info = {}
        
        print(f"Extracting hierarchical weakness profiles using {score_type} scores...")
        print(f"Parameters: σ1={self.sigma1}, σ2={self.sigma2}, α={self.alpha}")
        print(f"Statistical test: {'Enabled' if self.use_statistical_test else 'Disabled (simple threshold)'}")
        
        # Calculate global threshold
        self.calculate_global_threshold(tree_data, score_type)
        
        # Find starting nodes at desired depth
        starting_nodes = self.find_deeper_starting_nodes(tree_data, start_depth, score_type)
        print(f"Starting analysis from depth {start_depth}")
        print(f"Found {len(starting_nodes)} starting nodes with sufficient size")
        
        # Test all nodes for weakness
        print("\nTesting nodes for weakness...")
        for node, path in starting_nodes:
            self.test_node_weakness(node, score_type, path)
        
        # Extract weak path roots
        print("Extracting weak path roots...")
        for node, path in starting_nodes:
            self.extract_weak_paths(node, path)
        
        # Build full capability paths with scores for each weak path
        for weak_path in self.extracted_paths:
            capability_path, path_scores = self.build_capability_path_with_scores(
                tree_data, weak_path['path'], score_type
            )
            weak_path['capability_path'] = capability_path
            weak_path['path_scores'] = path_scores
            
            # Calculate mean weakness severity across the entire path
            if path_scores:
                path_weaknesses = [max(0, self.tau - score) for score in path_scores]  # Only count scores below threshold
                weak_path['path_mean_weakness'] = statistics.mean(path_weaknesses) if path_weaknesses else 0
                weak_path['path_max_weakness'] = max(path_weaknesses) if path_weaknesses else 0
        
        # Sort results by weakness severity (lowest mean score first)
        self.extracted_paths.sort(key=lambda x: x['mean_score'])
        
        return {
            'parameters': {
                'score_type': score_type,
                'global_threshold_tau': self.tau,
                'min_node_size_sigma1': self.sigma1,
                'min_child_size_sigma2': self.sigma2,
                'confidence_level_alpha': self.alpha,
                'use_statistical_test': self.use_statistical_test,
                'starting_depth': start_depth
            },
            'statistics': {
                'starting_nodes_found': len(starting_nodes),
                'total_nodes_tested': len(self.node_info),
                'nodes_marked_weak': sum(1 for info in self.node_info.values() if info['is_weak']),
                'weak_path_roots_extracted': len(self.extracted_paths)
            },
            'weak_paths': self.extracted_paths,
            'detailed_node_stats': self.node_info
        }

def main():
    print("Loading mmlu_new.json...")
    with open('/home/niso/LabWork/AgainAgain/mmlu_new.json', 'r') as f:
        mmlu_data = json.load(f)
    
    # Create extractor for continuous scores
    extractor = ContinuousWeaknessExtractor(
        min_node_size_sigma1=50,    # σ1 = minimum 50 samples for extraction
        min_child_size_sigma2=30,   # σ2 = minimum 30 samples for children
        confidence_level=0.05,      # α = 95% confidence
        use_statistical_test=True   # Use statistical test vs simple threshold
    )
    
    print("\n" + "="*80)
    print("HIERARCHICAL WEAKNESS PROFILE EXTRACTION - COMPARATIVE ANALYSIS")
    print("="*80)
    
    # Extract weaknesses using dove_ranking scores
    print("\n🔍 ANALYZING DOVE_RANKING SCORES...")
    dove_results = extractor.extract_hierarchical_weaknesses(
        mmlu_data, 
        score_type='dove_ranking', 
        start_depth=3
    )
    
    # Extract weaknesses using original ranking scores with less strict parameters
    print("\n🔍 ANALYZING ORIGINAL RANKING SCORES...")
    
    # Create a second extractor with less strict parameters for original ranking
    ranking_extractor = ContinuousWeaknessExtractor(
        min_node_size_sigma1=50,    # σ1 = minimum 50 samples for extraction
        min_child_size_sigma2=30,   # σ2 = minimum 30 samples for children
        confidence_level=0.1,       # α = 90% confidence (less strict)
        use_statistical_test=False  # Use simple threshold comparison
    )
    
    ranking_results = ranking_extractor.extract_hierarchical_weaknesses(
        mmlu_data, 
        score_type='ranking', 
        start_depth=3
    )
    
    # Create comparison results
    comparison_results = {
        'dove_ranking_analysis': dove_results,
        'original_ranking_analysis': ranking_results,
        'comparison_summary': {
            'dove_threshold': dove_results['parameters']['global_threshold_tau'],
            'ranking_threshold': ranking_results['parameters']['global_threshold_tau'],
            'dove_weak_paths': len(dove_results['weak_paths']),
            'ranking_weak_paths': len(ranking_results['weak_paths']),
            'dove_nodes_tested': dove_results['statistics']['total_nodes_tested'],
            'ranking_nodes_tested': ranking_results['statistics']['total_nodes_tested']
        }
    }
    
    # Save comprehensive results
    print(f"\nSaving comprehensive results...")
    with open('/home/niso/LabWork/AgainAgain/weakness_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Print comparative summary
    print("\n" + "="*80)
    print("COMPARATIVE WEAKNESS ANALYSIS SUMMARY")
    print("="*80)
    print(f"DOVE RANKING:")
    print(f"  Global threshold (τ): {dove_results['parameters']['global_threshold_tau']:.4f}")
    print(f"  Weak paths found: {len(dove_results['weak_paths'])}")
    print(f"  Nodes marked weak: {dove_results['statistics']['nodes_marked_weak']}")
    
    print(f"\nORIGINAL RANKING:")
    print(f"  Global threshold (τ): {ranking_results['parameters']['global_threshold_tau']:.4f}")
    print(f"  Weak paths found: {len(ranking_results['weak_paths'])}")
    print(f"  Nodes marked weak: {ranking_results['statistics']['nodes_marked_weak']}")
    
    # Create categorical comparison
    categorical_data = create_categorical_comparison(dove_results, ranking_results)
    
    print(f"\nResults saved to weakness_comparison.json")
    
    # Print individual capability agreement analysis summary
    print(f"\n" + "="*80)
    print("INDIVIDUAL CAPABILITY WEAKNESS AGREEMENT ANALYSIS")
    print("="*80)
    
    agreement_stats = categorical_data['individual_capability_analysis']['summary_stats']
    agreement_matrix = categorical_data['individual_capability_analysis']['agreement_matrix']
    all_capabilities = categorical_data['individual_capability_analysis']['all_capabilities_found']
    
    print(f"📊 AGREEMENT STATISTICS:")
    print(f"  • Total individual capabilities analyzed: {len(all_capabilities)}")
    print(f"  • Overall Agreement Rate: {agreement_stats['agreement_rate']:.1%}")
    print(f"  • Both methods detect weakness: {agreement_stats['both_detect']} capabilities")
    print(f"  • Dove ranking only: {agreement_stats['dove_only']} capabilities")
    print(f"  • Original ranking only: {agreement_stats['ranking_only']} capabilities")
    print(f"  • Neither detects weakness: {agreement_stats['neither_detects']} capabilities")
    
    print(f"\n🔍 INDIVIDUAL CAPABILITY AGREEMENT:")
    for i, (capability, analysis) in enumerate(agreement_matrix.items()):
        dove_status = "✓" if analysis['dove_detects'] else "✗"
        ranking_status = "✓" if analysis['ranking_detects'] else "✗"
        agreement_status = "🟢 AGREE" if analysis['agreement'] else "🔴 DISAGREE"
        
        # Truncate long capability names for readability
        display_capability = capability[:60] + '...' if len(capability) > 60 else capability
        print(f"  {i+1:2d}. {display_capability}")
        print(f"      Dove: {dove_status} | Ranking: {ranking_status} | {agreement_status}")
    
    print(f"\n🎯 METHODOLOGY VERIFICATION:")
    print(f"  • Individual capability comparison using final (most specific) capabilities")
    print(f"  • Granular analysis of weakness profile agreement at leaf-level")
    print(f"  • {len(all_capabilities)} distinct capabilities identified from weak paths")
    print(f"  • No aggregation - shows exact capability-level agreement between methods")
    
    print(f"\nIndividual capability agreement analysis complete!")

def create_categorical_comparison(dove_results, ranking_results):
    """Create categorical comparison based on actual capability tree structure"""
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    
    def extract_individual_capabilities(paths, debug=False):
        """Extract individual final capabilities (leaf nodes) for granular comparison"""
        capabilities = []
        capability_details = []
        
        for path in paths:
            capability_path = path['capability_path']
            
            # Use the final (most specific) capability as the category
            final_capability = capability_path[-1] if capability_path else 'Unknown'
            
            # Clean up for readability while preserving uniqueness
            if len(final_capability) > 80:
                # Truncate but keep meaningful part
                capability = final_capability[:77] + '...'
            else:
                capability = final_capability
            
            capabilities.append(capability)
            
            if debug:
                capability_details.append({
                    'path_id': path['path'],
                    'final_capability': capability,
                    'full_capability_path': capability_path,
                    'path_depth': len(capability_path),
                    'size': path.get('size', 0),
                    'mean_score': path.get('mean_score', 0)
                })
        
        if debug:
            return capabilities, capability_details
        return capabilities
    
    # Extract individual capabilities from both ranking methods with debug info
    dove_capabilities, dove_details = extract_individual_capabilities(dove_results['weak_paths'], debug=True)
    ranking_capabilities, ranking_details = extract_individual_capabilities(ranking_results['weak_paths'], debug=True)
    
    # Print capability details for verification
    print("\n🔍 INDIVIDUAL CAPABILITY WEAKNESS VERIFICATION:")
    print("DOVE RANKING weak capabilities:")
    for detail in dove_details:
        print(f"  Path {detail['path_id']}: '{detail['final_capability'][:70]}...' (size: {detail['size']}, score: {detail['mean_score']:.3f})")
    
    print("\nORIGINAL RANKING weak capabilities:")
    for detail in ranking_details:
        print(f"  Path {detail['path_id']}: '{detail['final_capability'][:70]}...' (size: {detail['size']}, score: {detail['mean_score']:.3f})")
    
    # Get all unique capabilities found in our data
    dove_set = set(dove_capabilities)
    ranking_set = set(ranking_capabilities)
    all_capabilities = sorted(list(dove_set | ranking_set))
    
    # Create agreement matrix
    dove_present = {cap: (cap in dove_set) for cap in all_capabilities}
    ranking_present = {cap: (cap in ranking_set) for cap in all_capabilities}
    
    # Create agreement matrix visualization - adjust figure size for many capabilities
    fig_height = max(10, len(all_capabilities) * 0.5)  # Dynamic height based on number of capabilities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, fig_height))
    
    # Plot 1: Agreement Matrix (dot plot)
    y_pos = np.arange(len(all_capabilities))
    
    # Create matrix data
    dove_values = [1 if dove_present[cap] else 0 for cap in all_capabilities]
    ranking_values = [1 if ranking_present[cap] else 0 for cap in all_capabilities]
    
    # Plot dots for presence/absence
    ax1.scatter([0] * len(all_capabilities), y_pos, 
               c=['#FF6B6B' if val else '#FFE5E5' for val in dove_values],
               s=[400 if val else 100 for val in dove_values], 
               alpha=0.8, label='Dove Ranking')
    
    ax1.scatter([1] * len(all_capabilities), y_pos, 
               c=['#4ECDC4' if val else '#E5F9F6' for val in ranking_values],
               s=[400 if val else 100 for val in ranking_values], 
               alpha=0.8, label='Original Ranking')
    
    # Add agreement indicators
    for i, cap in enumerate(all_capabilities):
        if dove_present[cap] and ranking_present[cap]:
            # Both agree - draw connecting line
            ax1.plot([0, 1], [i, i], 'g-', linewidth=5, alpha=0.8)
            ax1.text(0.5, i + 0.2, '✓', ha='center', va='bottom', fontsize=16, color='green', weight='bold')
        elif dove_present[cap] or ranking_present[cap]:
            # Disagreement - draw dotted line
            ax1.plot([0, 1], [i, i], 'r:', linewidth=4, alpha=0.6)
            ax1.text(0.5, i + 0.2, '✗', ha='center', va='bottom', fontsize=14, color='red')
    
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, len(all_capabilities) - 0.5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Dove\nRanking', 'Original\nRanking'], fontsize=14)
    ax1.set_yticks(y_pos)
    
    # Wrap capability labels for better readability
    wrapped_labels = []
    for cap in all_capabilities:
        if len(cap) > 60:
            # Split at logical points
            words = cap.split()
            if len(words) > 8:
                mid = len(words) // 2
                wrapped = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            else:
                wrapped = cap[:60] + '\n' + cap[60:]
            wrapped_labels.append(wrapped)
        else:
            wrapped_labels.append(cap)
    
    ax1.set_yticklabels(wrapped_labels, fontsize=9)
    ax1.set_title('Individual Capability Weakness Agreement\n(All Specific Capabilities)', fontsize=16, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add legend
    ax1.scatter([], [], c='#FF6B6B', s=400, alpha=0.8, label='Dove: Weakness detected')
    ax1.scatter([], [], c='#FFE5E5', s=100, alpha=0.8, label='Dove: No weakness')
    ax1.scatter([], [], c='#4ECDC4', s=400, alpha=0.8, label='Ranking: Weakness detected')
    ax1.scatter([], [], c='#E5F9F6', s=100, alpha=0.8, label='Ranking: No weakness')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Agreement Statistics
    agreements = sum(1 for cap in all_capabilities if dove_present[cap] == ranking_present[cap])
    total_capabilities = len(all_capabilities)
    agreement_rate = agreements / total_capabilities if total_capabilities > 0 else 0
    
    both_detect = sum(1 for cap in all_capabilities if dove_present[cap] and ranking_present[cap])
    dove_only = sum(1 for cap in all_capabilities if dove_present[cap] and not ranking_present[cap])
    ranking_only = sum(1 for cap in all_capabilities if ranking_present[cap] and not dove_present[cap])
    neither = sum(1 for cap in all_capabilities if not dove_present[cap] and not ranking_present[cap])
    
    agreement_data = {
        'Both Detect\nWeakness': both_detect,
        'Dove Only': dove_only,
        'Ranking Only': ranking_only,
        'Neither\nDetects': neither
    }
    
    # Filter out zero values for cleaner pie chart
    filtered_data = {k: v for k, v in agreement_data.items() if v > 0}
    
    if filtered_data:
        colors = ['#9B59B6', '#FF6B6B', '#4ECDC4', '#BDC3C7'][:len(filtered_data)]
        wedges, texts, autotexts = ax2.pie(filtered_data.values(), labels=filtered_data.keys(),
                                          colors=colors, autopct='%1.0f', startangle=90, textprops={'fontsize': 12})
        ax2.set_title(f'Individual Capability Agreement\n({total_capabilities} capabilities, {agreement_rate:.1%} agreement)', 
                     fontsize=14, weight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Agreement Analysis', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/niso/LabWork/AgainAgain/individual_capability_agreement_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed analysis data
    capability_analysis_data = {
        'methodology': {
            'description': 'Individual capability comparison using final (most specific) capabilities from weak paths',
            'approach': 'Uses the leaf-level capabilities for granular weakness profile agreement analysis',
            'dove_capability_details': dove_details,
            'ranking_capability_details': ranking_details
        },
        'individual_capability_analysis': {
            'all_capabilities_found': all_capabilities,
            'dove_detects': {cap: dove_present[cap] for cap in all_capabilities},
            'ranking_detects': {cap: ranking_present[cap] for cap in all_capabilities},
            'agreement_matrix': {
                cap: {
                    'dove_detects': dove_present[cap],
                    'ranking_detects': ranking_present[cap],
                    'agreement': dove_present[cap] == ranking_present[cap],
                    'both_detect': dove_present[cap] and ranking_present[cap]
                } for cap in all_capabilities
            },
            'summary_stats': {
                'total_capabilities': total_capabilities,
                'agreements': agreements,
                'agreement_rate': agreement_rate,
                'both_detect': both_detect,
                'dove_only': dove_only,
                'ranking_only': ranking_only,
                'neither_detects': neither
            }
        },
        'detailed_weak_paths': {
            'dove_ranking': [
                {
                    'final_capability': cap,
                    'full_capability_path': path['capability_path'],
                    'path_id': path['path'],
                    'size': path['size'],
                    'mean_score': path.get('mean_score', 0)
                }
                for path, cap in zip(dove_results['weak_paths'], dove_capabilities)
            ],
            'original_ranking': [
                {
                    'final_capability': cap,
                    'full_capability_path': path['capability_path'],
                    'path_id': path['path'],
                    'size': path['size'],
                    'mean_score': path.get('mean_score', 0)
                }
                for path, cap in zip(ranking_results['weak_paths'], ranking_capabilities)
            ]
        }
    }
    
    with open('/home/niso/LabWork/AgainAgain/individual_capability_agreement_analysis.json', 'w') as f:
        json.dump(capability_analysis_data, f, indent=2)
    
    print(f"\n📊 Individual capability agreement matrix saved as individual_capability_agreement_matrix.png")
    print(f"📋 Detailed capability analysis saved as individual_capability_agreement_analysis.json")
    
    return capability_analysis_data

if __name__ == "__main__":
    main() 