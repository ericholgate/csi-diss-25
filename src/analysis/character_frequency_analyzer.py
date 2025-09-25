"""
Character Frequency Analyzer
=============================

Comprehensive analysis of character appearances and interactions:
- Character appearance frequency
- Speaking frequency distributions
- Main vs minor character identification
- Character co-occurrence patterns
- Cross-episode character tracking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import networkx as nx
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our data models and preprocessing
from data.preprocessing import load_csi_data_complete
from data.models import Episode, Character, Sentence


@dataclass
class CharacterProfile:
    """Profile for a single character."""
    name: str
    normalized_name: str
    episodes: Set[str] = field(default_factory=set)
    total_sentences: int = 0
    total_words: int = 0
    sentence_by_episode: Dict[str, int] = field(default_factory=dict)
    words_by_episode: Dict[str, int] = field(default_factory=dict)
    is_recurring: bool = False
    is_main: bool = False
    avg_sentences_per_episode: float = 0.0
    episode_count: int = 0
    first_appearance: Optional[str] = None
    last_appearance: Optional[str] = None
    co_appearances: Counter = field(default_factory=Counter)
    
    def update_stats(self):
        """Update derived statistics."""
        self.episode_count = len(self.episodes)
        self.is_recurring = self.episode_count > 1
        self.avg_sentences_per_episode = self.total_sentences / max(1, self.episode_count)
        self.is_main = self.episode_count >= 10 and self.avg_sentences_per_episode >= 20


@dataclass
class EpisodeCharacterStats:
    """Character statistics for a single episode."""
    episode_id: str
    character_count: int
    total_sentences: int
    characters: Dict[str, Dict[str, Any]]  # character -> stats
    co_occurrence_matrix: Optional[np.ndarray] = None
    character_list: List[str] = field(default_factory=list)


class CharacterFrequencyAnalyzer:
    """Analyze character frequency and co-occurrence patterns using data models."""
    
    def __init__(self, data_dir: Path = Path("data/original"),
                 character_mode: str = 'episode-isolated'):
        """Initialize the analyzer.
        
        Args:
            data_dir: Path to the directory containing TSV files
            character_mode: 'episode-isolated' or 'cross-episode'
        """
        self.data_dir = data_dir
        self.character_mode = character_mode
        self.character_profiles = {}
        self.episode_stats = {}
        self.co_occurrence_network = nx.Graph()
        self.load_all_episodes()
    
    def load_all_episodes(self):
        """Load and process all episode data using data models."""
        print(f"Loading episode data for character analysis ({self.character_mode} mode)...")
        
        # Load data using our preprocessing pipeline
        self.csi_data = load_csi_data_complete(self.data_dir, self.character_mode)
        self.episodes = self.csi_data['episodes']
        self.summary_stats = self.csi_data['summary_stats']
        
        # Process each episode
        for episode in self.episodes:
            self._process_episode(episode)
        
        # Update character profiles with cross-episode stats
        self._finalize_character_profiles()
        
        print(f"Analyzed {len(self.episodes)} episodes")
        print(f"Found {len(self.character_profiles)} unique characters")
    
    def _process_episode(self, episode: Episode):
        """Process a single episode's character data using data models."""
        episode_chars = {}
        char_sentences = defaultdict(list)
        
        # Process sentences
        for idx, sentence in enumerate(episode.sentences):
            if not sentence.speaker:
                continue
                
            char_key = sentence.speaker.get_unique_id(self.character_mode)
            char_display = sentence.speaker.normalized_name.title()
            word_count = len(sentence.text.split())
            
            # Track for episode stats
            if char_key not in episode_chars:
                episode_chars[char_key] = {
                    'display_name': char_display,
                    'sentences': 0,
                    'words': 0,
                    'sent_ids': []
                }
            
            episode_chars[char_key]['sentences'] += 1
            episode_chars[char_key]['words'] += word_count
            episode_chars[char_key]['sent_ids'].append(idx)
            char_sentences[char_key].append(idx)
            
            # Update character profile
            if char_key not in self.character_profiles:
                self.character_profiles[char_key] = CharacterProfile(
                    name=char_display,
                    normalized_name=char_key,
                    first_appearance=episode.episode_id
                )
            
            profile = self.character_profiles[char_key]
            profile.episodes.add(episode.episode_id)
            profile.total_sentences += 1
            profile.total_words += word_count
            profile.last_appearance = episode.episode_id
            
            if episode.episode_id not in profile.sentence_by_episode:
                profile.sentence_by_episode[episode.episode_id] = 0
                profile.words_by_episode[episode.episode_id] = 0
            
            profile.sentence_by_episode[episode.episode_id] += 1
            profile.words_by_episode[episode.episode_id] += word_count
        
        # Calculate co-occurrences
        char_list = list(episode_chars.keys())
        for i, char1 in enumerate(char_list):
            for char2 in char_list[i+1:]:
                self.character_profiles[char1].co_appearances[char2] += 1
                self.character_profiles[char2].co_appearances[char1] += 1
                
                # Update network
                if self.co_occurrence_network.has_edge(char1, char2):
                    self.co_occurrence_network[char1][char2]['weight'] += 1
                else:
                    self.co_occurrence_network.add_edge(char1, char2, weight=1)
        
        # Store episode stats
        self.episode_stats[episode.episode_id] = EpisodeCharacterStats(
            episode_id=episode.episode_id,
            character_count=len(episode_chars),
            total_sentences=sum(c['sentences'] for c in episode_chars.values()),
            characters=episode_chars,
            character_list=char_list
        )
    
    def _finalize_character_profiles(self):
        """Finalize character profiles with cross-episode statistics."""
        for profile in self.character_profiles.values():
            profile.update_stats()
    
    def get_character_rankings(self, metric: str = 'total_sentences') -> List[Tuple[str, Any]]:
        """
        Get character rankings by various metrics.
        
        Args:
            metric: 'total_sentences', 'total_words', 'episode_count', 'avg_sentences'
        """
        if metric == 'total_sentences':
            return sorted([(p.name, p.total_sentences) for p in self.character_profiles.values()],
                         key=lambda x: x[1], reverse=True)
        elif metric == 'total_words':
            return sorted([(p.name, p.total_words) for p in self.character_profiles.values()],
                         key=lambda x: x[1], reverse=True)
        elif metric == 'episode_count':
            return sorted([(p.name, p.episode_count) for p in self.character_profiles.values()],
                         key=lambda x: x[1], reverse=True)
        elif metric == 'avg_sentences':
            return sorted([(p.name, p.avg_sentences_per_episode) for p in self.character_profiles.values()],
                         key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def identify_character_types(self) -> Dict[str, List[str]]:
        """Categorize characters into types based on appearance patterns."""
        types = {
            'main': [],      # Appear in many episodes with substantial dialogue
            'recurring': [], # Appear in multiple episodes
            'guest': [],     # Single episode, substantial dialogue
            'minor': []      # Single episode, minimal dialogue
        }
        
        for name, profile in self.character_profiles.items():
            if profile.is_main:
                types['main'].append(profile.name)
            elif profile.is_recurring:
                types['recurring'].append(profile.name)
            elif profile.episode_count == 1 and profile.total_sentences >= 10:
                types['guest'].append(profile.name)
            else:
                types['minor'].append(profile.name)
        
        return types
    
    def analyze_co_occurrences(self, min_weight: int = 2) -> Dict[str, Any]:
        """Analyze character co-occurrence patterns."""
        # Filter network to significant connections
        significant_edges = [(u, v, d) for u, v, d in self.co_occurrence_network.edges(data=True)
                            if d['weight'] >= min_weight]
        
        if not significant_edges:
            return {'error': 'No significant co-occurrences found'}
        
        # Create subgraph with significant edges
        G = nx.Graph()
        G.add_edges_from([(u, v, {'weight': d['weight']}) for u, v, d in significant_edges])
        
        # Calculate network metrics
        metrics = {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G, weight='weight') if len(G) > 0 else 0,
            'connected_components': nx.number_connected_components(G)
        }
        
        # Find central characters
        if len(G) > 0:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
            
            metrics['most_connected'] = sorted(degree_centrality.items(), 
                                              key=lambda x: x[1], reverse=True)[:10]
            metrics['most_between'] = sorted(betweenness_centrality.items(),
                                            key=lambda x: x[1], reverse=True)[:10]
        
        # Find character pairs that frequently appear together
        frequent_pairs = sorted([(u, v, d['weight']) for u, v, d in significant_edges],
                               key=lambda x: x[2], reverse=True)[:20]
        metrics['frequent_pairs'] = frequent_pairs
        
        # Detect communities
        if len(G) > 1:
            communities = list(nx.community.greedy_modularity_communities(G, weight='weight'))
            metrics['communities'] = [
                {
                    'members': [self.character_profiles[node].name for node in comm],
                    'size': len(comm)
                }
                for comm in communities if len(comm) > 1
            ]
        
        return metrics
    
    def get_character_timeline(self, character_name: str) -> Dict[str, Any]:
        """Get detailed timeline for a specific character."""
        # Find character by normalized name
        char_key = None
        for key, profile in self.character_profiles.items():
            if key == character_name.lower().replace(' ', '') or profile.name.lower() == character_name.lower():
                char_key = key
                break
        
        if not char_key:
            return {'error': f'Character {character_name} not found'}
        
        profile = self.character_profiles[char_key]
        
        timeline = {
            'character': profile.name,
            'total_episodes': profile.episode_count,
            'total_sentences': profile.total_sentences,
            'total_words': profile.total_words,
            'first_episode': profile.first_appearance,
            'last_episode': profile.last_appearance,
            'is_main': profile.is_main,
            'is_recurring': profile.is_recurring,
            'episode_details': []
        }
        
        # Add episode-by-episode details
        for episode in sorted(profile.episodes):
            timeline['episode_details'].append({
                'episode': episode,
                'sentences': profile.sentence_by_episode.get(episode, 0),
                'words': profile.words_by_episode.get(episode, 0)
            })
        
        # Add co-appearance info
        top_co_appearances = sorted(profile.co_appearances.items(),
                                   key=lambda x: x[1], reverse=True)[:10]
        timeline['frequent_co_appearances'] = [
            {'character': self.character_profiles[char].name, 'count': count}
            for char, count in top_co_appearances
        ]
        
        return timeline
    
    def calculate_character_similarity(self, method: str = 'co_occurrence') -> np.ndarray:
        """
        Calculate character similarity matrix.
        
        Args:
            method: 'co_occurrence' or 'dialogue_pattern'
        """
        char_list = sorted(self.character_profiles.keys())
        n_chars = len(char_list)
        similarity_matrix = np.zeros((n_chars, n_chars))
        
        if method == 'co_occurrence':
            # Similarity based on co-occurrence patterns
            for i, char1 in enumerate(char_list):
                for j, char2 in enumerate(char_list):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Jaccard similarity of co-appearances
                        co1 = set(self.character_profiles[char1].co_appearances.keys())
                        co2 = set(self.character_profiles[char2].co_appearances.keys())
                        
                        if co1 or co2:
                            intersection = len(co1 & co2)
                            union = len(co1 | co2)
                            similarity_matrix[i, j] = intersection / union if union > 0 else 0
        
        elif method == 'dialogue_pattern':
            # Similarity based on speaking patterns across episodes
            for i, char1 in enumerate(char_list):
                for j, char2 in enumerate(char_list):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Create episode vectors
                        all_episodes = set(self.character_profiles[char1].episodes) | \
                                     set(self.character_profiles[char2].episodes)
                        
                        if all_episodes:
                            vec1 = [self.character_profiles[char1].sentence_by_episode.get(ep, 0)
                                   for ep in sorted(all_episodes)]
                            vec2 = [self.character_profiles[char2].sentence_by_episode.get(ep, 0)
                                   for ep in sorted(all_episodes)]
                            
                            # Cosine similarity
                            if any(vec1) and any(vec2):
                                similarity_matrix[i, j] = 1 - cosine(vec1, vec2)
        
        return similarity_matrix
    
    def plot_character_distributions(self, top_n: int = 20, save_path: Optional[Path] = None):
        """Create comprehensive character distribution visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Top characters by total sentences
        ax = axes[0, 0]
        rankings = self.get_character_rankings('total_sentences')[:top_n]
        names, values = zip(*rankings)
        ax.barh(range(len(names)), values, color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Total Sentences')
        ax.set_title(f'Top {top_n} Characters by Speaking Frequency')
        ax.invert_yaxis()
        
        # 2. Episode appearance distribution
        ax = axes[0, 1]
        episode_counts = [p.episode_count for p in self.character_profiles.values()]
        ax.hist(episode_counts, bins=range(1, max(episode_counts) + 2), 
                edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Number of Episodes')
        ax.set_ylabel('Number of Characters')
        ax.set_title('Character Episode Appearance Distribution')
        ax.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Main threshold')
        ax.legend()
        
        # 3. Character type distribution
        ax = axes[0, 2]
        char_types = self.identify_character_types()
        type_counts = {t: len(chars) for t, chars in char_types.items()}
        ax.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%',
               colors=['gold', 'lightblue', 'lightgreen', 'lightcoral'])
        ax.set_title('Character Type Distribution')
        
        # 4. Speaking intensity (sentences per episode when present)
        ax = axes[1, 0]
        intensities = [p.avg_sentences_per_episode for p in self.character_profiles.values()
                      if p.episode_count > 0]
        ax.hist(intensities, bins=30, edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel('Average Sentences per Episode')
        ax.set_ylabel('Number of Characters')
        ax.set_title('Character Speaking Intensity Distribution')
        ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Main character threshold')
        ax.legend()
        
        # 5. Main characters timeline
        ax = axes[1, 1]
        main_chars = [p for p in self.character_profiles.values() if p.is_main][:10]
        for i, char in enumerate(main_chars):
            episodes = sorted(char.episodes)
            episode_nums = [int(ep.split('e')[1]) for ep in episodes if 'e' in ep]
            if episode_nums:
                ax.scatter(episode_nums, [i] * len(episode_nums), 
                          s=50, alpha=0.6, label=char.name[:15])
        ax.set_xlabel('Episode Number')
        ax.set_ylabel('Character')
        ax.set_title('Main Character Appearance Timeline')
        ax.set_yticks(range(len(main_chars)))
        ax.set_yticklabels([c.name[:15] for c in main_chars], fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 6. Co-occurrence network stats
        ax = axes[1, 2]
        co_occur = self.analyze_co_occurrences()
        if 'most_connected' in co_occur and co_occur['most_connected']:
            chars, centrality = zip(*co_occur['most_connected'][:10])
            char_names = [self.character_profiles[c].name[:15] for c in chars]
            ax.barh(range(len(char_names)), centrality, color='teal')
            ax.set_yticks(range(len(char_names)))
            ax.set_yticklabels(char_names, fontsize=8)
            ax.set_xlabel('Degree Centrality')
            ax.set_title('Most Connected Characters')
            ax.invert_yaxis()
        
        plt.suptitle('Character Frequency Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_co_occurrence_network(self, min_weight: int = 3, save_path: Optional[Path] = None):
        """Visualize character co-occurrence network."""
        # Filter to significant connections
        G = nx.Graph()
        significant_edges = [(u, v, d) for u, v, d in self.co_occurrence_network.edges(data=True)
                            if d['weight'] >= min_weight]
        
        if not significant_edges:
            print("No significant co-occurrences to plot")
            return None
        
        G.add_edges_from([(u, v, {'weight': d['weight']}) for u, v, d in significant_edges])
        
        # Only include characters that are in the filtered graph
        included_chars = set(G.nodes())
        
        # Set node sizes based on character importance
        node_sizes = []
        node_colors = []
        labels = {}
        
        for node in G.nodes():
            profile = self.character_profiles[node]
            # Size based on total sentences
            size = min(3000, 100 + profile.total_sentences * 5)
            node_sizes.append(size)
            
            # Color based on character type
            if profile.is_main:
                node_colors.append('gold')
            elif profile.is_recurring:
                node_colors.append('lightblue')
            else:
                node_colors.append('lightgray')
            
            # Label with display name
            labels[node] = profile.name[:10]  # Truncate long names
        
        # Create layout
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Use spring layout with weight
        pos = nx.spring_layout(G, weight='weight', k=2, iterations=50)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                              alpha=0.7, ax=ax)
        
        # Draw edges with varying thickness
        edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
        max_weight = max(edge_weights)
        edge_widths = [3 * w / max_weight for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='gold', s=200, alpha=0.7, label='Main Characters'),
            plt.scatter([], [], c='lightblue', s=200, alpha=0.7, label='Recurring Characters'),
            plt.scatter([], [], c='lightgray', s=200, alpha=0.7, label='Guest/Minor Characters')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(f'Character Co-occurrence Network (min {min_weight} episodes together)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def generate_report(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive character frequency report."""
        # Overall statistics
        total_chars = len(self.character_profiles)
        char_types = self.identify_character_types()
        
        report = {
            'character_mode': self.character_mode,
            'overview': {
                'total_characters': total_chars,
                'total_episodes': len(self.episodes),
                'main_characters': len(char_types['main']),
                'recurring_characters': len(char_types['recurring']),
                'guest_characters': len(char_types['guest']),
                'minor_characters': len(char_types['minor'])
            },
            'character_types': char_types,
            'top_characters': {
                'by_sentences': self.get_character_rankings('total_sentences')[:20],
                'by_episodes': self.get_character_rankings('episode_count')[:20],
                'by_intensity': self.get_character_rankings('avg_sentences')[:20]
            },
            'co_occurrence_analysis': self.analyze_co_occurrences(),
            'episode_statistics': {
                'avg_characters_per_episode': np.mean([s.character_count for s in self.episode_stats.values()]),
                'min_characters': min(s.character_count for s in self.episode_stats.values()),
                'max_characters': max(s.character_count for s in self.episode_stats.values()),
                'avg_sentences_per_episode': np.mean([s.total_sentences for s in self.episode_stats.values()])
            }
        }
        
        # Add main character profiles
        report['main_character_profiles'] = []
        for char_key, profile in self.character_profiles.items():
            if profile.is_main:
                report['main_character_profiles'].append({
                    'name': profile.name,
                    'episodes': profile.episode_count,
                    'total_sentences': profile.total_sentences,
                    'avg_sentences_per_episode': profile.avg_sentences_per_episode,
                    'first_appearance': profile.first_appearance,
                    'last_appearance': profile.last_appearance
                })
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {save_path}")
        
        return report


def main():
    """Run character frequency analysis for both character modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze character frequencies in CSI episodes")
    parser.add_argument('--data-dir', type=Path, default=Path("data/original"),
                       help='Directory containing TSV files')
    parser.add_argument('--output-dir', type=Path, default=Path("analysis/character_frequency"),
                       help='Directory to save results')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['episode-isolated', 'cross-episode', 'both'],
                       help='Character mode for analysis')
    parser.add_argument('--character', type=str, help='Get timeline for specific character')
    parser.add_argument('--plot', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    modes = ['episode-isolated', 'cross-episode'] if args.mode == 'both' else [args.mode]
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"Running character frequency analysis in {mode} mode")
        print(f"{'='*80}")
        
        # Run analysis
        analyzer = CharacterFrequencyAnalyzer(args.data_dir, character_mode=mode)
    
        # Generate report
        report_path = args.output_dir / f"character_frequency_report_{mode}.json"
        report = analyzer.generate_report(save_path=report_path)
    
        # Print summary
        print("\n" + "="*60)
        print(f"CHARACTER FREQUENCY ANALYSIS SUMMARY ({mode} mode)")
        print("="*60)
    
        overview = report['overview']
        print(f"\nOverview:")
        print(f"  Total unique characters: {overview['total_characters']}")
        print(f"  Main characters: {overview['main_characters']}")
        print(f"  Recurring characters: {overview['recurring_characters']}")
        print(f"  Guest characters: {overview['guest_characters']}")
    
        print(f"\nTop 5 Characters by Sentences:")
        for name, count in report['top_characters']['by_sentences'][:5]:
            print(f"  {name}: {count} sentences")
    
        if 'most_connected' in report['co_occurrence_analysis']:
            print(f"\nMost Connected Characters:")
            for char, centrality in report['co_occurrence_analysis']['most_connected'][:5]:
                char_name = analyzer.character_profiles[char].name
                print(f"  {char_name}: {centrality:.3f}")
    
        # Character timeline if requested
        if args.character:
            timeline = analyzer.get_character_timeline(args.character)
            if 'error' not in timeline:
                print(f"\nTimeline for {timeline['character']}:")
                print(f"  Episodes: {timeline['total_episodes']}")
                print(f"  Total sentences: {timeline['total_sentences']}")
                print(f"  First appearance: {timeline['first_episode']}")
                print(f"  Last appearance: {timeline['last_episode']}")
    
        # Generate plots if requested
        if args.plot:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Distribution plots
            dist_path = args.output_dir / f"character_distributions_{mode}.png"
            analyzer.plot_character_distributions(save_path=dist_path)
            
            # Network plot
            network_path = args.output_dir / f"co_occurrence_network_{mode}.png"
            analyzer.plot_co_occurrence_network(save_path=network_path)
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()