"""
Gold Label Statistics Analyzer
===============================

Comprehensive analysis of killer patterns in CSI episodes:
- Killer distribution across episodes
- Speaking patterns of killers vs non-killers
- Statistical tests for behavioral differences
- Temporal patterns of killer reveals
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our data models and preprocessing
from data.preprocessing import load_csi_data_complete
from data.models import Episode, Character, Sentence


@dataclass
class KillerStats:
    """Container for killer statistics."""
    episode_id: str
    total_characters: int
    killer_count: int
    killer_names: List[str]
    killer_speaking_ratio: float  # Killer sentences / total sentences
    killer_word_ratio: float  # Killer words / total words
    killer_first_appearance: float  # Normalized position (0-1)
    killer_last_appearance: float
    killer_spread: float  # How spread out killer's dialogue is
    non_killer_avg_sentences: float
    killer_avg_sentences: float
    

@dataclass
class EpisodeKillerAnalysis:
    """Detailed killer analysis for an episode."""
    episode_id: str
    killers: Dict[str, Dict[str, Any]]  # killer_name -> stats
    non_killers: Dict[str, Dict[str, Any]]  # non_killer_name -> stats
    timeline: List[Dict[str, Any]]  # Temporal sequence of who speaks when
    killer_reveal_point: Optional[float] = None  # When killer identity becomes clear
    red_herrings: List[str] = field(default_factory=list)  # Suspicious non-killers


class KillerStatisticsAnalyzer:
    """Analyze killer patterns across CSI episodes using proper data models."""
    
    def __init__(self, data_dir: Path = Path("data/original"),
                 character_mode: str = 'episode-isolated'):
        """Initialize the analyzer.
        
        Args:
            data_dir: Path to the directory containing TSV files
            character_mode: 'episode-isolated' or 'cross-episode'
        """
        self.data_dir = data_dir
        self.character_mode = character_mode
        self.killer_stats = {}
        self.load_all_episodes()
    
    def load_all_episodes(self):
        """Load and process all episode data using data models."""
        print(f"Loading episode data for killer analysis ({self.character_mode} mode)...")
        
        # Load data using our preprocessing pipeline
        self.csi_data = load_csi_data_complete(self.data_dir, self.character_mode)
        self.episodes = self.csi_data['episodes']
        self.summary_stats = self.csi_data['summary_stats']
        
        # Calculate killer statistics for each episode
        for episode in self.episodes:
            self.killer_stats[episode.episode_id] = self._analyze_episode_killers(episode)
        
        print(f"Analyzed {len(self.episodes)} episodes with {self.summary_stats['unique_characters']} unique characters")
    
    def _analyze_episode_killers(self, episode: Episode) -> KillerStats:
        """Analyze killer patterns in a single episode using data models."""
        # Process sentences to identify killers and their patterns
        killers = set()
        killer_sentences = []
        non_killer_sentences = []
        char_sentence_map = defaultdict(list)
        
        for idx, sentence in enumerate(episode.sentences):
            if not sentence.speaker:
                continue
                
            char_key = sentence.speaker.get_unique_id(self.character_mode)
            char_sentence_map[char_key].append({
                'position': idx,
                'word_count': len(sentence.text.split()),
                'is_killer': sentence.gold_labels and sentence.gold_labels.get('killer_gold') == 'Y'
            })
            
            if sentence.gold_labels and sentence.gold_labels.get('killer_gold') == 'Y':
                killers.add(char_key)
                killer_sentences.append({
                    'character': char_key,
                    'position': idx,
                    'word_count': len(sentence.text.split())
                })
            else:
                non_killer_sentences.append({
                    'character': char_key,
                    'position': idx,
                    'word_count': len(sentence.text.split())
                })
        
        # Calculate statistics
        total_chars = len(char_sentence_map)
        total_sentences = len(episode.sentences)
        
        if not killers:
            # No killer identified in this episode
            return KillerStats(
                episode_id=episode_id,
                total_characters=total_chars,
                killer_count=0,
                killer_names=[],
                killer_speaking_ratio=0,
                killer_word_ratio=0,
                killer_first_appearance=0,
                killer_last_appearance=0,
                killer_spread=0,
                non_killer_avg_sentences=total_sentences / max(1, total_chars),
                killer_avg_sentences=0
            )
        
        # Calculate killer metrics
        killer_word_count = sum(s['word_count'] for s in killer_sentences)
        total_word_count = sum(s['word_count'] for s in sentences)
        
        killer_positions = [s['position'] for s in killer_sentences]
        if killer_positions:
            first_pos = min(killer_positions) / max(1, total_sentences)
            last_pos = max(killer_positions) / max(1, total_sentences)
            spread = last_pos - first_pos
        else:
            first_pos = last_pos = spread = 0
        
        # Character-level statistics
        char_sentence_counts = Counter(s['character'] for s in sentences)
        killer_sentence_count = sum(char_sentence_counts[k] for k in killers)
        non_killer_count = total_chars - len(killers)
        non_killer_sentence_count = total_sentences - killer_sentence_count
        
        return KillerStats(
            episode_id=episode_id,
            total_characters=total_chars,
            killer_count=len(killers),
            killer_names=list(killers),
            killer_speaking_ratio=len(killer_sentences) / max(1, total_sentences),
            killer_word_ratio=killer_word_count / max(1, total_word_count),
            killer_first_appearance=first_pos,
            killer_last_appearance=last_pos,
            killer_spread=spread,
            non_killer_avg_sentences=non_killer_sentence_count / max(1, non_killer_count),
            killer_avg_sentences=killer_sentence_count / max(1, len(killers))
        )
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall statistics across all episodes."""
        stats = {
            'total_episodes': len(self.episodes),
            'episodes_with_killers': sum(1 for s in self.killer_stats.values() if s.killer_count > 0),
            'total_killers': sum(s.killer_count for s in self.killer_stats.values()),
            'avg_killers_per_episode': np.mean([s.killer_count for s in self.killer_stats.values()]),
            'killer_distribution': Counter(s.killer_count for s in self.killer_stats.values()),
            'avg_killer_speaking_ratio': np.mean([s.killer_speaking_ratio for s in self.killer_stats.values() if s.killer_count > 0]),
            'avg_killer_word_ratio': np.mean([s.killer_word_ratio for s in self.killer_stats.values() if s.killer_count > 0]),
            'avg_killer_first_appearance': np.mean([s.killer_first_appearance for s in self.killer_stats.values() if s.killer_count > 0]),
            'avg_killer_spread': np.mean([s.killer_spread for s in self.killer_stats.values() if s.killer_count > 0])
        }
        
        return stats
    
    def analyze_speaking_patterns(self) -> Dict[str, Any]:
        """Analyze speaking pattern differences between killers and non-killers."""
        killer_sentences = []
        non_killer_sentences = []
        killer_words = []
        non_killer_words = []
        
        for episode in self.episodes:
            # Track character statistics
            char_stats = defaultdict(lambda: {'sentences': 0, 'words': 0, 'is_killer': False})
            
            for sentence in episode.sentences:
                if not sentence.speaker:
                    continue
                    
                char_key = sentence.speaker.get_unique_id(self.character_mode)
                char_stats[char_key]['sentences'] += 1
                char_stats[char_key]['words'] += len(sentence.text.split())
                
                if sentence.gold_labels and sentence.gold_labels.get('killer_gold') == 'Y':
                    char_stats[char_key]['is_killer'] = True
            
            # Collect statistics
            for char_key, stats in char_stats.items():
                if stats['is_killer']:
                    killer_sentences.append(stats['sentences'])
                    killer_words.append(stats['words'])
                else:
                    non_killer_sentences.append(stats['sentences'])
                    non_killer_words.append(stats['words'])
        
        # Statistical tests
        sentence_test = mannwhitneyu(killer_sentences, non_killer_sentences, alternative='two-sided')
        word_test = mannwhitneyu(killer_words, non_killer_words, alternative='two-sided')
        
        return {
            'killer_sentences': {
                'mean': np.mean(killer_sentences),
                'median': np.median(killer_sentences),
                'std': np.std(killer_sentences),
                'min': np.min(killer_sentences),
                'max': np.max(killer_sentences)
            },
            'non_killer_sentences': {
                'mean': np.mean(non_killer_sentences),
                'median': np.median(non_killer_sentences),
                'std': np.std(non_killer_sentences),
                'min': np.min(non_killer_sentences),
                'max': np.max(non_killer_sentences)
            },
            'killer_words': {
                'mean': np.mean(killer_words),
                'median': np.median(killer_words),
                'std': np.std(killer_words),
                'min': np.min(killer_words),
                'max': np.max(killer_words)
            },
            'non_killer_words': {
                'mean': np.mean(non_killer_words),
                'median': np.median(non_killer_words),
                'std': np.std(non_killer_words),
                'min': np.min(non_killer_words),
                'max': np.max(non_killer_words)
            },
            'statistical_tests': {
                'sentence_mann_whitney_u': float(sentence_test.statistic),
                'sentence_p_value': float(sentence_test.pvalue),
                'sentence_significant': sentence_test.pvalue < 0.05,
                'word_mann_whitney_u': float(word_test.statistic),
                'word_p_value': float(word_test.pvalue),
                'word_significant': word_test.pvalue < 0.05
            }
        }
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze when killers appear and speak in episodes."""
        appearance_patterns = []
        reveal_patterns = []
        
        for episode in self.episodes:
            stats = self.killer_stats[episode.episode_id]
            if stats.killer_count == 0:
                continue
            
            # Track when each killer first and last appears
            appearance_patterns.append({
                'episode': episode.episode_id,
                'first_appearance': stats.killer_first_appearance,
                'last_appearance': stats.killer_last_appearance,
                'spread': stats.killer_spread
            })
            
            # Analyze potential "reveal" point (last third of killer dialogue)
            killer_positions = []
            for idx, sentence in enumerate(episode.sentences):
                if sentence.gold_labels and sentence.gold_labels.get('killer_gold') == 'Y':
                    killer_positions.append(idx)
            
            if killer_positions:
                total_sents = len(episode.sentences)
                # Consider "reveal" as when killer speaks in last quarter
                last_quarter_pos = 0.75
                reveal_sentences = [p for p in killer_positions if p > total_sents * last_quarter_pos]
                if reveal_sentences:
                    reveal_patterns.append({
                        'episode': episode.episode_id,
                        'reveal_position': min(reveal_sentences) / total_sents,
                        'post_reveal_dialogue': len(reveal_sentences) / len(killer_positions)
                    })
        
        return {
            'appearance_analysis': {
                'avg_first_appearance': np.mean([a['first_appearance'] for a in appearance_patterns]),
                'avg_last_appearance': np.mean([a['last_appearance'] for a in appearance_patterns]),
                'avg_spread': np.mean([a['spread'] for a in appearance_patterns]),
                'early_appearance_rate': sum(1 for a in appearance_patterns if a['first_appearance'] < 0.33) / len(appearance_patterns),
                'middle_appearance_rate': sum(1 for a in appearance_patterns if 0.33 <= a['first_appearance'] < 0.67) / len(appearance_patterns),
                'late_appearance_rate': sum(1 for a in appearance_patterns if a['first_appearance'] >= 0.67) / len(appearance_patterns)
            },
            'reveal_analysis': {
                'episodes_with_reveals': len(reveal_patterns),
                'avg_reveal_position': np.mean([r['reveal_position'] for r in reveal_patterns]) if reveal_patterns else 0,
                'avg_post_reveal_dialogue': np.mean([r['post_reveal_dialogue'] for r in reveal_patterns]) if reveal_patterns else 0
            } if reveal_patterns else {'episodes_with_reveals': 0}
        }
    
    def analyze_killer_archetypes(self) -> Dict[str, Any]:
        """Identify common killer archetypes based on behavior patterns."""
        archetypes = {
            'verbose': [],  # Talks a lot
            'silent': [],   # Talks very little
            'early_bird': [],  # Appears early
            'late_arrival': [],  # Appears late
            'consistent': [],  # Speaks throughout
            'burst': []  # Speaks in concentrated bursts
        }
        
        for episode_id, stats in self.killer_stats.items():
            if stats.killer_count == 0:
                continue
            
            # Classify based on speaking ratio
            if stats.killer_speaking_ratio > 0.3:
                archetypes['verbose'].append(episode_id)
            elif stats.killer_speaking_ratio < 0.1:
                archetypes['silent'].append(episode_id)
            
            # Classify based on appearance
            if stats.killer_first_appearance < 0.33:
                archetypes['early_bird'].append(episode_id)
            elif stats.killer_first_appearance > 0.67:
                archetypes['late_arrival'].append(episode_id)
            
            # Classify based on spread
            if stats.killer_spread > 0.6:
                archetypes['consistent'].append(episode_id)
            elif stats.killer_spread < 0.2:
                archetypes['burst'].append(episode_id)
        
        # Calculate archetype statistics
        archetype_stats = {}
        for archetype, episodes in archetypes.items():
            if episodes:
                archetype_stats[archetype] = {
                    'count': len(episodes),
                    'percentage': len(episodes) / max(1, sum(1 for s in self.killer_stats.values() if s.killer_count > 0)) * 100,
                    'example_episodes': episodes[:3]  # First 3 examples
                }
        
        return archetype_stats
    
    def compare_killers_vs_suspects(self) -> Dict[str, Any]:
        """Compare killers with other suspects (if suspect_gold is available)."""
        results = {
            'suspects_available': False,
            'comparison': None
        }
        
        # Check if suspect data is available
        if not self.episodes:
            return results
            
        sample_sentence = self.episodes[0].sentences[0] if self.episodes[0].sentences else None
        if not sample_sentence or not sample_sentence.gold_labels or 'suspect_gold' not in sample_sentence.gold_labels:
            return results
        
        results['suspects_available'] = True
        
        killer_stats = []
        suspect_stats = []
        innocent_stats = []
        
        for episode in self.episodes:
            char_data = defaultdict(lambda: {'sentences': 0, 'words': 0, 'is_killer': False, 'is_suspect': False})
            
            for sentence in episode.sentences:
                if not sentence.speaker:
                    continue
                    
                char_key = sentence.speaker.get_unique_id(self.character_mode)
                char_data[char_key]['sentences'] += 1
                char_data[char_key]['words'] += len(sentence.text.split())
                
                if sentence.gold_labels:
                    if sentence.gold_labels.get('killer_gold') == 'Y':
                        char_data[char_key]['is_killer'] = True
                    if sentence.gold_labels.get('suspect_gold') == 'Y':
                        char_data[char_key]['is_suspect'] = True
            
            for char_key, data in char_data.items():
                if data['is_killer']:
                    killer_stats.append({'sentences': data['sentences'], 'words': data['words']})
                elif data['is_suspect']:
                    suspect_stats.append({'sentences': data['sentences'], 'words': data['words']})
                else:
                    innocent_stats.append({'sentences': data['sentences'], 'words': data['words']})
        
        if killer_stats and suspect_stats and innocent_stats:
            # Kruskal-Wallis test for three groups
            kruskal_sentences = kruskal(
                [s['sentences'] for s in killer_stats],
                [s['sentences'] for s in suspect_stats],
                [s['sentences'] for s in innocent_stats]
            )
            
            results['comparison'] = {
                'killer_avg_sentences': np.mean([s['sentences'] for s in killer_stats]),
                'suspect_avg_sentences': np.mean([s['sentences'] for s in suspect_stats]),
                'innocent_avg_sentences': np.mean([s['sentences'] for s in innocent_stats]),
                'kruskal_wallis_statistic': float(kruskal_sentences.statistic),
                'kruskal_wallis_p_value': float(kruskal_sentences.pvalue),
                'significant_difference': kruskal_sentences.pvalue < 0.05
            }
        
        return results
    
    def generate_report(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive killer statistics report."""
        report = {
            'overview': self.get_overall_statistics(),
            'speaking_patterns': self.analyze_speaking_patterns(),
            'temporal_patterns': self.analyze_temporal_patterns(),
            'archetypes': self.analyze_killer_archetypes(),
            'killer_vs_suspect': self.compare_killers_vs_suspects()
        }
        
        # Add episode-level details
        report['episode_details'] = [
            {
                'episode_id': stats.episode_id,
                'killer_count': stats.killer_count,
                'killer_names': stats.killer_names,
                'killer_speaking_ratio': stats.killer_speaking_ratio,
                'killer_word_ratio': stats.killer_word_ratio,
                'killer_first_appearance': stats.killer_first_appearance,
                'killer_spread': stats.killer_spread
            }
            for stats in self.killer_stats.values()
        ]
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {save_path}")
        
        return report
    
    def plot_killer_distributions(self, save_path: Optional[Path] = None):
        """Create visualizations of killer distributions."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Killer count distribution
        ax = axes[0, 0]
        killer_counts = [s.killer_count for s in self.killer_stats.values()]
        ax.hist(killer_counts, bins=range(max(killer_counts) + 2), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Killers')
        ax.set_ylabel('Number of Episodes')
        ax.set_title('Distribution of Killer Count per Episode')
        ax.grid(True, alpha=0.3)
        
        # 2. Speaking ratio distribution
        ax = axes[0, 1]
        speaking_ratios = [s.killer_speaking_ratio for s in self.killer_stats.values() if s.killer_count > 0]
        ax.hist(speaking_ratios, bins=20, edgecolor='black', alpha=0.7, color='crimson')
        ax.set_xlabel('Killer Speaking Ratio')
        ax.set_ylabel('Number of Episodes')
        ax.set_title('Distribution of Killer Speaking Ratio')
        ax.grid(True, alpha=0.3)
        
        # 3. First appearance distribution
        ax = axes[0, 2]
        first_appearances = [s.killer_first_appearance for s in self.killer_stats.values() if s.killer_count > 0]
        ax.hist(first_appearances, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Normalized First Appearance Position')
        ax.set_ylabel('Number of Episodes')
        ax.set_title('When Killers First Appear')
        ax.axvline(x=0.33, color='red', linestyle='--', alpha=0.5, label='Early/Middle')
        ax.axvline(x=0.67, color='red', linestyle='--', alpha=0.5, label='Middle/Late')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Spread distribution
        ax = axes[1, 0]
        spreads = [s.killer_spread for s in self.killer_stats.values() if s.killer_count > 0]
        ax.hist(spreads, bins=20, edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel('Dialogue Spread (0=concentrated, 1=throughout)')
        ax.set_ylabel('Number of Episodes')
        ax.set_title('Killer Dialogue Spread')
        ax.grid(True, alpha=0.3)
        
        # 5. Killer vs Non-killer sentences
        ax = axes[1, 1]
        killer_sents = [s.killer_avg_sentences for s in self.killer_stats.values() if s.killer_count > 0]
        non_killer_sents = [s.non_killer_avg_sentences for s in self.killer_stats.values() if s.killer_count > 0]
        
        data_to_plot = [killer_sents, non_killer_sents]
        bp = ax.boxplot(data_to_plot, labels=['Killers', 'Non-Killers'], patch_artist=True)
        bp['boxes'][0].set_facecolor('crimson')
        bp['boxes'][1].set_facecolor('steelblue')
        ax.set_ylabel('Average Sentences per Character')
        ax.set_title('Speaking Frequency: Killers vs Non-Killers')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Archetype distribution
        ax = axes[1, 2]
        archetypes = self.analyze_killer_archetypes()
        if archetypes:
            archetype_names = list(archetypes.keys())
            archetype_counts = [archetypes[a]['count'] for a in archetype_names]
            
            ax.bar(archetype_names, archetype_counts, color='teal', alpha=0.7)
            ax.set_xlabel('Archetype')
            ax.set_ylabel('Number of Episodes')
            ax.set_title('Killer Archetype Distribution')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Killer Statistics Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def get_baseline_prediction_accuracy(self) -> Dict[str, float]:
        """Calculate baseline prediction accuracies for comparison."""
        total_episodes = len(self.episodes)
        episodes_with_killers = sum(1 for s in self.killer_stats.values() if s.killer_count > 0)
        
        # Random baseline: 50% chance
        random_accuracy = 0.5
        
        # Majority class baseline
        if episodes_with_killers > total_episodes / 2:
            majority_class = 1  # Predict "has killer"
            majority_accuracy = episodes_with_killers / total_episodes
        else:
            majority_class = 0  # Predict "no killer"
            majority_accuracy = (total_episodes - episodes_with_killers) / total_episodes
        
        # Always predict most common killer count
        killer_count_dist = Counter(s.killer_count for s in self.killer_stats.values())
        most_common_count = killer_count_dist.most_common(1)[0][0]
        most_common_accuracy = killer_count_dist[most_common_count] / total_episodes
        
        return {
            'random_baseline': random_accuracy,
            'majority_class_baseline': majority_accuracy,
            'majority_class_prediction': majority_class,
            'most_common_count_baseline': most_common_accuracy,
            'most_common_killer_count': most_common_count
        }


def main():
    """Run killer statistics analysis for both character modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze killer statistics in CSI episodes")
    parser.add_argument('--data-dir', type=Path, default=Path("data/original"),
                       help='Directory containing TSV files')
    parser.add_argument('--output-dir', type=Path, default=Path("analysis/killer_stats"),
                       help='Directory to save results')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['episode-isolated', 'cross-episode', 'both'],
                       help='Character mode for analysis')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    modes = ['episode-isolated', 'cross-episode'] if args.mode == 'both' else [args.mode]
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"Running killer statistics analysis in {mode} mode")
        print(f"{'='*80}")
        
        # Run analysis
        analyzer = KillerStatisticsAnalyzer(args.data_dir, character_mode=mode)
    
        # Generate report
        report_path = args.output_dir / f"killer_statistics_report_{mode}.json"
        report = analyzer.generate_report(save_path=report_path)
    
        # Print summary
        print("\n" + "="*60)
        print(f"KILLER STATISTICS SUMMARY ({mode} mode)")
        print("="*60)
    
        overview = report['overview']
        print(f"\nOverview:")
        print(f"  Total episodes: {overview['total_episodes']}")
        print(f"  Episodes with killers: {overview['episodes_with_killers']}")
        print(f"  Average killers per episode: {overview['avg_killers_per_episode']:.2f}")
        print(f"  Average killer speaking ratio: {overview['avg_killer_speaking_ratio']:.3f}")
    
        patterns = report['speaking_patterns']
        print(f"\nSpeaking Patterns:")
        print(f"  Killer avg sentences: {patterns['killer_sentences']['mean']:.1f}")
        print(f"  Non-killer avg sentences: {patterns['non_killer_sentences']['mean']:.1f}")
        print(f"  Statistical significance: p={patterns['statistical_tests']['sentence_p_value']:.4f}")
    
        temporal = report['temporal_patterns']
        print(f"\nTemporal Patterns:")
        print(f"  Average first appearance: {temporal['appearance_analysis']['avg_first_appearance']:.2f}")
        print(f"  Average spread: {temporal['appearance_analysis']['avg_spread']:.2f}")
    
        baselines = analyzer.get_baseline_prediction_accuracy()
        print(f"\nBaseline Accuracies:")
        print(f"  Random: {baselines['random_baseline']:.3f}")
        print(f"  Majority class: {baselines['majority_class_baseline']:.3f}")
    
        # Generate plots if requested
        if args.plot:
            plot_path = args.output_dir / f"killer_distributions_{mode}.png"
            analyzer.plot_killer_distributions(save_path=plot_path)
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()