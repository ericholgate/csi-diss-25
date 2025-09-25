"""
Episode Structure Analyzer
==========================

Analyzes the structural patterns of CSI episodes:
- Character introduction timing
- Dialogue density over time
- Case structure patterns
- Narrative arc detection
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
from scipy import signal
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our data models and preprocessing
from data.preprocessing import load_csi_data_complete
from data.models import Episode, Character, Sentence


@dataclass
class EpisodeSegment:
    """Represents a segment of an episode."""
    segment_id: int
    start_position: float  # 0-1 normalized
    end_position: float
    character_count: int
    sentence_count: int
    word_count: int
    new_characters: List[str] = field(default_factory=list)
    active_characters: List[str] = field(default_factory=list)
    dialogue_density: float = 0.0
    is_killer_active: bool = False


@dataclass
class EpisodeStructure:
    """Complete structural analysis of an episode."""
    episode_id: str
    total_sentences: int
    total_characters: int
    segments: List[EpisodeSegment] = field(default_factory=list)
    character_introduction_curve: List[int] = field(default_factory=list)
    dialogue_density_curve: List[float] = field(default_factory=list)
    narrative_phases: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    peak_activity_points: List[float] = field(default_factory=list)
    

class EpisodeAnalyzer:
    """Analyze episode structure and narrative patterns using data models."""
    
    def __init__(self, data_dir: Path = Path("data/original"), 
                 n_segments: int = 10,
                 character_mode: str = 'episode-isolated'):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing TSV files
            n_segments: Number of segments to divide each episode into
            character_mode: 'episode-isolated' or 'cross-episode'
        """
        self.data_dir = data_dir
        self.n_segments = n_segments
        self.character_mode = character_mode
        self.episode_structures = {}
        self.load_all_episodes()
        
    def load_all_episodes(self):
        """Load and analyze all episode data using data models."""
        print(f"Loading episodes for structure analysis (segments={self.n_segments}, mode={self.character_mode})...")
        
        # Load data using our preprocessing pipeline
        self.csi_data = load_csi_data_complete(self.data_dir, self.character_mode)
        self.episodes = self.csi_data['episodes']
        self.summary_stats = self.csi_data['summary_stats']
        
        # Analyze structure for each episode
        for episode in self.episodes:
            self.episode_structures[episode.episode_id] = self._analyze_episode_structure(episode)
        
        print(f"Analyzed structure of {len(self.episodes)} episodes")
    
    def _analyze_episode_structure(self, episode: Episode) -> EpisodeStructure:
        """Analyze the structure of a single episode using data models."""
        # Get sentence sequence
        sentences = []
        for idx, sentence in enumerate(episode.sentences):
            if not sentence.speaker:
                continue
                
            char_key = sentence.speaker.get_unique_id(self.character_mode)
            sentences.append({
                'sent_id': idx,
                'position': idx,
                'character': char_key,
                'word_count': len(sentence.text.split()),
                'is_killer': sentence.gold_labels and sentence.gold_labels.get('killer_gold') == 'Y'
            })
        
        total_sentences = len(sentences)
        if total_sentences == 0:
            return EpisodeStructure(episode_id=episode.episode_id, total_sentences=0, total_characters=0)
        
        # Divide into segments
        segment_size = total_sentences / self.n_segments
        segments = []
        
        characters_seen = set()
        character_introduction_curve = []
        dialogue_density_curve = []
        
        for seg_idx in range(self.n_segments):
            start_idx = int(seg_idx * segment_size)
            end_idx = int((seg_idx + 1) * segment_size) if seg_idx < self.n_segments - 1 else total_sentences
            
            segment_sentences = sentences[start_idx:end_idx]
            
            # Analyze segment
            segment_chars = set()
            word_count = 0
            killer_active = False
            
            for sent in segment_sentences:
                segment_chars.add(sent['character'])
                word_count += sent['word_count']
                if sent['is_killer']:
                    killer_active = True
            
            new_chars = segment_chars - characters_seen
            characters_seen.update(segment_chars)
            
            segment = EpisodeSegment(
                segment_id=seg_idx,
                start_position=start_idx / total_sentences,
                end_position=end_idx / total_sentences,
                character_count=len(segment_chars),
                sentence_count=len(segment_sentences),
                word_count=word_count,
                new_characters=list(new_chars),
                active_characters=list(segment_chars),
                dialogue_density=word_count / max(1, len(segment_sentences)),
                is_killer_active=killer_active
            )
            
            segments.append(segment)
            character_introduction_curve.append(len(characters_seen))
            dialogue_density_curve.append(segment.dialogue_density)
        
        # Detect narrative phases
        narrative_phases = self._detect_narrative_phases(segments)
        
        # Find peak activity points
        peak_points = self._find_peak_activity(dialogue_density_curve)
        
        return EpisodeStructure(
            episode_id=episode.episode_id,
            total_sentences=total_sentences,
            total_characters=len(set(s['character'] for s in sentences)),
            segments=segments,
            character_introduction_curve=character_introduction_curve,
            dialogue_density_curve=dialogue_density_curve,
            narrative_phases=narrative_phases,
            peak_activity_points=peak_points
        )
    
    def _detect_narrative_phases(self, segments: List[EpisodeSegment]) -> Dict[str, Tuple[float, float]]:
        """Detect narrative phases based on character and dialogue patterns."""
        phases = {}
        
        # Setup phase: Many new characters introduced
        max_new_chars = max(len(s.new_characters) for s in segments[:3])
        if max_new_chars >= 3:
            phases['setup'] = (0.0, 0.3)
        
        # Investigation phase: Steady dialogue, multiple active characters
        mid_segments = segments[3:7]
        avg_chars = np.mean([s.character_count for s in mid_segments])
        if avg_chars >= 3:
            phases['investigation'] = (0.3, 0.7)
        
        # Climax: Peak dialogue density
        density_values = [s.dialogue_density for s in segments]
        max_density_idx = np.argmax(density_values)
        if max_density_idx >= 6:
            phases['climax'] = (max_density_idx / len(segments), 
                               min(1.0, (max_density_idx + 2) / len(segments)))
        
        # Resolution: Final segments
        phases['resolution'] = (0.8, 1.0)
        
        return phases
    
    def _find_peak_activity(self, density_curve: List[float], 
                           prominence: float = 0.3) -> List[float]:
        """Find peaks in dialogue activity."""
        if len(density_curve) < 3:
            return []
        
        # Find peaks using scipy
        peaks, properties = signal.find_peaks(density_curve, 
                                             prominence=np.std(density_curve) * prominence)
        
        # Convert to normalized positions
        peak_positions = [p / len(density_curve) for p in peaks]
        return peak_positions
    
    def analyze_character_introduction_patterns(self) -> Dict[str, Any]:
        """Analyze how characters are introduced across episodes."""
        introduction_patterns = []
        
        for episode_id, structure in self.episode_structures.items():
            if structure.total_sentences == 0:
                continue
            
            # Analyze introduction timing for each character
            char_first_appearance = {}
            for seg_idx, segment in enumerate(structure.segments):
                for char in segment.new_characters:
                    if char not in char_first_appearance:
                        char_first_appearance[char] = seg_idx / self.n_segments
            
            # Categorize characters by introduction timing
            early_chars = [c for c, pos in char_first_appearance.items() if pos < 0.33]
            middle_chars = [c for c, pos in char_first_appearance.items() if 0.33 <= pos < 0.67]
            late_chars = [c for c, pos in char_first_appearance.items() if pos >= 0.67]
            
            introduction_patterns.append({
                'episode': episode_id,
                'total_characters': structure.total_characters,
                'early_introductions': len(early_chars),
                'middle_introductions': len(middle_chars),
                'late_introductions': len(late_chars),
                'introduction_entropy': entropy([len(early_chars), len(middle_chars), len(late_chars)])
            })
        
        # Aggregate statistics
        df = pd.DataFrame(introduction_patterns)
        
        return {
            'avg_early_introductions': df['early_introductions'].mean(),
            'avg_middle_introductions': df['middle_introductions'].mean(), 
            'avg_late_introductions': df['late_introductions'].mean(),
            'avg_introduction_entropy': df['introduction_entropy'].mean(),
            'most_front_loaded': df.nlargest(3, 'early_introductions')['episode'].tolist(),
            'most_back_loaded': df.nlargest(3, 'late_introductions')['episode'].tolist()
        }
    
    def analyze_dialogue_flow(self) -> Dict[str, Any]:
        """Analyze dialogue flow patterns across episodes."""
        flow_patterns = []
        
        for episode_id, structure in self.episode_structures.items():
            if not structure.dialogue_density_curve:
                continue
            
            density_curve = structure.dialogue_density_curve
            
            # Calculate flow metrics
            flow_patterns.append({
                'episode': episode_id,
                'mean_density': np.mean(density_curve),
                'std_density': np.std(density_curve),
                'max_density': np.max(density_curve),
                'min_density': np.min(density_curve),
                'density_range': np.max(density_curve) - np.min(density_curve),
                'peak_count': len(structure.peak_activity_points),
                'smoothness': 1 / (1 + np.mean(np.abs(np.diff(density_curve)))),  # Inverse of roughness
                'crescendo': density_curve[-1] > density_curve[0]  # Builds to end
            })
        
        df = pd.DataFrame(flow_patterns)
        
        return {
            'avg_dialogue_density': df['mean_density'].mean(),
            'avg_peak_count': df['peak_count'].mean(),
            'avg_smoothness': df['smoothness'].mean(),
            'crescendo_episodes': df[df['crescendo']]['episode'].tolist(),
            'most_dynamic': df.nlargest(3, 'density_range')['episode'].tolist(),
            'most_consistent': df.nsmallest(3, 'std_density')['episode'].tolist()
        }
    
    def analyze_killer_appearance_timing(self) -> Dict[str, Any]:
        """Analyze when killers appear in episodes."""
        killer_timing = []
        
        for episode_id, structure in self.episode_structures.items():
            killer_segments = [seg.segment_id for seg in structure.segments 
                             if seg.is_killer_active]
            
            if killer_segments:
                first_appearance = min(killer_segments) / self.n_segments
                last_appearance = max(killer_segments) / self.n_segments
                active_segments = len(killer_segments) / self.n_segments
                
                killer_timing.append({
                    'episode': episode_id,
                    'first_appearance': first_appearance,
                    'last_appearance': last_appearance,
                    'active_duration': active_segments,
                    'appears_in_setup': any(s < 3 for s in killer_segments),
                    'appears_in_climax': any(s >= 7 for s in killer_segments)
                })
        
        if not killer_timing:
            return {'no_killer_data': True}
        
        df = pd.DataFrame(killer_timing)
        
        return {
            'episodes_with_killers': len(df),
            'avg_first_appearance': df['first_appearance'].mean(),
            'avg_last_appearance': df['last_appearance'].mean(),
            'avg_active_duration': df['active_duration'].mean(),
            'setup_appearance_rate': df['appears_in_setup'].mean(),
            'climax_appearance_rate': df['appears_in_climax'].mean(),
            'early_reveal_episodes': df[df['first_appearance'] < 0.3]['episode'].tolist(),
            'late_reveal_episodes': df[df['first_appearance'] > 0.7]['episode'].tolist()
        }
    
    def get_episode_profile(self, episode_id: str) -> Dict[str, Any]:
        """Get detailed structural profile for a specific episode."""
        if episode_id not in self.episode_structures:
            return {'error': f'Episode {episode_id} not found'}
        
        structure = self.episode_structures[episode_id]
        
        profile = {
            'episode_id': episode_id,
            'total_sentences': structure.total_sentences,
            'total_characters': structure.total_characters,
            'narrative_phases': structure.narrative_phases,
            'peak_activity_points': structure.peak_activity_points,
            'segment_details': []
        }
        
        for segment in structure.segments:
            profile['segment_details'].append({
                'segment': segment.segment_id,
                'position': f"{segment.start_position:.2f}-{segment.end_position:.2f}",
                'characters': segment.character_count,
                'new_characters': len(segment.new_characters),
                'sentences': segment.sentence_count,
                'dialogue_density': segment.dialogue_density,
                'killer_active': segment.is_killer_active
            })
        
        # Add shape classification
        density_curve = structure.dialogue_density_curve
        if density_curve:
            start_third = np.mean(density_curve[:3])
            middle_third = np.mean(density_curve[3:7])
            end_third = np.mean(density_curve[7:])
            
            if end_third > middle_third > start_third:
                shape = 'crescendo'
            elif start_third > middle_third > end_third:
                shape = 'diminuendo'
            elif middle_third > start_third and middle_third > end_third:
                shape = 'arc'
            else:
                shape = 'flat'
            
            profile['narrative_shape'] = shape
        
        return profile
    
    def plot_episode_structure(self, episode_id: str, save_path: Optional[Path] = None):
        """Create visualization of episode structure."""
        if episode_id not in self.episode_structures:
            print(f"Episode {episode_id} not found")
            return None
        
        structure = self.episode_structures[episode_id]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Character introduction curve
        ax = axes[0]
        ax.plot(range(len(structure.character_introduction_curve)), 
               structure.character_introduction_curve, 
               marker='o', linewidth=2, color='steelblue')
        ax.set_xlabel('Episode Segment')
        ax.set_ylabel('Cumulative Characters')
        ax.set_title(f'Character Introduction Curve - {episode_id}')
        ax.grid(True, alpha=0.3)
        
        # Dialogue density
        ax = axes[1]
        ax.bar(range(len(structure.dialogue_density_curve)),
              structure.dialogue_density_curve,
              color='green', alpha=0.7)
        
        # Mark peak points
        for peak in structure.peak_activity_points:
            ax.axvline(x=peak * len(structure.dialogue_density_curve), 
                      color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Episode Segment')
        ax.set_ylabel('Dialogue Density')
        ax.set_title('Dialogue Density Distribution')
        ax.grid(True, alpha=0.3)
        
        # Character activity heatmap
        ax = axes[2]
        
        # Create character presence matrix
        all_chars = set()
        for segment in structure.segments:
            all_chars.update(segment.active_characters)
        
        char_list = sorted(all_chars)[:20]  # Limit to top 20 characters
        presence_matrix = np.zeros((len(char_list), self.n_segments))
        
        for seg_idx, segment in enumerate(structure.segments):
            for char_idx, char in enumerate(char_list):
                if char in segment.active_characters:
                    presence_matrix[char_idx, seg_idx] = 1
        
        im = ax.imshow(presence_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Episode Segment')
        ax.set_ylabel('Character')
        ax.set_yticks(range(len(char_list)))
        ax.set_yticklabels(char_list, fontsize=8)
        ax.set_title('Character Presence Over Time')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'Episode Structure Analysis: {episode_id}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def generate_report(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive episode structure report."""
        report = {
            'character_mode': self.character_mode,
            'overview': {
                'total_episodes': len(self.episodes),
                'segments_per_episode': self.n_segments,
                'avg_sentences_per_episode': np.mean([s.total_sentences for s in self.episode_structures.values()]),
                'avg_characters_per_episode': np.mean([s.total_characters for s in self.episode_structures.values()])
            },
            'character_introduction': self.analyze_character_introduction_patterns(),
            'dialogue_flow': self.analyze_dialogue_flow(),
            'killer_timing': self.analyze_killer_appearance_timing(),
            'episode_classifications': self._classify_all_episodes()
        }
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to {save_path}")
        
        return report
    
    def _classify_all_episodes(self) -> Dict[str, List[str]]:
        """Classify all episodes by structural patterns."""
        classifications = {
            'crescendo': [],
            'diminuendo': [],
            'arc': [],
            'flat': [],
            'front_loaded': [],
            'back_loaded': [],
            'balanced': []
        }
        
        for episode_id, structure in self.episode_structures.items():
            if not structure.dialogue_density_curve:
                continue
            
            # Narrative shape
            density_curve = structure.dialogue_density_curve
            start_third = np.mean(density_curve[:3])
            middle_third = np.mean(density_curve[3:7])
            end_third = np.mean(density_curve[7:])
            
            if end_third > middle_third > start_third:
                classifications['crescendo'].append(episode_id)
            elif start_third > middle_third > end_third:
                classifications['diminuendo'].append(episode_id)
            elif middle_third > start_third and middle_third > end_third:
                classifications['arc'].append(episode_id)
            else:
                classifications['flat'].append(episode_id)
            
            # Character loading
            char_curve = structure.character_introduction_curve
            if char_curve:
                early_chars = char_curve[2] if len(char_curve) > 2 else 0
                late_chars = char_curve[-1] - char_curve[-3] if len(char_curve) > 3 else 0
                
                if early_chars > structure.total_characters * 0.6:
                    classifications['front_loaded'].append(episode_id)
                elif late_chars > structure.total_characters * 0.3:
                    classifications['back_loaded'].append(episode_id)
                else:
                    classifications['balanced'].append(episode_id)
        
        return classifications


def main():
    """Run episode structure analysis for both character modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze episode structure in CSI")
    parser.add_argument('--data-dir', type=Path, default=Path("data/original"),
                       help='Directory containing TSV files')
    parser.add_argument('--output-dir', type=Path, default=Path("analysis/episode_structure"),
                       help='Directory to save results')
    parser.add_argument('--segments', type=int, default=10,
                       help='Number of segments per episode')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['episode-isolated', 'cross-episode', 'both'],
                       help='Character mode for analysis')
    parser.add_argument('--episode', type=str, help='Plot specific episode')
    
    args = parser.parse_args()
    
    modes = ['episode-isolated', 'cross-episode'] if args.mode == 'both' else [args.mode]
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"Running episode structure analysis in {mode} mode")
        print(f"{'='*80}")
        
        # Run analysis
        analyzer = EpisodeAnalyzer(args.data_dir, n_segments=args.segments, character_mode=mode)
    
        # Generate report
        report_path = args.output_dir / f"episode_structure_report_{mode}.json"
        report = analyzer.generate_report(save_path=report_path)
        
        # Print summary
        print("\n" + "="*60)
        print(f"EPISODE STRUCTURE ANALYSIS SUMMARY ({mode} mode)")
        print("="*60)
    
        print(f"\nCharacter Introduction Patterns:")
        intro = report['character_introduction']
        print(f"  Average early introductions: {intro['avg_early_introductions']:.1f}")
        print(f"  Average late introductions: {intro['avg_late_introductions']:.1f}")
        
        print(f"\nDialogue Flow Patterns:")
        flow = report['dialogue_flow']
        print(f"  Average dialogue density: {flow['avg_dialogue_density']:.1f}")
        print(f"  Average peak count: {flow['avg_peak_count']:.1f}")
        print(f"  Crescendo episodes: {len(flow['crescendo_episodes'])}")
        
        print(f"\nEpisode Classifications:")
        for pattern, episodes in report['episode_classifications'].items():
            if episodes:
                print(f"  {pattern}: {len(episodes)} episodes")
        
        # Plot specific episode if requested
        if args.episode and mode == modes[0]:  # Only plot once
            plot_path = args.output_dir / f"episode_{args.episode}_structure_{mode}.png"
            analyzer.plot_episode_structure(args.episode, save_path=plot_path)
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()