"""
Sentence Pattern Analyzer
=========================

Analyzes linguistic patterns in character dialogue:
- Sentence length distributions
- Vocabulary richness metrics
- Speech pattern clustering
- Dialogue complexity measures
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
import string
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import warnings
warnings.filterwarnings('ignore')

# Import our data models and preprocessing
from data.preprocessing import load_csi_data_complete
from data.models import Episode, Character, Sentence

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


@dataclass
class SentenceMetrics:
    """Metrics for a single sentence."""
    text: str
    character: str
    episode: str
    length_words: int
    length_chars: int
    avg_word_length: float
    unique_words: int
    complexity_score: float
    has_question: bool
    has_exclamation: bool
    sentiment_indicator: str  # positive, negative, neutral
    pos_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass 
class CharacterSpeechProfile:
    """Speech pattern profile for a character."""
    character: str
    total_sentences: int
    total_words: int
    unique_words: int
    vocabulary_richness: float  # unique/total ratio
    avg_sentence_length: float
    std_sentence_length: float
    question_rate: float
    exclamation_rate: float
    complexity_score: float
    top_words: List[Tuple[str, int]] = field(default_factory=list)
    pos_pattern: Dict[str, float] = field(default_factory=dict)
    cluster_id: Optional[int] = None


class SentencePatternAnalyzer:
    """Analyze sentence-level patterns in dialogue using data models."""
    
    def __init__(self, data_dir: Path = Path("data/original"),
                 character_mode: str = 'episode-isolated'):
        """Initialize the analyzer.
        
        Args:
            data_dir: Path to the directory containing TSV files
            character_mode: 'episode-isolated' or 'cross-episode'
        """
        self.data_dir = data_dir
        self.character_mode = character_mode
        self.sentence_metrics = []
        self.character_profiles = {}
        self.stop_words = set(stopwords.words('english'))
        self.load_all_episodes()
        
    def load_all_episodes(self):
        """Load and process all episode data using data models."""
        print(f"Loading episode data for sentence pattern analysis ({self.character_mode} mode)...")
        
        # Load data using our preprocessing pipeline
        self.csi_data = load_csi_data_complete(self.data_dir, self.character_mode)
        self.episodes = self.csi_data['episodes']
        self.summary_stats = self.csi_data['summary_stats']
        
        # Process each episode
        for episode in self.episodes:
            self._analyze_episode_sentences(episode)
        
        print(f"Analyzed {len(self.episodes)} episodes")
        print(f"Processed {len(self.sentence_metrics)} sentences")
        
        # Build character profiles
        self._build_character_profiles()
        
    def _analyze_episode_sentences(self, episode: Episode):
        """Analyze sentences in a single episode using data models."""
        for sentence in episode.sentences:
            if not sentence.speaker:
                continue
                
            char_key = sentence.speaker.get_unique_id(self.character_mode)
            
            # Calculate metrics
            metrics = self._calculate_sentence_metrics(
                sentence.text, char_key, episode.episode_id
            )
            self.sentence_metrics.append(metrics)
    
    def _calculate_sentence_metrics(self, text: str, character: str, 
                                   episode: str) -> SentenceMetrics:
        """Calculate linguistic metrics for a sentence."""
        # Basic metrics
        words = word_tokenize(text.lower())
        words_no_punct = [w for w in words if w not in string.punctuation]
        unique_words = set(words_no_punct)
        
        # Length metrics
        length_words = len(words_no_punct)
        length_chars = len(text)
        avg_word_length = np.mean([len(w) for w in words_no_punct]) if words_no_punct else 0
        
        # Complexity (using Flesch reading ease approximation)
        syllables_per_word = avg_word_length / 3  # Rough approximation
        complexity_score = 206.835 - 1.015 * length_words - 84.6 * syllables_per_word
        complexity_score = max(0, min(100, complexity_score))  # Normalize to 0-100
        
        # Punctuation patterns
        has_question = '?' in text
        has_exclamation = '!' in text
        
        # POS tagging
        pos_tags = pos_tag(words)
        pos_distribution = Counter(tag for word, tag in pos_tags)
        
        # Simple sentiment indicator
        positive_words = {'good', 'great', 'love', 'happy', 'yes', 'sure', 'thanks'}
        negative_words = {'bad', 'hate', 'no', 'never', 'kill', 'dead', 'murder'}
        
        word_set = set(words_no_punct)
        pos_count = len(word_set & positive_words)
        neg_count = len(word_set & negative_words)
        
        if pos_count > neg_count:
            sentiment = 'positive'
        elif neg_count > pos_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return SentenceMetrics(
            text=text,
            character=character,
            episode=episode,
            length_words=length_words,
            length_chars=length_chars,
            avg_word_length=avg_word_length,
            unique_words=len(unique_words),
            complexity_score=complexity_score,
            has_question=has_question,
            has_exclamation=has_exclamation,
            sentiment_indicator=sentiment,
            pos_distribution=dict(pos_distribution)
        )
    
    def _build_character_profiles(self):
        """Build speech profiles for each character."""
        character_data = defaultdict(lambda: {
            'sentences': [],
            'all_words': [],
            'unique_words': set(),
            'questions': 0,
            'exclamations': 0,
            'complexity_scores': [],
            'pos_tags': []
        })
        
        # Aggregate data by character
        for metrics in self.sentence_metrics:
            char = metrics.character
            character_data[char]['sentences'].append(metrics)
            
            words = word_tokenize(metrics.text.lower())
            words_no_punct = [w for w in words if w not in string.punctuation]
            character_data[char]['all_words'].extend(words_no_punct)
            character_data[char]['unique_words'].update(words_no_punct)
            
            if metrics.has_question:
                character_data[char]['questions'] += 1
            if metrics.has_exclamation:
                character_data[char]['exclamations'] += 1
                
            character_data[char]['complexity_scores'].append(metrics.complexity_score)
            character_data[char]['pos_tags'].extend(metrics.pos_distribution.items())
        
        # Create profiles
        for char, data in character_data.items():
            if not data['sentences']:
                continue
                
            # Calculate statistics
            sentence_lengths = [m.length_words for m in data['sentences']]
            
            # Word frequency
            word_freq = Counter(data['all_words'])
            # Remove stop words from top words
            top_words = [(w, c) for w, c in word_freq.most_common(50) 
                         if w not in self.stop_words][:10]
            
            # POS pattern
            pos_freq = Counter(dict(data['pos_tags']))
            total_pos = sum(pos_freq.values())
            pos_pattern = {tag: count/total_pos for tag, count in pos_freq.items()} if total_pos > 0 else {}
            
            profile = CharacterSpeechProfile(
                character=char,
                total_sentences=len(data['sentences']),
                total_words=len(data['all_words']),
                unique_words=len(data['unique_words']),
                vocabulary_richness=len(data['unique_words']) / max(1, len(data['all_words'])),
                avg_sentence_length=np.mean(sentence_lengths),
                std_sentence_length=np.std(sentence_lengths),
                question_rate=data['questions'] / max(1, len(data['sentences'])),
                exclamation_rate=data['exclamations'] / max(1, len(data['sentences'])),
                complexity_score=np.mean(data['complexity_scores']),
                top_words=top_words,
                pos_pattern=pos_pattern
            )
            
            self.character_profiles[char] = profile
    
    def cluster_speech_patterns(self, n_clusters: int = 5, 
                               min_sentences: int = 20) -> Dict[str, Any]:
        """Cluster characters based on speech patterns."""
        # Filter to characters with enough data
        eligible_chars = [char for char, profile in self.character_profiles.items()
                         if profile.total_sentences >= min_sentences]
        
        if len(eligible_chars) < n_clusters:
            return {'error': f'Not enough characters ({len(eligible_chars)}) for {n_clusters} clusters'}
        
        # Create feature matrix
        features = []
        char_names = []
        
        for char in eligible_chars:
            profile = self.character_profiles[char]
            feature_vec = [
                profile.avg_sentence_length,
                profile.std_sentence_length,
                profile.vocabulary_richness,
                profile.question_rate,
                profile.exclamation_rate,
                profile.complexity_score
            ]
            features.append(feature_vec)
            char_names.append(char)
        
        features = np.array(features)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Assign cluster IDs to profiles
        for char, cluster_id in zip(char_names, clusters):
            self.character_profiles[char].cluster_id = cluster_id
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_chars = [char_names[i] for i, c in enumerate(clusters) if c == cluster_id]
            cluster_features = features[clusters == cluster_id]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_chars),
                'members': cluster_chars[:10],  # Top 10 members
                'avg_sentence_length': np.mean(cluster_features[:, 0]),
                'avg_vocabulary_richness': np.mean(cluster_features[:, 2]),
                'avg_question_rate': np.mean(cluster_features[:, 3]),
                'avg_complexity': np.mean(cluster_features[:, 5])
            }
        
        return {
            'n_clusters': n_clusters,
            'n_characters': len(char_names),
            'clusters': cluster_analysis,
            'feature_importance': {
                'sentence_length': np.std(features[:, 0]),
                'vocabulary_richness': np.std(features[:, 2]),
                'question_rate': np.std(features[:, 3]),
                'complexity': np.std(features[:, 5])
            }
        }
    
    def analyze_dialogue_complexity(self) -> Dict[str, Any]:
        """Analyze overall dialogue complexity patterns."""
        # Episode-level complexity
        episode_complexity = {}
        for episode in self.episodes:
            episode_sentences = [m for m in self.sentence_metrics if m.episode == episode.episode_id]
            if episode_sentences:
                episode_complexity[episode.episode_id] = {
                    'avg_complexity': np.mean([s.complexity_score for s in episode_sentences]),
                    'avg_length': np.mean([s.length_words for s in episode_sentences]),
                    'unique_vocabulary': len(set(word for s in episode_sentences 
                                                for word in word_tokenize(s.text.lower())
                                                if word not in string.punctuation))
                }
        
        # Character type complexity (if killer labels available)
        killer_complexity = []
        non_killer_complexity = []
        
        for episode in self.episodes:
            for sentence in episode.sentences:
                if not sentence.speaker:
                    continue
                    
                char_key = sentence.speaker.get_unique_id(self.character_mode)
                is_killer = sentence.gold_labels and sentence.gold_labels.get('killer_gold') == 'Y'
                
                if char_key in self.character_profiles:
                    if is_killer:
                        killer_complexity.append(self.character_profiles[char_key].complexity_score)
                    else:
                        non_killer_complexity.append(self.character_profiles[char_key].complexity_score)
        
        analysis = {
            'overall_stats': {
                'total_sentences': len(self.sentence_metrics),
                'avg_sentence_length': np.mean([s.length_words for s in self.sentence_metrics]),
                'avg_complexity': np.mean([s.complexity_score for s in self.sentence_metrics]),
                'question_rate': np.mean([s.has_question for s in self.sentence_metrics]),
                'exclamation_rate': np.mean([s.has_exclamation for s in self.sentence_metrics])
            },
            'episode_variation': {
                'complexity_std': np.std([e['avg_complexity'] for e in episode_complexity.values()]),
                'length_std': np.std([e['avg_length'] for e in episode_complexity.values()]),
                'most_complex_episode': max(episode_complexity.items(), 
                                           key=lambda x: x[1]['avg_complexity'])[0] if episode_complexity else None,
                'simplest_episode': min(episode_complexity.items(), 
                                       key=lambda x: x[1]['avg_complexity'])[0] if episode_complexity else None
            }
        }
        
        if killer_complexity and non_killer_complexity:
            analysis['killer_comparison'] = {
                'killer_avg_complexity': np.mean(killer_complexity),
                'non_killer_avg_complexity': np.mean(non_killer_complexity),
                'complexity_difference': np.mean(killer_complexity) - np.mean(non_killer_complexity),
                't_statistic': stats.ttest_ind(killer_complexity, non_killer_complexity)[0],
                'p_value': stats.ttest_ind(killer_complexity, non_killer_complexity)[1]
            }
        
        return analysis
    
    def find_distinctive_patterns(self, character: str) -> Dict[str, Any]:
        """Find distinctive speech patterns for a specific character."""
        if character not in self.character_profiles:
            char_key = None
            # Try to find by normalized name
            for key in self.character_profiles:
                if key.lower().replace(' ', '') == character.lower().replace(' ', ''):
                    char_key = key
                    break
            
            if not char_key:
                return {'error': f'Character {character} not found'}
            character = char_key
        
        profile = self.character_profiles[character]
        
        # Compare to average
        all_profiles = list(self.character_profiles.values())
        avg_sentence_length = np.mean([p.avg_sentence_length for p in all_profiles])
        avg_vocabulary_richness = np.mean([p.vocabulary_richness for p in all_profiles])
        avg_question_rate = np.mean([p.question_rate for p in all_profiles])
        avg_complexity = np.mean([p.complexity_score for p in all_profiles])
        
        distinctive = {
            'character': character,
            'total_sentences': profile.total_sentences,
            'distinctive_features': {
                'sentence_length_diff': profile.avg_sentence_length - avg_sentence_length,
                'vocabulary_richness_diff': profile.vocabulary_richness - avg_vocabulary_richness,
                'question_rate_diff': profile.question_rate - avg_question_rate,
                'complexity_diff': profile.complexity_score - avg_complexity
            },
            'signature_words': profile.top_words[:5],
            'speech_style': self._classify_speech_style(profile),
            'most_similar_characters': self._find_similar_characters(character, top_n=5)
        }
        
        return distinctive
    
    def _classify_speech_style(self, profile: CharacterSpeechProfile) -> str:
        """Classify a character's speech style."""
        styles = []
        
        if profile.avg_sentence_length > 15:
            styles.append('verbose')
        elif profile.avg_sentence_length < 8:
            styles.append('terse')
        
        if profile.question_rate > 0.3:
            styles.append('inquisitive')
        
        if profile.exclamation_rate > 0.2:
            styles.append('emphatic')
        
        if profile.vocabulary_richness > 0.6:
            styles.append('diverse')
        elif profile.vocabulary_richness < 0.3:
            styles.append('repetitive')
        
        if profile.complexity_score > 60:
            styles.append('simple')
        elif profile.complexity_score < 30:
            styles.append('complex')
        
        return ', '.join(styles) if styles else 'neutral'
    
    def _find_similar_characters(self, character: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Find characters with similar speech patterns."""
        if character not in self.character_profiles:
            return []
        
        target_profile = self.character_profiles[character]
        similarities = []
        
        for char, profile in self.character_profiles.items():
            if char == character or profile.total_sentences < 10:
                continue
            
            # Calculate similarity based on multiple features
            feature_diffs = [
                abs(target_profile.avg_sentence_length - profile.avg_sentence_length) / max(1, target_profile.avg_sentence_length),
                abs(target_profile.vocabulary_richness - profile.vocabulary_richness),
                abs(target_profile.question_rate - profile.question_rate),
                abs(target_profile.complexity_score - profile.complexity_score) / 100
            ]
            
            similarity = 1 - np.mean(feature_diffs)
            similarities.append((char, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def generate_report(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive sentence pattern report."""
        report = {
            'character_mode': self.character_mode,
            'overview': {
                'total_episodes': len(self.episodes),
                'total_sentences': len(self.sentence_metrics),
                'total_characters': len(self.character_profiles),
                'avg_sentence_length': np.mean([s.length_words for s in self.sentence_metrics]),
                'avg_complexity': np.mean([s.complexity_score for s in self.sentence_metrics])
            },
            'dialogue_complexity': self.analyze_dialogue_complexity(),
            'speech_clustering': self.cluster_speech_patterns(),
            'top_speakers': []
        }
        
        # Add top speakers by different metrics
        sorted_by_sentences = sorted(self.character_profiles.items(), 
                                    key=lambda x: x[1].total_sentences, reverse=True)[:10]
        
        for char, profile in sorted_by_sentences:
            report['top_speakers'].append({
                'character': char,
                'sentences': profile.total_sentences,
                'avg_length': profile.avg_sentence_length,
                'vocabulary_richness': profile.vocabulary_richness,
                'speech_style': self._classify_speech_style(profile)
            })
        
        if save_path:
            import json
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                # Convert complex objects to serializable format
                serializable_report = self._make_serializable(report)
                json.dump(serializable_report, f, indent=2)
            print(f"Report saved to {save_path}")
        
        return report
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def main():
    """Run sentence pattern analysis for both character modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze sentence patterns in CSI episodes")
    parser.add_argument('--data-dir', type=Path, default=Path("data/original"),
                       help='Directory containing TSV files')
    parser.add_argument('--output-dir', type=Path, default=Path("analysis/sentence_patterns"),
                       help='Directory to save results')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['episode-isolated', 'cross-episode', 'both'],
                       help='Character mode for analysis')
    parser.add_argument('--character', type=str, help='Analyze specific character')
    
    args = parser.parse_args()
    
    modes = ['episode-isolated', 'cross-episode'] if args.mode == 'both' else [args.mode]
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"Running sentence pattern analysis in {mode} mode")
        print(f"{'='*80}")
        
        # Run analysis
        analyzer = SentencePatternAnalyzer(args.data_dir, character_mode=mode)
    
        # Generate report
        report_path = args.output_dir / f"sentence_patterns_report_{mode}.json"
        report = analyzer.generate_report(save_path=report_path)
        
        # Print summary
        print("\n" + "="*60)
        print(f"SENTENCE PATTERN ANALYSIS SUMMARY ({mode} mode)")
        print("="*60)
    
        print(f"\nOverview:")
        for key, value in report['overview'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nSpeech Clustering:")
        if 'clusters' in report['speech_clustering']:
            for cluster_id, info in report['speech_clustering']['clusters'].items():
                print(f"  {cluster_id}: {info['size']} characters")
        
        print(f"\nTop Speakers:")
        for speaker in report['top_speakers'][:5]:
            print(f"  {speaker['character']}: {speaker['sentences']} sentences, style: {speaker['speech_style']}")
        
        # Character analysis if requested
        if args.character:
            distinctive = analyzer.find_distinctive_patterns(args.character)
            if 'error' not in distinctive:
                print(f"\n{args.character} Analysis:")
                print(f"  Speech style: {distinctive['speech_style']}")
                print(f"  Signature words: {distinctive['signature_words'][:3]}")
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()