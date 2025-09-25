"""
Baseline Models for Killer Prediction
======================================

This module implements non-neural baseline models for killer prediction:
1. Frequency-based baselines (speaking frequency, appearance order)
2. Traditional ML baselines (BoW + LogReg, TF-IDF + SVM)

These provide comparison points for the neural embedding approach.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our data models and preprocessing
from data.preprocessing import load_csi_data_complete
from data.models import Episode, Character, Sentence


@dataclass
class BaselineResults:
    """Container for baseline model results."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_scores: Optional[List[float]] = None
    confusion: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None


class BaselineModels:
    """Baseline models for killer prediction using proper data models."""
    
    def __init__(self, data_dir: Path = Path("data/original"), 
                 character_mode: str = 'episode-isolated'):
        """
        Initialize baseline models.
        
        Args:
            data_dir: Path to the directory containing TSV files
            character_mode: 'episode-isolated' or 'cross-episode'
        """
        self.data_dir = data_dir
        self.character_mode = character_mode
        
        # Load data using our preprocessing pipeline
        print(f"Loading data in {character_mode} mode...")
        self.csi_data = load_csi_data_complete(data_dir, character_mode)
        self.episodes = self.csi_data['episodes']
        self.summary_stats = self.csi_data['summary_stats']
        
        # Process episodes for analysis
        self.episode_character_data = {}
        self.character_labels = {}
        self._process_episodes()
        
        print(f"Loaded {len(self.episodes)} episodes with {self.summary_stats['unique_characters']} unique characters")
    
    def _process_episodes(self):
        """Process episodes to extract character data and labels."""
        for episode in self.episodes:
            # Group sentences by character
            char_data = defaultdict(lambda: {
                'sentences': [],
                'word_count': 0,
                'is_killer': False,
                'first_position': float('inf'),
                'last_position': 0
            })
            
            for idx, sentence in enumerate(episode.sentences):
                if sentence.speaker:
                    char_key = sentence.speaker.get_unique_id(self.character_mode)
                    char_data[char_key]['sentences'].append(sentence.text)
                    char_data[char_key]['word_count'] += len(sentence.text.split())
                    char_data[char_key]['first_position'] = min(char_data[char_key]['first_position'], idx)
                    char_data[char_key]['last_position'] = max(char_data[char_key]['last_position'], idx)
                    
                    # Check if killer
                    if sentence.gold_labels and sentence.gold_labels.get('killer_gold') == 'Y':
                        char_data[char_key]['is_killer'] = True
            
            self.episode_character_data[episode.episode_id] = dict(char_data)
            
            # Extract killer labels for this episode
            killers = {char for char, data in char_data.items() if data['is_killer']}
            self.character_labels[episode.episode_id] = killers
    
    def get_episode_characters(self, episode_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get character statistics for an episode.
        
        Returns:
            Dict mapping character IDs to their statistics
        """
        return self.episode_character_data.get(episode_id, {})
    
    def frequency_baseline(self, verbose: bool = True) -> BaselineResults:
        """
        Predict killer based on speaking frequency.
        
        Strategy: The character who speaks the most is predicted as the killer.
        """
        if verbose:
            print("\n" + "="*60)
            print("FREQUENCY-BASED BASELINE")
            print("="*60)
            print("Strategy: Predict character with most sentences as killer")
            print(f"Character mode: {self.character_mode}")
        
        correct = 0
        total = 0
        predictions = []
        actuals = []
        
        for episode in self.episodes:
            char_stats = self.get_episode_characters(episode.episode_id)
            if not char_stats:
                continue
            
            # Find character with most sentences
            most_frequent = max(char_stats.items(), 
                              key=lambda x: len(x[1]['sentences']))[0] if char_stats else None
            
            # Get actual killers
            killers = self.character_labels[episode.episode_id]
            
            # Binary classification: is most frequent character a killer?
            pred = 1 if most_frequent and most_frequent in killers else 0
            actual = 1 if killers else 0  # Episode has killer(s)
            
            predictions.append(pred)
            actuals.append(actual)
            
            if pred == actual and actual == 1:
                correct += 1
            total += 1
            
            if verbose and episode.episode_id in ['s01e07', 's01e08']:  # Sample episodes
                print(f"\n{episode.episode_id}:")
                if most_frequent and most_frequent in char_stats:
                    print(f"  Most frequent: {most_frequent} ({len(char_stats[most_frequent]['sentences'])} sentences)")
                print(f"  Actual killers: {killers if killers else 'None'}")
                print(f"  Correct: {'✓' if pred == actual else '✗'}")
        
        accuracy = correct / total if total > 0 else 0
        precision, recall, f1, _ = precision_recall_fscore_support(
            actuals, predictions, average='binary', zero_division=0
        )
        
        if verbose:
            print(f"\nResults:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1: {f1:.3f}")
        
        return BaselineResults(
            model_name="Frequency Baseline",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1
        )
    
    def appearance_order_baseline(self, verbose: bool = True) -> BaselineResults:
        """
        Predict killer based on appearance order.
        
        Strategy: Characters who appear in the middle of the episode are more likely to be killers.
        """
        if verbose:
            print("\n" + "="*60)
            print("APPEARANCE ORDER BASELINE")
            print("="*60)
            print("Strategy: Predict character appearing in middle third as killer")
            print(f"Character mode: {self.character_mode}")
        
        correct = 0
        total = 0
        predictions = []
        actuals = []
        
        for episode in self.episodes:
            char_stats = self.get_episode_characters(episode.episode_id)
            if not char_stats:
                continue
            
            episode_length = len(episode.sentences)
            middle_start = episode_length // 3
            middle_end = 2 * episode_length // 3
            
            # Find character whose first appearance is closest to middle
            middle_char = None
            min_distance = float('inf')
            
            for char, stats in char_stats.items():
                first_app = stats['first_position']
                if middle_start <= first_app <= middle_end:
                    distance = abs(first_app - episode_length // 2)
                    if distance < min_distance:
                        min_distance = distance
                        middle_char = char
            
            # If no character in middle third, pick one with first appearance closest to middle
            if middle_char is None and char_stats:
                middle_char = min(char_stats.items(),
                                 key=lambda x: abs(x[1]['first_position'] - episode_length // 2))[0]
            
            # Get actual killers
            killers = self.character_labels[episode.episode_id]
            
            # Binary classification
            pred = 1 if middle_char and middle_char in killers else 0
            actual = 1 if killers else 0
            
            predictions.append(pred)
            actuals.append(actual)
            
            if pred == actual and actual == 1:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        precision, recall, f1, _ = precision_recall_fscore_support(
            actuals, predictions, average='binary', zero_division=0
        )
        
        if verbose:
            print(f"\nResults:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1: {f1:.3f}")
        
        return BaselineResults(
            model_name="Appearance Order Baseline",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1
        )
    
    def prepare_ml_data(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Prepare data for traditional ML models.
        
        Returns:
            texts: List of concatenated character dialogue
            labels: Binary labels (1 if killer, 0 if not)
            episode_ids: Episode ID for each sample
        """
        texts = []
        labels = []
        episode_ids = []
        
        for episode in self.episodes:
            char_stats = self.get_episode_characters(episode.episode_id)
            
            for char, stats in char_stats.items():
                if stats['word_count'] < 10:  # Skip characters with very little dialogue
                    continue
                
                all_text = ' '.join(stats['sentences'])
                texts.append(all_text)
                labels.append(1 if stats['is_killer'] else 0)
                episode_ids.append(episode.episode_id)
        
        return texts, labels, episode_ids
    
    def bow_logistic_regression(self, verbose: bool = True) -> BaselineResults:
        """
        Bag-of-Words + Logistic Regression baseline.
        """
        if verbose:
            print("\n" + "="*60)
            print("BAG-OF-WORDS + LOGISTIC REGRESSION")
            print("="*60)
            print("Strategy: BoW features with L2-regularized logistic regression")
            print(f"Character mode: {self.character_mode}")
        
        texts, labels, episode_ids = self.prepare_ml_data()
        
        if len(set(labels)) < 2:
            print("Warning: Not enough class diversity for classification")
            return BaselineResults(
                model_name="BoW + Logistic Regression",
                accuracy=0.0, precision=0.0, recall=0.0, f1=0.0
            )
        
        # Create BoW features
        vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Train with cross-validation
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=min(5, np.sum(y)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Train final model on all data for feature importance
        model.fit(X, y)
        
        # Get predictions for metrics
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary', zero_division=0
        )
        
        # Get top features (if we have killers)
        feature_importance = {}
        if np.sum(y) > 0 and hasattr(model, 'coef_'):
            feature_names = vectorizer.get_feature_names_out()
            coef = model.coef_[0]
            top_killer_idx = np.argsort(coef)[-10:]
            feature_importance = {feature_names[i]: float(coef[i]) for i in top_killer_idx}
        
        if verbose:
            print(f"\nCross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            print(f"\nFinal model performance:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1: {f1:.3f}")
            
            if feature_importance:
                print(f"\nTop killer-indicative words:")
                for word, score in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {word}: {score:.3f}")
        
        return BaselineResults(
            model_name="BoW + Logistic Regression",
            accuracy=cv_scores.mean() if len(cv_scores) > 0 else 0.0,
            precision=precision,
            recall=recall,
            f1=f1,
            cv_scores=cv_scores.tolist() if len(cv_scores) > 0 else None,
            feature_importance=feature_importance
        )
    
    def tfidf_svm(self, verbose: bool = True) -> BaselineResults:
        """
        TF-IDF + SVM baseline.
        """
        if verbose:
            print("\n" + "="*60)
            print("TF-IDF + SVM")
            print("="*60)
            print("Strategy: TF-IDF features with RBF kernel SVM")
            print(f"Character mode: {self.character_mode}")
        
        texts, labels, episode_ids = self.prepare_ml_data()
        
        if len(set(labels)) < 2:
            print("Warning: Not enough class diversity for classification")
            return BaselineResults(
                model_name="TF-IDF + SVM",
                accuracy=0.0, precision=0.0, recall=0.0, f1=0.0
            )
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Train with cross-validation
        model = SVC(kernel='rbf', random_state=42, probability=True)
        cv = StratifiedKFold(n_splits=min(5, np.sum(y)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Train final model
        model.fit(X, y)
        
        # Get predictions for metrics
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary', zero_division=0
        )
        
        if verbose:
            print(f"\nCross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            print(f"\nFinal model performance:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1: {f1:.3f}")
        
        return BaselineResults(
            model_name="TF-IDF + SVM",
            accuracy=cv_scores.mean() if len(cv_scores) > 0 else 0.0,
            precision=precision,
            recall=recall,
            f1=f1,
            cv_scores=cv_scores.tolist() if len(cv_scores) > 0 else None
        )
    
    def ngram_features_baseline(self, verbose: bool = True) -> BaselineResults:
        """
        N-gram features + Logistic Regression baseline.
        """
        if verbose:
            print("\n" + "="*60)
            print("N-GRAM FEATURES + LOGISTIC REGRESSION")
            print("="*60)
            print("Strategy: Character and word n-grams with logistic regression")
            print(f"Character mode: {self.character_mode}")
        
        texts, labels, episode_ids = self.prepare_ml_data()
        
        if len(set(labels)) < 2:
            print("Warning: Not enough class diversity for classification")
            return BaselineResults(
                model_name="N-gram Features",
                accuracy=0.0, precision=0.0, recall=0.0, f1=0.0
            )
        
        # Create n-gram features (both word and character n-grams)
        word_vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=500, stop_words='english')
        char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
        
        X_words = word_vectorizer.fit_transform(texts)
        X_chars = char_vectorizer.fit_transform(texts)
        
        # Combine features
        from scipy.sparse import hstack
        X = hstack([X_words, X_chars])
        y = np.array(labels)
        
        # Train with cross-validation
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=min(5, np.sum(y)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Train final model
        model.fit(X, y)
        
        # Get predictions for metrics
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary', zero_division=0
        )
        
        if verbose:
            print(f"\nCross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            print(f"\nFinal model performance:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1: {f1:.3f}")
        
        return BaselineResults(
            model_name="N-gram Features",
            accuracy=cv_scores.mean() if len(cv_scores) > 0 else 0.0,
            precision=precision,
            recall=recall,
            f1=f1,
            cv_scores=cv_scores.tolist() if len(cv_scores) > 0 else None
        )
    
    def combined_features_baseline(self, verbose: bool = True) -> BaselineResults:
        """
        Combined statistical and text features baseline.
        
        Combines speaking frequency, appearance order, and text features.
        """
        if verbose:
            print("\n" + "="*60)
            print("COMBINED FEATURES BASELINE")
            print("="*60)
            print("Strategy: Statistical + text features with logistic regression")
            print(f"Character mode: {self.character_mode}")
        
        texts, labels, episode_ids = self.prepare_ml_data()
        
        if len(set(labels)) < 2:
            print("Warning: Not enough class diversity for classification")
            return BaselineResults(
                model_name="Combined Features",
                accuracy=0.0, precision=0.0, recall=0.0, f1=0.0
            )
        
        # Create text features
        tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        X_text = tfidf.fit_transform(texts)
        
        # Create statistical features
        stat_features = []
        for i, (text, ep_id) in enumerate(zip(texts, episode_ids)):
            char_stats = self.get_episode_characters(ep_id)
            episode_sentences = len([e for e in self.episodes if e.episode_id == ep_id][0].sentences)
            
            # Find the character this sample represents
            # This is a simplification - in practice we'd need better matching
            sample_features = [
                len(text.split()),  # Word count
                len(text.split()) / max(1, episode_sentences),  # Relative frequency
                0.5,  # Default position (would need proper tracking)
                0.5,  # Default spread
                text.count('?') / max(1, len(text.split())),  # Question rate
                text.count('!') / max(1, len(text.split()))   # Exclamation rate
            ]
            
            stat_features.append(sample_features)
        
        # Combine all features
        from scipy.sparse import hstack
        X_stats = StandardScaler().fit_transform(np.array(stat_features))
        X = hstack([X_text, X_stats])
        y = np.array(labels)
        
        # Train with cross-validation
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=min(5, np.sum(y)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Train final model
        model.fit(X, y)
        
        # Get predictions for metrics
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary', zero_division=0
        )
        
        if verbose:
            print(f"\nFeature types:")
            print(f"  Text features: {X_text.shape[1]}")
            print(f"  Statistical features: {X_stats.shape[1]}")
            print(f"  Total features: {X.shape[1]}")
            print(f"\nCross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
            print(f"\nFinal model performance:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1: {f1:.3f}")
        
        return BaselineResults(
            model_name="Combined Features",
            accuracy=cv_scores.mean() if len(cv_scores) > 0 else 0.0,
            precision=precision,
            recall=recall,
            f1=f1,
            cv_scores=cv_scores.tolist() if len(cv_scores) > 0 else None
        )
    
    def run_all_baselines(self) -> pd.DataFrame:
        """
        Run all baseline models and return results as DataFrame.
        """
        print("\n" + "="*80)
        print(f"RUNNING ALL BASELINE MODELS FOR KILLER PREDICTION ({self.character_mode} mode)")
        print("="*80)
        
        results = []
        
        # Run frequency-based baselines
        results.append(self.frequency_baseline())
        results.append(self.appearance_order_baseline())
        
        # Run traditional ML baselines
        results.append(self.bow_logistic_regression())
        results.append(self.tfidf_svm())
        results.append(self.ngram_features_baseline())
        results.append(self.combined_features_baseline())
        
        # Create results DataFrame
        df_results = pd.DataFrame([
            {
                'Model': r.model_name,
                'Accuracy': r.accuracy,
                'Precision': r.precision,
                'Recall': r.recall,
                'F1': r.f1,
                'CV_Std': np.std(r.cv_scores) if r.cv_scores else None
            }
            for r in results
        ])
        
        print("\n" + "="*80)
        print(f"BASELINE RESULTS SUMMARY ({self.character_mode} mode)")
        print("="*80)
        print(df_results.to_string(index=False))
        
        # Save results
        output_dir = Path(f"experiments/baseline_results_{self.character_mode}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_results.to_csv(output_dir / "baseline_scores.csv", index=False)
        
        # Save detailed results as JSON
        import json
        detailed_results = []
        for r in results:
            detailed = {
                'model_name': r.model_name,
                'accuracy': r.accuracy,
                'precision': r.precision,
                'recall': r.recall,
                'f1': r.f1,
                'cv_scores': r.cv_scores,
                'feature_importance': r.feature_importance
            }
            detailed_results.append(detailed)
        
        with open(output_dir / "baseline_detailed.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
        
        return df_results


def main():
    """Run baseline analysis for both character modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline models for killer prediction")
    parser.add_argument('--data-dir', type=Path, default=Path("data/original"),
                       help='Directory containing TSV files')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['episode-isolated', 'cross-episode', 'both'],
                       help='Character mode for analysis')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'frequency', 'appearance', 'bow', 'tfidf', 'ngram', 'combined'],
                       help='Which baseline model to run')
    
    args = parser.parse_args()
    
    modes = ['episode-isolated', 'cross-episode'] if args.mode == 'both' else [args.mode]
    
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"Running baselines in {mode} mode")
        print(f"{'='*80}")
        
        baselines = BaselineModels(args.data_dir, character_mode=mode)
        
        if args.model == 'all':
            baselines.run_all_baselines()
        elif args.model == 'frequency':
            baselines.frequency_baseline()
        elif args.model == 'appearance':
            baselines.appearance_order_baseline()
        elif args.model == 'bow':
            baselines.bow_logistic_regression()
        elif args.model == 'tfidf':
            baselines.tfidf_svm()
        elif args.model == 'ngram':
            baselines.ngram_features_baseline()
        elif args.model == 'combined':
            baselines.combined_features_baseline()


if __name__ == "__main__":
    main()