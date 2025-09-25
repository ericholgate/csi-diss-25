"""
Cross-Validation Statistics Module
===================================

Statistical analysis of cross-validation results for understanding
model stability, generalization, and episode-specific performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

from .statistical_significance import StatisticalSignificance, TestResult


@dataclass
class FoldResults:
    """Results from a single CV fold."""
    fold_id: int
    train_scores: Dict[str, float]
    val_scores: Dict[str, float]
    episode_scores: Optional[Dict[str, float]] = None  # episode_id -> score
    predictions: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None
    
    
@dataclass
class CVAnalysisReport:
    """Comprehensive CV analysis report."""
    n_folds: int
    fold_variance: Dict[str, float]  # metric -> variance
    fold_consistency: float
    difficult_episodes: List[Tuple[str, float]]  # (episode_id, avg_score)
    generalization_gap: Dict[str, float]  # metric -> gap
    outlier_folds: List[int]
    stability_score: float
    recommendations: List[str]


class CVStatistics:
    """Statistical analysis of cross-validation results."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize CV statistics analyzer.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        self.stats = StatisticalSignificance(alpha=alpha)
    
    def analyze_fold_variance(self, 
                             fold_results: List[FoldResults],
                             metrics: List[str] = ['f1', 'accuracy']) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance variance across CV folds.
        
        Args:
            fold_results: List of results from each fold
            metrics: Metrics to analyze
        
        Returns:
            Dictionary with variance statistics for each metric
        """
        variance_stats = {}
        
        for metric in metrics:
            # Extract scores for this metric
            train_scores = []
            val_scores = []
            
            for fold in fold_results:
                if metric in fold.train_scores:
                    train_scores.append(fold.train_scores[metric])
                if metric in fold.val_scores:
                    val_scores.append(fold.val_scores[metric])
            
            if not val_scores:
                continue
            
            train_scores = np.array(train_scores)
            val_scores = np.array(val_scores)
            
            # Calculate statistics
            variance_stats[metric] = {
                'val_mean': np.mean(val_scores),
                'val_std': np.std(val_scores),
                'val_variance': np.var(val_scores),
                'val_cv': np.std(val_scores) / np.mean(val_scores) if np.mean(val_scores) > 0 else 0,
                'val_min': np.min(val_scores),
                'val_max': np.max(val_scores),
                'val_range': np.max(val_scores) - np.min(val_scores),
                'val_iqr': np.percentile(val_scores, 75) - np.percentile(val_scores, 25),
            }
            
            if len(train_scores) > 0:
                variance_stats[metric].update({
                    'train_mean': np.mean(train_scores),
                    'train_std': np.std(train_scores),
                    'overfit_gap': np.mean(train_scores) - np.mean(val_scores)
                })
            
            # Test for fold consistency (are folds from same distribution?)
            if len(val_scores) > 2:
                # Levene's test for equal variances
                stat, p_value = stats.levene(*[np.array([s]) for s in val_scores])
                variance_stats[metric]['variance_homogeneity_p'] = p_value
                variance_stats[metric]['consistent_variance'] = p_value > self.alpha
        
        return variance_stats
    
    def test_fold_consistency(self, 
                             fold_results: List[FoldResults],
                             metric: str = 'f1') -> TestResult:
        """
        Test if performance is consistent across folds.
        
        Args:
            fold_results: List of fold results
            metric: Metric to test
        
        Returns:
            Statistical test result
        """
        val_scores = []
        for fold in fold_results:
            if metric in fold.val_scores:
                val_scores.append(fold.val_scores[metric])
        
        if len(val_scores) < 2:
            return TestResult(
                test_name="Fold Consistency Test",
                statistic=0,
                p_value=1.0,
                significant=False,
                confidence_level=1 - self.alpha,
                interpretation="Insufficient folds for consistency test"
            )
        
        # One-way ANOVA to test if all folds have same mean
        if len(val_scores) > 2:
            # Create groups for ANOVA
            groups = [[score] for score in val_scores]
            stat, p_value = stats.f_oneway(*groups)
            
            interpretation = "Folds are consistent" if p_value > self.alpha else "Significant variation across folds"
        else:
            # T-test for 2 folds
            stat, p_value = stats.ttest_ind([val_scores[0]], [val_scores[1]])
            interpretation = "Two folds are consistent" if p_value > self.alpha else "Significant difference between folds"
        
        return TestResult(
            test_name="Fold Consistency Test",
            statistic=stat,
            p_value=p_value,
            significant=(p_value < self.alpha),
            confidence_level=1 - self.alpha,
            interpretation=interpretation
        )
    
    def identify_outlier_folds(self, 
                              fold_results: List[FoldResults],
                              metric: str = 'f1',
                              method: str = 'iqr') -> List[int]:
        """
        Identify folds with outlier performance.
        
        Args:
            fold_results: List of fold results
            metric: Metric to check for outliers
            method: 'iqr' (interquartile range) or 'zscore'
        
        Returns:
            List of outlier fold indices
        """
        scores = []
        fold_ids = []
        
        for fold in fold_results:
            if metric in fold.val_scores:
                scores.append(fold.val_scores[metric])
                fold_ids.append(fold.fold_id)
        
        if len(scores) < 3:
            return []  # Need at least 3 folds to detect outliers
        
        scores = np.array(scores)
        outlier_indices = []
        
        if method == 'iqr':
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, score in enumerate(scores):
                if score < lower_bound or score > upper_bound:
                    outlier_indices.append(fold_ids[i])
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(scores))
            threshold = 2.5  # Scores more than 2.5 std from mean
            
            for i, z in enumerate(z_scores):
                if z > threshold:
                    outlier_indices.append(fold_ids[i])
        
        return outlier_indices
    
    def calculate_generalization_gap(self, 
                                    fold_results: List[FoldResults],
                                    metrics: List[str] = ['f1', 'accuracy']) -> Dict[str, float]:
        """
        Calculate the generalization gap (train - val performance).
        
        Args:
            fold_results: List of fold results
            metrics: Metrics to calculate gap for
        
        Returns:
            Dictionary of metric -> generalization gap
        """
        gaps = {}
        
        for metric in metrics:
            train_scores = []
            val_scores = []
            
            for fold in fold_results:
                if metric in fold.train_scores and metric in fold.val_scores:
                    train_scores.append(fold.train_scores[metric])
                    val_scores.append(fold.val_scores[metric])
            
            if train_scores and val_scores:
                gaps[metric] = np.mean(train_scores) - np.mean(val_scores)
        
        return gaps
    
    def episode_difficulty_ranking(self, 
                                  fold_results: List[FoldResults]) -> List[Tuple[str, float, float]]:
        """
        Rank episodes by prediction difficulty.
        
        Args:
            fold_results: List of fold results with episode scores
        
        Returns:
            List of (episode_id, mean_score, std_score) sorted by difficulty
        """
        # Aggregate scores by episode across folds
        episode_scores = {}
        
        for fold in fold_results:
            if fold.episode_scores:
                for episode_id, score in fold.episode_scores.items():
                    if episode_id not in episode_scores:
                        episode_scores[episode_id] = []
                    episode_scores[episode_id].append(score)
        
        # Calculate statistics per episode
        episode_stats = []
        for episode_id, scores in episode_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores) if len(scores) > 1 else 0
            episode_stats.append((episode_id, mean_score, std_score))
        
        # Sort by mean score (lower = more difficult)
        episode_stats.sort(key=lambda x: x[1])
        
        return episode_stats
    
    def test_episode_effects(self, 
                           fold_results: List[FoldResults]) -> TestResult:
        """
        Test if episode identity significantly affects performance.
        
        Args:
            fold_results: List of fold results with episode scores
        
        Returns:
            Statistical test result
        """
        # Collect scores grouped by episode
        episode_groups = {}
        
        for fold in fold_results:
            if fold.episode_scores:
                for episode_id, score in fold.episode_scores.items():
                    if episode_id not in episode_groups:
                        episode_groups[episode_id] = []
                    episode_groups[episode_id].append(score)
        
        if len(episode_groups) < 2:
            return TestResult(
                test_name="Episode Effects Test",
                statistic=0,
                p_value=1.0,
                significant=False,
                confidence_level=1 - self.alpha,
                interpretation="Insufficient episodes for effects test"
            )
        
        # Prepare data for Kruskal-Wallis test (non-parametric ANOVA)
        groups = [scores for scores in episode_groups.values() if len(scores) > 0]
        
        if len(groups) < 2:
            return TestResult(
                test_name="Episode Effects Test",
                statistic=0,
                p_value=1.0,
                significant=False,
                confidence_level=1 - self.alpha,
                interpretation="Insufficient data for episode effects test"
            )
        
        # Kruskal-Wallis H-test
        stat, p_value = stats.kruskal(*groups)
        
        # Calculate effect size (eta-squared)
        all_scores = np.concatenate(groups)
        n = len(all_scores)
        k = len(groups)
        eta_squared = (stat - k + 1) / (n - k) if n > k else 0
        
        interpretation = "Significant episode effects" if p_value < self.alpha else "No significant episode effects"
        interpretation += f" (η² = {eta_squared:.3f})"
        
        return TestResult(
            test_name="Episode Effects Test (Kruskal-Wallis)",
            statistic=stat,
            p_value=p_value,
            significant=(p_value < self.alpha),
            confidence_level=1 - self.alpha,
            effect_size=eta_squared,
            interpretation=interpretation
        )
    
    def analyze_embedding_stability(self, 
                                   fold_results: List[FoldResults],
                                   character_id: Optional[str] = None) -> Dict[str, float]:
        """
        Analyze stability of character embeddings across folds.
        
        Args:
            fold_results: List of fold results with embeddings
            character_id: Specific character to analyze (or None for all)
        
        Returns:
            Stability metrics
        """
        if not any(fold.embeddings is not None for fold in fold_results):
            return {"error": "No embeddings found in fold results"}
        
        # Extract embeddings for analysis
        embeddings_by_fold = []
        for fold in fold_results:
            if fold.embeddings is not None:
                embeddings_by_fold.append(fold.embeddings)
        
        if len(embeddings_by_fold) < 2:
            return {"error": "Need at least 2 folds with embeddings"}
        
        # Calculate pairwise cosine similarities between fold embeddings
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = []
        for i in range(len(embeddings_by_fold)):
            for j in range(i + 1, len(embeddings_by_fold)):
                # Match embeddings (assuming same order)
                sim_matrix = cosine_similarity(embeddings_by_fold[i], embeddings_by_fold[j])
                # Take diagonal (self-similarity)
                diagonal_sim = np.diag(sim_matrix)
                similarities.extend(diagonal_sim)
        
        similarities = np.array(similarities)
        
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'stability_score': np.mean(similarities) * (1 - np.std(similarities))  # Combined metric
        }
    
    def generate_cv_report(self, 
                          fold_results: List[FoldResults],
                          metrics: List[str] = ['f1', 'accuracy']) -> CVAnalysisReport:
        """
        Generate comprehensive CV analysis report.
        
        Args:
            fold_results: List of fold results
            metrics: Metrics to analyze
        
        Returns:
            Complete CV analysis report
        """
        # Analyze fold variance
        variance_stats = self.analyze_fold_variance(fold_results, metrics)
        
        # Extract key variance metrics
        fold_variance = {}
        for metric in metrics:
            if metric in variance_stats:
                fold_variance[metric] = variance_stats[metric]['val_variance']
        
        # Test fold consistency
        consistency_test = self.test_fold_consistency(fold_results)
        fold_consistency = 1.0 - consistency_test.p_value  # Higher = more consistent
        
        # Identify difficult episodes
        episode_ranking = self.episode_difficulty_ranking(fold_results)
        difficult_episodes = episode_ranking[:5] if len(episode_ranking) > 5 else episode_ranking
        
        # Calculate generalization gap
        gen_gap = self.calculate_generalization_gap(fold_results, metrics)
        
        # Identify outlier folds
        outlier_folds = self.identify_outlier_folds(fold_results)
        
        # Calculate overall stability score
        stability_scores = []
        for metric in metrics:
            if metric in variance_stats:
                cv = variance_stats[metric].get('val_cv', 0)
                stability_scores.append(1.0 - cv)  # Lower CV = higher stability
        
        stability_score = np.mean(stability_scores) if stability_scores else 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            variance_stats, consistency_test, gen_gap, outlier_folds, stability_score
        )
        
        return CVAnalysisReport(
            n_folds=len(fold_results),
            fold_variance=fold_variance,
            fold_consistency=fold_consistency,
            difficult_episodes=[(ep[0], ep[1]) for ep in difficult_episodes],
            generalization_gap=gen_gap,
            outlier_folds=outlier_folds,
            stability_score=stability_score,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self,
                                variance_stats: Dict,
                                consistency_test: TestResult,
                                gen_gap: Dict,
                                outlier_folds: List[int],
                                stability_score: float) -> List[str]:
        """Generate actionable recommendations based on CV analysis."""
        recommendations = []
        
        # Check variance
        if variance_stats:
            avg_cv = np.mean([stats.get('val_cv', 0) for stats in variance_stats.values()])
            if avg_cv > 0.15:
                recommendations.append(
                    f"High fold variance (CV={avg_cv:.2f}) - consider increasing data or regularization"
                )
            elif avg_cv < 0.05:
                recommendations.append(
                    f"Low fold variance (CV={avg_cv:.2f}) - model shows stable performance"
                )
        
        # Check consistency
        if not consistency_test.significant:
            recommendations.append("Folds are statistically consistent - good CV setup")
        else:
            recommendations.append("Significant fold variation detected - check for data leakage or imbalance")
        
        # Check generalization gap
        if gen_gap:
            avg_gap = np.mean(list(gen_gap.values()))
            if avg_gap > 0.15:
                recommendations.append(
                    f"Large generalization gap ({avg_gap:.2f}) - model may be overfitting"
                )
            elif avg_gap < 0.05:
                recommendations.append(
                    f"Small generalization gap ({avg_gap:.2f}) - good generalization"
                )
        
        # Check outliers
        if outlier_folds:
            recommendations.append(
                f"Outlier folds detected: {outlier_folds} - investigate these folds"
            )
        
        # Overall stability
        if stability_score > 0.9:
            recommendations.append("Excellent model stability across folds")
        elif stability_score < 0.7:
            recommendations.append("Poor model stability - consider ensemble methods")
        
        return recommendations
    
    def plot_cv_analysis(self, 
                        fold_results: List[FoldResults],
                        metrics: List[str] = ['f1', 'accuracy'],
                        save_path: Optional[Path] = None):
        """
        Create visualization of CV analysis.
        
        Args:
            fold_results: List of fold results
            metrics: Metrics to plot
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Fold performance comparison
        ax = axes[0, 0]
        for metric in metrics:
            scores = [fold.val_scores.get(metric, 0) for fold in fold_results]
            fold_ids = [fold.fold_id for fold in fold_results]
            ax.plot(fold_ids, scores, marker='o', label=metric)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_title('Performance Across Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Train vs Validation
        ax = axes[0, 1]
        for metric in metrics[:1]:  # Just first metric for clarity
            train_scores = [fold.train_scores.get(metric, 0) for fold in fold_results]
            val_scores = [fold.val_scores.get(metric, 0) for fold in fold_results]
            x = np.arange(len(fold_results))
            width = 0.35
            ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.7)
            ax.bar(x + width/2, val_scores, width, label='Val', alpha=0.7)
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(f'{metrics[0]} Score')
        ax.set_title('Train vs Validation Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Episode difficulty (if available)
        ax = axes[1, 0]
        episode_stats = self.episode_difficulty_ranking(fold_results)
        if episode_stats:
            episodes = [e[0] for e in episode_stats[:10]]  # Top 10
            scores = [e[1] for e in episode_stats[:10]]
            ax.barh(episodes, scores, color='steelblue')
            ax.set_xlabel('Average Score')
            ax.set_title('Most Difficult Episodes')
        else:
            ax.text(0.5, 0.5, 'No episode data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 4. Variance analysis
        ax = axes[1, 1]
        variance_stats = self.analyze_fold_variance(fold_results, metrics)
        if variance_stats:
            metric_names = []
            cvs = []
            for metric, stats in variance_stats.items():
                metric_names.append(metric)
                cvs.append(stats.get('val_cv', 0))
            
            ax.bar(metric_names, cvs, color='coral')
            ax.set_ylabel('Coefficient of Variation')
            ax.set_title('Metric Stability (Lower is Better)')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Cross-Validation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


def demo_cv_statistics():
    """Demonstrate CV statistics functionality."""
    print("Cross-Validation Statistics Demo")
    print("=" * 60)
    
    # Create synthetic fold results
    np.random.seed(42)
    fold_results = []
    
    for i in range(5):  # 5 folds
        # Simulate some variance across folds
        base_f1 = 0.75 + np.random.normal(0, 0.03)
        base_acc = 0.80 + np.random.normal(0, 0.02)
        
        # Episode scores (some episodes harder than others)
        episode_scores = {}
        for ep in ['s01e01', 's01e02', 's01e03', 's02e01', 's02e02']:
            if ep.startswith('s01'):
                episode_scores[ep] = base_f1 + np.random.normal(0, 0.02)
            else:
                episode_scores[ep] = base_f1 - 0.1 + np.random.normal(0, 0.03)  # Season 2 harder
        
        fold = FoldResults(
            fold_id=i,
            train_scores={'f1': base_f1 + 0.05, 'accuracy': base_acc + 0.04},
            val_scores={'f1': base_f1, 'accuracy': base_acc},
            episode_scores=episode_scores
        )
        fold_results.append(fold)
    
    # Initialize analyzer
    cv_stats = CVStatistics(alpha=0.05)
    
    # 1. Variance analysis
    print("\n1. Fold Variance Analysis")
    variance = cv_stats.analyze_fold_variance(fold_results)
    for metric, stats in variance.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['val_mean']:.3f} ± {stats['val_std']:.3f}")
        print(f"  CV: {stats['val_cv']:.3f}")
        print(f"  Range: [{stats['val_min']:.3f}, {stats['val_max']:.3f}]")
    
    # 2. Consistency test
    print("\n2. Fold Consistency Test")
    consistency = cv_stats.test_fold_consistency(fold_results)
    print(consistency)
    
    # 3. Episode effects
    print("\n3. Episode Effects Test")
    episode_test = cv_stats.test_episode_effects(fold_results)
    print(episode_test)
    
    # 4. Difficult episodes
    print("\n4. Episode Difficulty Ranking")
    difficult = cv_stats.episode_difficulty_ranking(fold_results)
    for ep, score, std in difficult[:3]:
        print(f"  {ep}: {score:.3f} ± {std:.3f}")
    
    # 5. Generate report
    print("\n5. Comprehensive CV Report")
    report = cv_stats.generate_cv_report(fold_results)
    print(f"  Folds: {report.n_folds}")
    print(f"  Stability Score: {report.stability_score:.3f}")
    print(f"  Generalization Gap: {report.generalization_gap}")
    print(f"  Outlier Folds: {report.outlier_folds}")
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  • {rec}")
    
    # 6. Visualization
    print("\n6. Creating CV analysis plots...")
    cv_stats.plot_cv_analysis(fold_results)
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo_cv_statistics()