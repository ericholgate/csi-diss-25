"""
Model Comparison Module
=======================

Specialized statistical comparisons between different model types,
character modes, and CV paradigms.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import json

from .statistical_significance import StatisticalSignificance, TestResult


@dataclass 
class ModelResults:
    """Container for model results."""
    model_name: str
    scores: Dict[str, List[float]]  # metric_name -> scores
    predictions: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


@dataclass
class ComparisonReport:
    """Comprehensive comparison report."""
    comparison_type: str
    models: List[str]
    statistical_tests: List[TestResult]
    summary_table: pd.DataFrame
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'comparison_type': self.comparison_type,
            'models': self.models,
            'statistical_tests': [
                {
                    'test_name': t.test_name,
                    'p_value': t.p_value,
                    'significant': t.significant,
                    'effect_size': t.effect_size,
                    'interpretation': t.interpretation
                }
                for t in self.statistical_tests
            ],
            'summary_table': self.summary_table.to_dict(),
            'recommendations': self.recommendations
        }
    
    def print_summary(self):
        """Print formatted summary."""
        print(f"\n{'='*60}")
        print(f"Model Comparison: {self.comparison_type}")
        print(f"{'='*60}")
        print(f"Models compared: {', '.join(self.models)}")
        
        print("\nStatistical Tests:")
        for test in self.statistical_tests:
            print(f"\n{test}")
        
        print("\nSummary Table:")
        print(self.summary_table.to_string())
        
        if self.recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"  {i}. {rec}")


class ModelComparison:
    """Compare neural models, baselines, and different configurations."""
    
    def __init__(self, alpha: float = 0.05, random_state: int = 42):
        """
        Initialize model comparison framework.
        
        Args:
            alpha: Significance level
            random_state: Random seed for reproducibility
        """
        self.stats = StatisticalSignificance(alpha=alpha, random_state=random_state)
        self.alpha = alpha
    
    def compare_neural_vs_baseline(self,
                                  neural_results: ModelResults,
                                  baseline_results: ModelResults,
                                  metrics: List[str] = ['f1', 'accuracy']) -> ComparisonReport:
        """
        Compare neural model against baseline models.
        
        Args:
            neural_results: Results from neural model
            baseline_results: Results from baseline model
            metrics: Metrics to compare
        
        Returns:
            Comprehensive comparison report
        """
        tests = []
        summary_data = []
        
        for metric in metrics:
            if metric not in neural_results.scores or metric not in baseline_results.scores:
                continue
            
            neural_scores = neural_results.scores[metric]
            baseline_scores = baseline_results.scores[metric]
            
            # Permutation test
            perm_test = self.stats.permutation_test(
                neural_scores, baseline_scores,
                statistic='mean_diff'
            )
            perm_test.test_name = f"Permutation Test ({metric})"
            tests.append(perm_test)
            
            # Bootstrap comparison
            boot_test = self.stats.bootstrap_comparison(
                neural_scores, baseline_scores
            )
            boot_test.test_name = f"Bootstrap Comparison ({metric})"
            tests.append(boot_test)
            
            # If we have paired predictions, do McNemar test
            if (neural_results.predictions is not None and 
                baseline_results.predictions is not None and
                neural_results.labels is not None):
                
                mcnemar = self.stats.mcnemar_test(
                    neural_results.predictions,
                    baseline_results.predictions,
                    neural_results.labels
                )
                mcnemar.test_name = f"McNemar Test ({metric})"
                tests.append(mcnemar)
            
            # Summary statistics
            summary_data.append({
                'Metric': metric,
                'Neural Mean': np.mean(neural_scores),
                'Neural Std': np.std(neural_scores),
                'Baseline Mean': np.mean(baseline_scores),
                'Baseline Std': np.std(baseline_scores),
                'Difference': np.mean(neural_scores) - np.mean(baseline_scores),
                'Effect Size': self.stats.cohens_d(neural_scores, baseline_scores),
                'p-value': perm_test.p_value,
                'Significant': perm_test.significant
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations_neural_baseline(tests, summary_df)
        
        return ComparisonReport(
            comparison_type="Neural vs Baseline",
            models=[neural_results.model_name, baseline_results.model_name],
            statistical_tests=tests,
            summary_table=summary_df,
            recommendations=recommendations
        )
    
    def compare_character_modes(self,
                               isolated_results: ModelResults,
                               cross_episode_results: ModelResults,
                               metrics: List[str] = ['f1', 'accuracy']) -> ComparisonReport:
        """
        Compare episode-isolated vs cross-episode character modes.
        
        Args:
            isolated_results: Results from episode-isolated mode
            cross_episode_results: Results from cross-episode mode
            metrics: Metrics to compare
        
        Returns:
            Comparison report
        """
        tests = []
        summary_data = []
        
        for metric in metrics:
            if metric not in isolated_results.scores or metric not in cross_episode_results.scores:
                continue
            
            iso_scores = isolated_results.scores[metric]
            cross_scores = cross_episode_results.scores[metric]
            
            # Wilcoxon test (paired samples if from same folds)
            if len(iso_scores) == len(cross_scores):
                wilcoxon = self.stats.wilcoxon_test(iso_scores, cross_scores)
                wilcoxon.test_name = f"Wilcoxon Test ({metric})"
                tests.append(wilcoxon)
            
            # Permutation test
            perm_test = self.stats.permutation_test(
                iso_scores, cross_scores,
                statistic='mean_diff'
            )
            perm_test.test_name = f"Permutation Test ({metric})"
            tests.append(perm_test)
            
            # Effect size
            effect = self.stats.cliffs_delta(iso_scores, cross_scores)
            
            summary_data.append({
                'Metric': metric,
                'Episode-Isolated Mean': np.mean(iso_scores),
                'Episode-Isolated Std': np.std(iso_scores),
                'Cross-Episode Mean': np.mean(cross_scores),
                'Cross-Episode Std': np.std(cross_scores),
                'Difference': np.mean(iso_scores) - np.mean(cross_scores),
                'Cliff\'s Delta': effect,
                'p-value': perm_test.p_value,
                'Significant': perm_test.significant
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations_character_modes(tests, summary_df)
        
        return ComparisonReport(
            comparison_type="Character Mode Comparison",
            models=["Episode-Isolated", "Cross-Episode"],
            statistical_tests=tests,
            summary_table=summary_df,
            recommendations=recommendations
        )
    
    def compare_cv_paradigms(self,
                           sequential_results: ModelResults,
                           parallel_results: ModelResults,
                           metrics: List[str] = ['f1', 'accuracy']) -> ComparisonReport:
        """
        Compare Sequential CV vs Parallel CV training paradigms.
        
        Args:
            sequential_results: Results from sequential CV
            parallel_results: Results from parallel CV
            metrics: Metrics to compare
        
        Returns:
            Comparison report with paradigm-specific insights
        """
        tests = []
        summary_data = []
        
        for metric in metrics:
            if metric not in sequential_results.scores or metric not in parallel_results.scores:
                continue
            
            seq_scores = sequential_results.scores[metric]
            par_scores = parallel_results.scores[metric]
            
            # Permutation test
            perm_test = self.stats.permutation_test(
                seq_scores, par_scores,
                statistic='mean_diff'
            )
            perm_test.test_name = f"Permutation Test ({metric})"
            tests.append(perm_test)
            
            # Bootstrap comparison
            boot_test = self.stats.bootstrap_comparison(seq_scores, par_scores)
            boot_test.test_name = f"Bootstrap Comparison ({metric})"
            tests.append(boot_test)
            
            # Calculate generalization gap
            gen_gap = np.mean(par_scores) - np.mean(seq_scores)
            
            summary_data.append({
                'Metric': metric,
                'Sequential CV Mean': np.mean(seq_scores),
                'Sequential CV Std': np.std(seq_scores),
                'Parallel CV Mean': np.mean(par_scores),
                'Parallel CV Std': np.std(par_scores),
                'Generalization Gap': gen_gap,
                'Effect Size': self.stats.cohens_d(seq_scores, par_scores),
                'p-value': perm_test.p_value,
                'Significant': perm_test.significant
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Generate paradigm-specific recommendations
        recommendations = self._generate_recommendations_cv_paradigms(tests, summary_df)
        
        return ComparisonReport(
            comparison_type="CV Paradigm Comparison",
            models=["Sequential CV", "Parallel CV"],
            statistical_tests=tests,
            summary_table=summary_df,
            recommendations=recommendations
        )
    
    def compare_multiple_models(self,
                              model_results: Dict[str, ModelResults],
                              metric: str = 'f1',
                              use_friedman: bool = True) -> ComparisonReport:
        """
        Compare multiple models using appropriate statistical tests.
        
        Args:
            model_results: Dictionary of model_name -> ModelResults
            metric: Metric to compare
            use_friedman: Use Friedman test for multiple comparisons
        
        Returns:
            Comprehensive comparison across all models
        """
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            raise ValueError("Need at least 2 models to compare")
        
        # Extract scores
        scores_matrix = []
        for name in model_names:
            if metric in model_results[name].scores:
                scores_matrix.append(model_results[name].scores[metric])
            else:
                raise ValueError(f"Metric {metric} not found for model {name}")
        
        scores_matrix = np.array(scores_matrix)
        
        tests = []
        
        # Friedman test for multiple related samples
        if use_friedman and scores_matrix.shape[1] > 1:
            from scipy.stats import friedmanchisquare
            stat, p_value = friedmanchisquare(*scores_matrix)
            
            friedman_test = TestResult(
                test_name="Friedman Test",
                statistic=stat,
                p_value=p_value,
                significant=(p_value < self.alpha),
                confidence_level=1 - self.alpha,
                interpretation=f"Significant difference among {n_models} models" if p_value < self.alpha else "No significant difference"
            )
            tests.append(friedman_test)
        
        # Pairwise comparisons with correction
        pairwise_p_values = []
        pairwise_comparisons = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                name_i, name_j = model_names[i], model_names[j]
                scores_i = scores_matrix[i]
                scores_j = scores_matrix[j]
                
                # Permutation test for each pair
                perm_test = self.stats.permutation_test(scores_i, scores_j)
                pairwise_p_values.append(perm_test.p_value)
                pairwise_comparisons.append((name_i, name_j, perm_test))
        
        # Apply multiple testing correction
        if len(pairwise_p_values) > 1:
            corrected_p, reject = self.stats.benjamini_hochberg(pairwise_p_values)
            
            for idx, (name_i, name_j, test) in enumerate(pairwise_comparisons):
                test.p_value = corrected_p[idx]
                test.significant = reject[idx]
                test.test_name = f"Pairwise: {name_i} vs {name_j}"
                test.interpretation = f"FDR-corrected p-value: {corrected_p[idx]:.4f}"
                tests.append(test)
        
        # Create summary table
        summary_data = []
        for name in model_names:
            scores = model_results[name].scores[metric]
            ci = self.stats.bootstrap_ci(scores)
            
            summary_data.append({
                'Model': name,
                'Mean': np.mean(scores),
                'Std': np.std(scores),
                'CI Lower': ci[0],
                'CI Upper': ci[1],
                'Rank': 0  # Will be filled
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df['Rank'] = summary_df['Mean'].rank(ascending=False).astype(int)
        summary_df = summary_df.sort_values('Rank')
        
        # Generate recommendations
        recommendations = self._generate_recommendations_multiple(tests, summary_df, model_names)
        
        return ComparisonReport(
            comparison_type="Multiple Model Comparison",
            models=model_names,
            statistical_tests=tests,
            summary_table=summary_df,
            recommendations=recommendations
        )
    
    # ================== Helper Methods ==================
    
    def _generate_recommendations_neural_baseline(self, 
                                                 tests: List[TestResult],
                                                 summary_df: pd.DataFrame) -> List[str]:
        """Generate recommendations for neural vs baseline comparison."""
        recommendations = []
        
        # Check if neural significantly outperforms
        significant_improvements = summary_df[summary_df['Significant'] == True]
        
        if len(significant_improvements) > 0:
            recommendations.append(
                f"Neural model shows significant improvement on {len(significant_improvements)} metrics"
            )
            
            # Check effect sizes
            large_effects = summary_df[summary_df['Effect Size'].abs() > 0.8]
            if len(large_effects) > 0:
                recommendations.append(
                    f"Large effect sizes observed for {', '.join(large_effects['Metric'].tolist())}"
                )
        else:
            recommendations.append(
                "No significant improvement over baseline - consider model refinement"
            )
        
        # Check consistency
        if 'f1' in summary_df['Metric'].values:
            f1_row = summary_df[summary_df['Metric'] == 'f1'].iloc[0]
            if f1_row['Neural Std'] > f1_row['Baseline Std'] * 1.5:
                recommendations.append(
                    "Neural model shows higher variance - consider regularization"
                )
        
        return recommendations
    
    def _generate_recommendations_character_modes(self,
                                                 tests: List[TestResult],
                                                 summary_df: pd.DataFrame) -> List[str]:
        """Generate recommendations for character mode comparison."""
        recommendations = []
        
        # Check which mode performs better
        for _, row in summary_df.iterrows():
            if row['Significant']:
                if row['Difference'] > 0:
                    recommendations.append(
                        f"Episode-isolated mode significantly better for {row['Metric']}"
                    )
                else:
                    recommendations.append(
                        f"Cross-episode mode significantly better for {row['Metric']}"
                    )
        
        # Check variance differences
        iso_var = summary_df['Episode-Isolated Std'].mean()
        cross_var = summary_df['Cross-Episode Std'].mean()
        
        if iso_var < cross_var * 0.8:
            recommendations.append(
                "Episode-isolated mode shows more stable performance"
            )
        elif cross_var < iso_var * 0.8:
            recommendations.append(
                "Cross-episode mode shows more stable performance"
            )
        
        # Character consolidation insight
        if summary_df['Difference'].mean() > 0.05:
            recommendations.append(
                "Consider using episode-isolated for better character distinction"
            )
        elif summary_df['Difference'].mean() < -0.05:
            recommendations.append(
                "Cross-episode consolidation helps with limited data"
            )
        
        return recommendations
    
    def _generate_recommendations_cv_paradigms(self,
                                              tests: List[TestResult],
                                              summary_df: pd.DataFrame) -> List[str]:
        """Generate recommendations for CV paradigm comparison."""
        recommendations = []
        
        # Check generalization gap
        gen_gaps = summary_df['Generalization Gap'].values
        avg_gap = np.mean(gen_gaps)
        
        if avg_gap > 0.1:
            recommendations.append(
                f"Large generalization gap ({avg_gap:.3f}) suggests information leakage in Parallel CV"
            )
            recommendations.append(
                "Sequential CV provides more reliable generalization estimates"
            )
        elif avg_gap > 0.05:
            recommendations.append(
                "Moderate generalization gap - Sequential CV recommended for publication"
            )
        else:
            recommendations.append(
                "Small generalization gap - both paradigms provide similar estimates"
            )
        
        # Check significance
        if any(summary_df['Significant']):
            recommendations.append(
                "Statistically significant difference between paradigms detected"
            )
            recommendations.append(
                "Use Sequential CV for theoretical rigor in final results"
            )
        
        return recommendations
    
    def _generate_recommendations_multiple(self,
                                          tests: List[TestResult],
                                          summary_df: pd.DataFrame,
                                          model_names: List[str]) -> List[str]:
        """Generate recommendations for multiple model comparison."""
        recommendations = []
        
        # Best model
        best_model = summary_df.iloc[0]['Model']
        recommendations.append(f"Best performing model: {best_model}")
        
        # Check if there's a clear winner
        if len(summary_df) > 1:
            diff = summary_df.iloc[0]['Mean'] - summary_df.iloc[1]['Mean']
            if diff > 0.1:
                recommendations.append(
                    f"{best_model} substantially outperforms others (>{diff:.3f} difference)"
                )
        
        # Friedman test result
        friedman_tests = [t for t in tests if t.test_name == "Friedman Test"]
        if friedman_tests and friedman_tests[0].significant:
            recommendations.append(
                "Significant differences exist among models (Friedman test)"
            )
        
        # Consistency check
        std_variation = summary_df['Std'].std()
        if std_variation > 0.05:
            recommendations.append(
                "High variance in model stability - consider ensemble methods"
            )
        
        return recommendations
    
    def save_report(self, report: ComparisonReport, output_path: Path):
        """Save comparison report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        # Save summary table as CSV
        csv_path = output_path.with_suffix('.csv')
        report.summary_table.to_csv(csv_path, index=False)
        
        print(f"Report saved to {output_path}")
        print(f"Summary table saved to {csv_path}")


def demo_model_comparison():
    """Demonstrate model comparison functionality."""
    print("Model Comparison Demo")
    print("=" * 60)
    
    # Create synthetic results
    np.random.seed(42)
    
    # Neural model results
    neural = ModelResults(
        model_name="Neural Embeddings",
        scores={
            'f1': np.random.normal(0.75, 0.05, 5).tolist(),
            'accuracy': np.random.normal(0.80, 0.04, 5).tolist()
        }
    )
    
    # Baseline results
    baseline = ModelResults(
        model_name="TF-IDF + SVM",
        scores={
            'f1': np.random.normal(0.65, 0.07, 5).tolist(),
            'accuracy': np.random.normal(0.70, 0.06, 5).tolist()
        }
    )
    
    # Episode-isolated results
    isolated = ModelResults(
        model_name="Episode-Isolated",
        scores={
            'f1': np.random.normal(0.72, 0.04, 5).tolist(),
            'accuracy': np.random.normal(0.78, 0.03, 5).tolist()
        }
    )
    
    # Cross-episode results
    cross_episode = ModelResults(
        model_name="Cross-Episode",
        scores={
            'f1': np.random.normal(0.68, 0.06, 5).tolist(),
            'accuracy': np.random.normal(0.74, 0.05, 5).tolist()
        }
    )
    
    # Initialize comparison framework
    comparison = ModelComparison(alpha=0.05)
    
    # 1. Neural vs Baseline
    print("\n1. Neural vs Baseline Comparison")
    report1 = comparison.compare_neural_vs_baseline(neural, baseline)
    report1.print_summary()
    
    # 2. Character Modes
    print("\n2. Character Mode Comparison")
    report2 = comparison.compare_character_modes(isolated, cross_episode)
    report2.print_summary()
    
    # 3. Multiple models
    print("\n3. Multiple Model Comparison")
    all_models = {
        "Neural": neural,
        "Baseline": baseline,
        "Isolated": isolated,
        "Cross-Episode": cross_episode
    }
    report3 = comparison.compare_multiple_models(all_models, metric='f1')
    report3.print_summary()
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo_model_comparison()