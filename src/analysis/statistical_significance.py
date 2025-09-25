"""
Statistical Significance Testing Framework
==========================================

Core statistical testing module for validating CSI character embedding results.
Provides rigorous hypothesis testing, confidence intervals, and effect sizes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy import stats
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float = 0.95
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None
    
    def __str__(self):
        """Pretty print test results."""
        sig_symbol = "✓" if self.significant else "✗"
        result = f"{self.test_name}:\n"
        result += f"  Statistic: {self.statistic:.4f}\n"
        result += f"  p-value: {self._format_p_value(self.p_value)}\n"
        result += f"  Significant: {sig_symbol} (α={1-self.confidence_level:.2f})\n"
        
        if self.effect_size is not None:
            result += f"  Effect size: {self.effect_size:.3f} ({self._interpret_effect_size()})\n"
        
        if self.confidence_interval is not None:
            result += f"  {self.confidence_level*100:.0f}% CI: [{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}]\n"
        
        if self.interpretation:
            result += f"  Interpretation: {self.interpretation}\n"
        
        return result
    
    def _format_p_value(self, p: float) -> str:
        """Format p-value for display."""
        if p < 0.001:
            return "< 0.001"
        elif p < 0.01:
            return f"{p:.3f}"
        else:
            return f"{p:.4f}"
    
    def _interpret_effect_size(self) -> str:
        """Interpret Cohen's d effect size."""
        if self.effect_size is None:
            return ""
        
        d = abs(self.effect_size)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


class StatisticalSignificance:
    """
    Core statistical testing framework for model validation.
    
    Provides:
    - Permutation tests for model comparison
    - Bootstrap confidence intervals
    - Paired comparison tests (McNemar, Wilcoxon)
    - Effect size calculations
    - Multiple testing corrections
    """
    
    def __init__(self, 
                 alpha: float = 0.05,
                 n_bootstrap: int = 10000,
                 n_permutations: int = 10000,
                 random_state: int = 42):
        """
        Initialize statistical testing framework.
        
        Args:
            alpha: Significance level (default 0.05 for 95% confidence)
            n_bootstrap: Number of bootstrap samples
            n_permutations: Number of permutations for permutation tests
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.random_state = random_state
        np.random.seed(random_state)
    
    # ================== Permutation Testing ==================
    
    def permutation_test(self, 
                        scores_a: Union[List[float], np.ndarray],
                        scores_b: Union[List[float], np.ndarray],
                        statistic: str = 'mean_diff',
                        alternative: str = 'two-sided') -> TestResult:
        """
        Non-parametric permutation test for difference between two sets of scores.
        
        Args:
            scores_a: First set of scores (e.g., neural model F1 scores)
            scores_b: Second set of scores (e.g., baseline F1 scores)
            statistic: Test statistic ('mean_diff', 'median_diff', 't_statistic')
            alternative: 'two-sided', 'greater', or 'less'
        
        Returns:
            TestResult with p-value and test details
        """
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        # Calculate observed statistic
        if statistic == 'mean_diff':
            observed = np.mean(scores_a) - np.mean(scores_b)
            stat_func = lambda a, b: np.mean(a) - np.mean(b)
        elif statistic == 'median_diff':
            observed = np.median(scores_a) - np.median(scores_b)
            stat_func = lambda a, b: np.median(a) - np.median(b)
        elif statistic == 't_statistic':
            observed = self._calculate_t_statistic(scores_a, scores_b)
            stat_func = lambda a, b: self._calculate_t_statistic(a, b)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        # Combine scores for permutation
        combined = np.concatenate([scores_a, scores_b])
        n_a = len(scores_a)
        
        # Generate permutation distribution
        perm_stats = []
        for _ in range(self.n_permutations):
            np.random.shuffle(combined)
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]
            perm_stats.append(stat_func(perm_a, perm_b))
        
        perm_stats = np.array(perm_stats)
        
        # Calculate p-value
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
        elif alternative == 'greater':
            p_value = np.mean(perm_stats >= observed)
        elif alternative == 'less':
            p_value = np.mean(perm_stats <= observed)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        # Calculate effect size
        effect_size = self.cohens_d(scores_a, scores_b)
        
        # Create interpretation
        if p_value < self.alpha:
            if observed > 0:
                interpretation = f"Model A significantly outperforms Model B"
            else:
                interpretation = f"Model B significantly outperforms Model A"
        else:
            interpretation = "No significant difference between models"
        
        return TestResult(
            test_name="Permutation Test",
            statistic=observed,
            p_value=p_value,
            significant=(p_value < self.alpha),
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=interpretation
        )
    
    def permutation_test_paired(self,
                               predictions_a: Union[List[int], np.ndarray],
                               predictions_b: Union[List[int], np.ndarray],
                               labels: Union[List[int], np.ndarray]) -> TestResult:
        """
        Permutation test for paired predictions (same test set).
        
        Args:
            predictions_a: Predictions from model A
            predictions_b: Predictions from model B
            labels: True labels
        
        Returns:
            TestResult with paired comparison
        """
        predictions_a = np.array(predictions_a)
        predictions_b = np.array(predictions_b)
        labels = np.array(labels)
        
        # Calculate accuracy difference
        correct_a = (predictions_a == labels).astype(int)
        correct_b = (predictions_b == labels).astype(int)
        
        observed_diff = np.mean(correct_a) - np.mean(correct_b)
        
        # Permutation test on paired differences
        differences = correct_a - correct_b
        
        perm_diffs = []
        for _ in range(self.n_permutations):
            # Randomly flip signs
            signs = np.random.choice([-1, 1], size=len(differences))
            perm_diff = np.mean(differences * signs)
            perm_diffs.append(perm_diff)
        
        perm_diffs = np.array(perm_diffs)
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        return TestResult(
            test_name="Paired Permutation Test",
            statistic=observed_diff,
            p_value=p_value,
            significant=(p_value < self.alpha),
            confidence_level=self.confidence_level,
            interpretation=f"Accuracy difference: {observed_diff:.3f}"
        )
    
    # ================== Bootstrap Methods ==================
    
    def bootstrap_ci(self,
                     scores: Union[List[float], np.ndarray],
                     metric: str = 'mean',
                     method: str = 'bca') -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for any metric.
        
        Args:
            scores: Array of scores
            metric: Metric to calculate ('mean', 'median', 'std')
            method: Bootstrap method ('percentile', 'bca', 'basic')
        
        Returns:
            (lower, upper) confidence interval bounds
        """
        scores = np.array(scores)
        
        # Define statistic function
        if metric == 'mean':
            stat_func = np.mean
        elif metric == 'median':
            stat_func = np.median
        elif metric == 'std':
            stat_func = np.std
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Use scipy's bootstrap for BCa method
        if method == 'bca':
            res = bootstrap((scores,), stat_func, 
                          n_resamples=self.n_bootstrap,
                          confidence_level=self.confidence_level,
                          method='BCa',
                          random_state=self.random_state)
            return (res.confidence_interval.low, res.confidence_interval.high)
        
        # Manual bootstrap for other methods
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_stats.append(stat_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        if method == 'percentile':
            lower = np.percentile(bootstrap_stats, (self.alpha/2) * 100)
            upper = np.percentile(bootstrap_stats, (1 - self.alpha/2) * 100)
        elif method == 'basic':
            observed = stat_func(scores)
            lower = 2 * observed - np.percentile(bootstrap_stats, (1 - self.alpha/2) * 100)
            upper = 2 * observed - np.percentile(bootstrap_stats, (self.alpha/2) * 100)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return (lower, upper)
    
    def bootstrap_comparison(self,
                           scores_a: Union[List[float], np.ndarray],
                           scores_b: Union[List[float], np.ndarray]) -> TestResult:
        """
        Bootstrap comparison of two sets of scores.
        
        Args:
            scores_a: First set of scores
            scores_b: Second set of scores
        
        Returns:
            TestResult with bootstrap confidence interval for difference
        """
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        # Bootstrap the difference
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            sample_a = np.random.choice(scores_a, size=len(scores_a), replace=True)
            sample_b = np.random.choice(scores_b, size=len(scores_b), replace=True)
            bootstrap_diffs.append(np.mean(sample_a) - np.mean(sample_b))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_diffs, (self.alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - self.alpha/2) * 100)
        
        # Test if 0 is in the CI
        significant = not (ci_lower <= 0 <= ci_upper)
        observed_diff = np.mean(scores_a) - np.mean(scores_b)
        
        # Calculate p-value (proportion of bootstrap samples with opposite sign)
        if observed_diff > 0:
            p_value = np.mean(bootstrap_diffs <= 0) * 2  # Two-sided
        else:
            p_value = np.mean(bootstrap_diffs >= 0) * 2
        
        p_value = min(p_value, 1.0)
        
        return TestResult(
            test_name="Bootstrap Comparison",
            statistic=observed_diff,
            p_value=p_value,
            significant=significant,
            confidence_level=self.confidence_level,
            confidence_interval=(ci_lower, ci_upper),
            effect_size=self.cohens_d(scores_a, scores_b),
            interpretation=f"Mean difference: {observed_diff:.3f}"
        )
    
    # ================== Paired Tests ==================
    
    def mcnemar_test(self,
                    predictions_a: Union[List[int], np.ndarray],
                    predictions_b: Union[List[int], np.ndarray],
                    labels: Union[List[int], np.ndarray]) -> TestResult:
        """
        McNemar's test for paired binary predictions.
        
        Args:
            predictions_a: Binary predictions from model A
            predictions_b: Binary predictions from model B
            labels: True binary labels
        
        Returns:
            TestResult with McNemar's test statistics
        """
        predictions_a = np.array(predictions_a)
        predictions_b = np.array(predictions_b)
        labels = np.array(labels)
        
        # Create contingency table
        correct_a = (predictions_a == labels)
        correct_b = (predictions_b == labels)
        
        # Count disagreements
        a_correct_b_wrong = np.sum(correct_a & ~correct_b)
        a_wrong_b_correct = np.sum(~correct_a & correct_b)
        
        # McNemar's test
        n = a_correct_b_wrong + a_wrong_b_correct
        
        if n == 0:
            # Perfect agreement
            return TestResult(
                test_name="McNemar's Test",
                statistic=0,
                p_value=1.0,
                significant=False,
                confidence_level=self.confidence_level,
                interpretation="Perfect agreement between models"
            )
        
        # Use exact binomial test for small samples
        if n < 25:
            p_value = stats.binom.cdf(min(a_correct_b_wrong, a_wrong_b_correct), n, 0.5) * 2
        else:
            # Use chi-squared approximation with continuity correction
            statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / n
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        # Calculate odds ratio
        if a_wrong_b_correct > 0:
            odds_ratio = a_correct_b_wrong / a_wrong_b_correct
        else:
            odds_ratio = float('inf') if a_correct_b_wrong > 0 else 1.0
        
        interpretation = f"Model A correct/B wrong: {a_correct_b_wrong}, "
        interpretation += f"Model B correct/A wrong: {a_wrong_b_correct}"
        
        return TestResult(
            test_name="McNemar's Test",
            statistic=odds_ratio,
            p_value=p_value,
            significant=(p_value < self.alpha),
            confidence_level=self.confidence_level,
            interpretation=interpretation
        )
    
    def wilcoxon_test(self,
                     scores_a: Union[List[float], np.ndarray],
                     scores_b: Union[List[float], np.ndarray],
                     alternative: str = 'two-sided') -> TestResult:
        """
        Wilcoxon signed-rank test for paired samples.
        
        Args:
            scores_a: First set of paired scores
            scores_b: Second set of paired scores
            alternative: 'two-sided', 'greater', or 'less'
        
        Returns:
            TestResult with Wilcoxon test statistics
        """
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        if len(scores_a) != len(scores_b):
            raise ValueError("Scores must be paired (same length)")
        
        # Perform Wilcoxon test
        statistic, p_value = stats.wilcoxon(scores_a, scores_b, alternative=alternative)
        
        # Calculate effect size (rank-biserial correlation)
        differences = scores_a - scores_b
        n = len(differences)
        effect_size = 1 - (2 * statistic) / (n * (n + 1))
        
        # Median difference
        median_diff = np.median(scores_a - scores_b)
        
        return TestResult(
            test_name="Wilcoxon Signed-Rank Test",
            statistic=statistic,
            p_value=p_value,
            significant=(p_value < self.alpha),
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=f"Median difference: {median_diff:.3f}"
        )
    
    # ================== Effect Sizes ==================
    
    def cohens_d(self,
                scores_a: Union[List[float], np.ndarray],
                scores_b: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            scores_a: First set of scores
            scores_b: Second set of scores
        
        Returns:
            Cohen's d effect size
        """
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        mean_diff = np.mean(scores_a) - np.mean(scores_b)
        
        # Pooled standard deviation
        n_a, n_b = len(scores_a), len(scores_b)
        var_a, var_b = np.var(scores_a, ddof=1), np.var(scores_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return mean_diff / pooled_std
    
    def cliffs_delta(self,
                    scores_a: Union[List[float], np.ndarray],
                    scores_b: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Cliff's delta (non-parametric effect size).
        
        Args:
            scores_a: First set of scores
            scores_b: Second set of scores
        
        Returns:
            Cliff's delta (-1 to 1)
        """
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        # Count dominance
        dominance = 0
        for a in scores_a:
            for b in scores_b:
                if a > b:
                    dominance += 1
                elif a < b:
                    dominance -= 1
        
        # Normalize
        n_comparisons = len(scores_a) * len(scores_b)
        if n_comparisons == 0:
            return 0.0
        
        return dominance / n_comparisons
    
    # ================== Multiple Testing Corrections ==================
    
    def bonferroni_correction(self,
                             p_values: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Args:
            p_values: Array of p-values
        
        Returns:
            Corrected p-values
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        return np.minimum(p_values * n_tests, 1.0)
    
    def benjamini_hochberg(self,
                          p_values: Union[List[float], np.ndarray],
                          alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Benjamini-Hochberg FDR correction.
        
        Args:
            p_values: Array of p-values
            alpha: Significance level (uses self.alpha if None)
        
        Returns:
            (corrected_p_values, reject_null) arrays
        """
        p_values = np.array(p_values)
        alpha = alpha or self.alpha
        
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Calculate critical values
        critical_values = (np.arange(1, n + 1) / n) * alpha
        
        # Find largest i where P(i) <= critical_value(i)
        reject = sorted_p <= critical_values
        if reject.any():
            max_idx = np.max(np.where(reject)[0])
            reject[:max_idx + 1] = True
        
        # Map back to original order
        reject_original = np.zeros(n, dtype=bool)
        reject_original[sorted_idx] = reject
        
        # Calculate adjusted p-values
        adjusted_p = np.zeros(n)
        adjusted_p[sorted_idx] = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            adjusted_p[sorted_idx[i]] = min(adjusted_p[sorted_idx[i]], 
                                           adjusted_p[sorted_idx[i + 1]])
        
        return adjusted_p, reject_original
    
    # ================== Helper Methods ==================
    
    def _calculate_t_statistic(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate t-statistic for two samples."""
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        n_a, n_b = len(a), len(b)
        
        pooled_se = np.sqrt(var_a/n_a + var_b/n_b)
        if pooled_se == 0:
            return 0.0
        
        return (mean_a - mean_b) / pooled_se
    
    def summary_table(self, results: List[TestResult]) -> pd.DataFrame:
        """
        Create summary DataFrame from multiple test results.
        
        Args:
            results: List of TestResult objects
        
        Returns:
            DataFrame with test summaries
        """
        data = []
        for r in results:
            data.append({
                'Test': r.test_name,
                'Statistic': r.statistic,
                'p-value': r.p_value,
                'Significant': r.significant,
                'Effect Size': r.effect_size,
                'CI Lower': r.confidence_interval[0] if r.confidence_interval else None,
                'CI Upper': r.confidence_interval[1] if r.confidence_interval else None,
            })
        
        return pd.DataFrame(data)


def demo_statistical_tests():
    """Demonstrate statistical testing framework."""
    print("Statistical Significance Testing Framework Demo")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    neural_scores = np.random.normal(0.75, 0.05, 10)  # Neural model F1 scores
    baseline_scores = np.random.normal(0.65, 0.07, 10)  # Baseline F1 scores
    
    # Initialize framework
    stats_test = StatisticalSignificance(alpha=0.05)
    
    # 1. Permutation test
    print("\n1. Permutation Test")
    result = stats_test.permutation_test(neural_scores, baseline_scores)
    print(result)
    
    # 2. Bootstrap comparison
    print("\n2. Bootstrap Comparison")
    result = stats_test.bootstrap_comparison(neural_scores, baseline_scores)
    print(result)
    
    # 3. Bootstrap confidence interval
    print("\n3. Bootstrap Confidence Interval for Neural Model")
    ci = stats_test.bootstrap_ci(neural_scores, metric='mean')
    print(f"   95% CI for mean: [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    # 4. Effect size
    print("\n4. Effect Sizes")
    d = stats_test.cohens_d(neural_scores, baseline_scores)
    delta = stats_test.cliffs_delta(neural_scores, baseline_scores)
    print(f"   Cohen's d: {d:.3f}")
    print(f"   Cliff's delta: {delta:.3f}")
    
    # 5. Multiple testing correction
    print("\n5. Multiple Testing Correction")
    p_values = [0.01, 0.04, 0.03, 0.20, 0.001]
    corrected_p = stats_test.bonferroni_correction(p_values)
    print(f"   Original p-values: {p_values}")
    print(f"   Bonferroni corrected: {corrected_p.round(3).tolist()}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo_statistical_tests()