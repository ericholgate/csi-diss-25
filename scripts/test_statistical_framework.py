#!/usr/bin/env python3
"""
Test Statistical Framework with Baseline Models
===============================================

Demonstrates the statistical significance testing framework using
actual baseline model results from our CSI data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.baseline_models import BaselineModels
from analysis.statistical_significance import StatisticalSignificance
from analysis.model_comparison import ModelComparison, ModelResults
from analysis.cv_statistics import CVStatistics, FoldResults


def load_baseline_results(character_mode: str = 'episode-isolated') -> dict:
    """Load baseline results if they exist, otherwise run baselines."""
    results_dir = Path(f'experiments/baseline_results_{character_mode}')
    
    if results_dir.exists():
        # Try to load existing results
        scores_file = results_dir / 'baseline_scores.csv'
        detailed_file = results_dir / 'baseline_detailed.json'
        
        if scores_file.exists() and detailed_file.exists():
            print(f"Loading existing baseline results from {results_dir}")
            scores_df = pd.read_csv(scores_file)
            with open(detailed_file, 'r') as f:
                detailed = json.load(f)
            return {'scores_df': scores_df, 'detailed': detailed}
    
    # Run baselines if no results exist
    print(f"Running baseline models in {character_mode} mode...")
    baselines = BaselineModels(Path('data/original'), character_mode=character_mode)
    scores_df = baselines.run_all_baselines()
    
    # Load the detailed results that were just saved
    with open(results_dir / 'baseline_detailed.json', 'r') as f:
        detailed = json.load(f)
    
    return {'scores_df': scores_df, 'detailed': detailed}


def test_baseline_comparisons():
    """Test statistical comparisons between baseline models."""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS OF BASELINE MODELS")
    print("="*80)
    
    # Load results for both character modes
    iso_results = load_baseline_results('episode-isolated')
    cross_results = load_baseline_results('cross-episode')
    
    # Initialize statistical framework
    stats = StatisticalSignificance(alpha=0.05)
    comparison = ModelComparison(alpha=0.05)
    
    # ========== 1. Compare Best vs Worst Baseline ==========
    print("\n1. COMPARING BEST VS WORST BASELINE (Episode-Isolated)")
    print("-" * 60)
    
    iso_df = iso_results['scores_df']
    best_idx = iso_df['F1'].idxmax()
    worst_idx = iso_df['F1'].idxmin()
    
    best_model = iso_df.loc[best_idx, 'Model']
    worst_model = iso_df.loc[worst_idx, 'Model']
    
    print(f"Best model: {best_model} (F1={iso_df.loc[best_idx, 'F1']:.3f})")
    print(f"Worst model: {worst_model} (F1={iso_df.loc[worst_idx, 'F1']:.3f})")
    
    # Get CV scores if available
    best_detailed = next((d for d in iso_results['detailed'] if d['model_name'] == best_model), None)
    worst_detailed = next((d for d in iso_results['detailed'] if d['model_name'] == worst_model), None)
    
    if best_detailed and worst_detailed and best_detailed.get('cv_scores') and worst_detailed.get('cv_scores'):
        best_scores = best_detailed['cv_scores']
        worst_scores = worst_detailed['cv_scores']
        
        # Permutation test
        print("\nPermutation Test:")
        perm_result = stats.permutation_test(best_scores, worst_scores)
        print(perm_result)
        
        # Bootstrap comparison
        print("\nBootstrap Comparison:")
        boot_result = stats.bootstrap_comparison(best_scores, worst_scores)
        print(boot_result)
        
        # Effect size
        effect = stats.cohens_d(best_scores, worst_scores)
        print(f"\nCohen's d effect size: {effect:.3f}")
    
    # ========== 2. Character Mode Comparison ==========
    print("\n2. CHARACTER MODE COMPARISON (Same Model)")
    print("-" * 60)
    
    # Compare TF-IDF model across character modes
    model_name = "TF-IDF + SVM"
    
    iso_model = iso_df[iso_df['Model'] == model_name]
    cross_model = cross_results['scores_df'][cross_results['scores_df']['Model'] == model_name]
    
    if not iso_model.empty and not cross_model.empty:
        print(f"Model: {model_name}")
        print(f"Episode-Isolated F1: {iso_model['F1'].values[0]:.3f}")
        print(f"Cross-Episode F1: {cross_model['F1'].values[0]:.3f}")
        
        # Get detailed scores
        iso_detailed = next((d for d in iso_results['detailed'] if d['model_name'] == model_name), None)
        cross_detailed = next((d for d in cross_results['detailed'] if d['model_name'] == model_name), None)
        
        if iso_detailed and cross_detailed and iso_detailed.get('cv_scores') and cross_detailed.get('cv_scores'):
            iso_scores = iso_detailed['cv_scores']
            cross_scores = cross_detailed['cv_scores']
            
            # Create ModelResults objects
            iso_model_results = ModelResults(
                model_name=f"{model_name} (Episode-Isolated)",
                scores={'f1': iso_scores, 'accuracy': [iso_model['Accuracy'].values[0]] * len(iso_scores)}
            )
            
            cross_model_results = ModelResults(
                model_name=f"{model_name} (Cross-Episode)",
                scores={'f1': cross_scores, 'accuracy': [cross_model['Accuracy'].values[0]] * len(cross_scores)}
            )
            
            # Compare character modes
            mode_report = comparison.compare_character_modes(
                iso_model_results, 
                cross_model_results,
                metrics=['f1']
            )
            mode_report.print_summary()
    
    # ========== 3. Multiple Model Comparison ==========
    print("\n3. MULTIPLE MODEL COMPARISON (Episode-Isolated)")
    print("-" * 60)
    
    # Create ModelResults for each baseline
    all_models = {}
    for _, row in iso_df.iterrows():
        model_name = row['Model']
        detailed = next((d for d in iso_results['detailed'] if d['model_name'] == model_name), None)
        
        if detailed and detailed.get('cv_scores'):
            all_models[model_name] = ModelResults(
                model_name=model_name,
                scores={'f1': detailed['cv_scores']}
            )
    
    if len(all_models) >= 2:
        multi_report = comparison.compare_multiple_models(all_models, metric='f1')
        multi_report.print_summary()
    
    # ========== 4. Cross-Validation Analysis ==========
    print("\n4. CROSS-VALIDATION STABILITY ANALYSIS")
    print("-" * 60)
    
    # Analyze CV stability for models with CV scores
    cv_stats = CVStatistics(alpha=0.05)
    
    # Create synthetic fold results from CV scores
    for model_name, detailed in zip(iso_df['Model'], iso_results['detailed']):
        if detailed.get('cv_scores'):
            cv_scores = detailed['cv_scores']
            
            # Create FoldResults objects
            fold_results = []
            for i, score in enumerate(cv_scores):
                fold = FoldResults(
                    fold_id=i,
                    train_scores={'f1': score + 0.05, 'accuracy': score + 0.08},  # Simulate train scores
                    val_scores={'f1': score, 'accuracy': score + 0.03}
                )
                fold_results.append(fold)
            
            if len(fold_results) >= 3:
                print(f"\nModel: {model_name}")
                
                # Variance analysis
                variance = cv_stats.analyze_fold_variance(fold_results, metrics=['f1'])
                if 'f1' in variance:
                    stats_f1 = variance['f1']
                    print(f"  CV Coefficient of Variation: {stats_f1['val_cv']:.3f}")
                    print(f"  Range: [{stats_f1['val_min']:.3f}, {stats_f1['val_max']:.3f}]")
                
                # Consistency test
                consistency = cv_stats.test_fold_consistency(fold_results, metric='f1')
                print(f"  Fold Consistency: {consistency.interpretation}")
                
                # Outlier detection
                outliers = cv_stats.identify_outlier_folds(fold_results, metric='f1')
                if outliers:
                    print(f"  Outlier folds: {outliers}")
                
                break  # Just show one model for demo
    
    # ========== 5. Statistical Summary ==========
    print("\n5. STATISTICAL SUMMARY")
    print("-" * 60)
    
    # Test all baseline F1 scores for normality
    all_f1_scores = iso_df['F1'].values
    
    # Shapiro-Wilk test for normality
    from scipy import stats as scipy_stats
    statistic, p_value = scipy_stats.shapiro(all_f1_scores)
    print(f"Normality test (Shapiro-Wilk):")
    print(f"  Statistic: {statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Normal distribution: {'Yes' if p_value > 0.05 else 'No'}")
    
    # Multiple testing correction example
    print("\nMultiple Testing Correction Example:")
    print("If we compared all pairs of models:")
    n_models = len(iso_df)
    n_comparisons = n_models * (n_models - 1) // 2
    print(f"  Number of pairwise comparisons: {n_comparisons}")
    
    # Simulate some p-values
    np.random.seed(42)
    simulated_p_values = np.random.beta(1, 10, n_comparisons)  # Skewed toward low p-values
    
    # Apply corrections
    bonferroni_p = stats.bonferroni_correction(simulated_p_values)
    fdr_p, fdr_reject = stats.benjamini_hochberg(simulated_p_values)
    
    print(f"  Significant (uncorrected, α=0.05): {np.sum(simulated_p_values < 0.05)}")
    print(f"  Significant (Bonferroni): {np.sum(bonferroni_p < 0.05)}")
    print(f"  Significant (FDR): {np.sum(fdr_reject)}")
    
    # ========== 6. Recommendations ==========
    print("\n6. OVERALL RECOMMENDATIONS")
    print("-" * 60)
    
    # Based on the analysis
    recommendations = []
    
    # Check best model performance
    best_f1 = iso_df['F1'].max()
    if best_f1 > 0.7:
        recommendations.append(f"✓ Baseline achieves good performance (F1={best_f1:.3f})")
        recommendations.append("  → Neural model should target >10% improvement")
    else:
        recommendations.append(f"⚠ Baseline performance is moderate (F1={best_f1:.3f})")
        recommendations.append("  → Significant room for neural model improvement")
    
    # Check model variance
    f1_std = iso_df['F1'].std()
    if f1_std > 0.1:
        recommendations.append(f"✓ Large variance between models (σ={f1_std:.3f})")
        recommendations.append("  → Choice of model matters significantly")
    else:
        recommendations.append(f"⚠ Small variance between models (σ={f1_std:.3f})")
        recommendations.append("  → All baselines perform similarly")
    
    # Character mode recommendation
    iso_mean = iso_df['F1'].mean()
    cross_mean = cross_results['scores_df']['F1'].mean()
    if abs(iso_mean - cross_mean) > 0.05:
        if iso_mean > cross_mean:
            recommendations.append("✓ Episode-isolated mode performs better")
            recommendations.append("  → Use episode-isolated for neural training")
        else:
            recommendations.append("✓ Cross-episode mode performs better")
            recommendations.append("  → Consider cross-episode for data efficiency")
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "="*80)
    print("Statistical analysis complete!")
    print("="*80)


def main():
    """Run statistical framework tests."""
    try:
        test_baseline_comparisons()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()