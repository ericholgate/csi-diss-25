"""
Unified Analysis Runner
========================

Run all analysis modules with consistent configuration for both
character modes (episode-isolated and cross-episode).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime
from typing import Dict, Any, List

# Import all analysis modules
from analysis.baseline_models import BaselineModels
from analysis.killer_statistics import KillerStatisticsAnalyzer
from analysis.character_frequency_analyzer import CharacterFrequencyAnalyzer
from analysis.sentence_pattern_analyzer import SentencePatternAnalyzer
from analysis.episode_analyzer import EpisodeAnalyzer


class UnifiedAnalysisRunner:
    """Run all analyses in a coordinated manner."""
    
    def __init__(self, data_dir: Path = Path("data/original"),
                 output_dir: Path = Path("analysis/results"),
                 character_modes: List[str] = None):
        """
        Initialize the unified runner.
        
        Args:
            data_dir: Directory containing TSV files
            output_dir: Directory to save all results
            character_modes: List of modes to run ('episode-isolated', 'cross-episode')
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.character_modes = character_modes or ['episode-isolated', 'cross-episode']
        self.results = {}
        
    def run_all_analyses(self, modules: List[str] = None, 
                         generate_plots: bool = False) -> Dict[str, Any]:
        """
        Run all requested analysis modules.
        
        Args:
            modules: List of modules to run (default: all)
            generate_plots: Whether to generate visualization plots
            
        Returns:
            Dictionary containing all analysis results
        """
        available_modules = [
            'baseline', 'killer_stats', 'character_frequency',
            'sentence_patterns', 'episode_structure'
        ]
        
        modules = modules or available_modules
        
        print("\n" + "="*80)
        print("UNIFIED CSI ANALYSIS RUNNER")
        print("="*80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Character modes: {', '.join(self.character_modes)}")
        print(f"Modules to run: {', '.join(modules)}")
        print("="*80)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run each mode
        for mode in self.character_modes:
            print(f"\n{'='*80}")
            print(f"ANALYZING IN {mode.upper()} MODE")
            print(f"{'='*80}")
            
            mode_results = {}
            mode_output_dir = self.output_dir / mode
            mode_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run baseline models
            if 'baseline' in modules:
                print(f"\n[{mode}] Running baseline models...")
                analyzer = BaselineModels(self.data_dir, character_mode=mode)
                results_df = analyzer.run_all_baselines()
                mode_results['baseline'] = {
                    'summary': results_df.to_dict('records'),
                    'best_model': results_df.loc[results_df['F1'].idxmax()].to_dict()
                }
                
            # Run killer statistics
            if 'killer_stats' in modules:
                print(f"\n[{mode}] Running killer statistics analysis...")
                analyzer = KillerStatisticsAnalyzer(self.data_dir, character_mode=mode)
                report_path = mode_output_dir / "killer_statistics.json"
                report = analyzer.generate_report(save_path=report_path)
                mode_results['killer_stats'] = {
                    'overview': report['overview'],
                    'archetypes': report.get('archetypes', {})
                }
                
                if generate_plots:
                    plot_path = mode_output_dir / "killer_distributions.png"
                    analyzer.plot_killer_distributions(save_path=plot_path)
            
            # Run character frequency analysis
            if 'character_frequency' in modules:
                print(f"\n[{mode}] Running character frequency analysis...")
                analyzer = CharacterFrequencyAnalyzer(self.data_dir, character_mode=mode)
                report_path = mode_output_dir / "character_frequency.json"
                report = analyzer.generate_report(save_path=report_path)
                mode_results['character_frequency'] = {
                    'overview': report['overview'],
                    'top_characters': report['top_characters']['by_sentences'][:10]
                }
                
                if generate_plots:
                    dist_path = mode_output_dir / "character_distributions.png"
                    analyzer.plot_character_distributions(save_path=dist_path)
                    network_path = mode_output_dir / "co_occurrence_network.png"
                    analyzer.plot_co_occurrence_network(save_path=network_path)
            
            # Run sentence pattern analysis
            if 'sentence_patterns' in modules:
                print(f"\n[{mode}] Running sentence pattern analysis...")
                analyzer = SentencePatternAnalyzer(self.data_dir, character_mode=mode)
                report_path = mode_output_dir / "sentence_patterns.json"
                report = analyzer.generate_report(save_path=report_path)
                mode_results['sentence_patterns'] = {
                    'overview': report['overview'],
                    'dialogue_complexity': report['dialogue_complexity'].get('overall_stats', {})
                }
            
            # Run episode structure analysis
            if 'episode_structure' in modules:
                print(f"\n[{mode}] Running episode structure analysis...")
                analyzer = EpisodeAnalyzer(self.data_dir, character_mode=mode)
                report_path = mode_output_dir / "episode_structure.json"
                report = analyzer.generate_report(save_path=report_path)
                mode_results['episode_structure'] = {
                    'overview': report['overview'],
                    'character_introduction': report['character_introduction'],
                    'dialogue_flow': report['dialogue_flow']
                }
            
            self.results[mode] = mode_results
        
        # Generate comparative report
        self._generate_comparative_report()
        
        return self.results
    
    def _generate_comparative_report(self):
        """Generate a report comparing results across character modes."""
        if len(self.character_modes) < 2:
            return
        
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS")
        print("="*80)
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'modes_compared': self.character_modes,
            'comparisons': {}
        }
        
        # Compare baseline model performance
        if all('baseline' in self.results.get(mode, {}) for mode in self.character_modes):
            print("\nBaseline Model Performance:")
            for mode in self.character_modes:
                best = self.results[mode]['baseline']['best_model']
                print(f"  {mode}: {best['Model']} (F1={best['F1']:.3f})")
            
            comparison['comparisons']['baseline'] = {
                mode: self.results[mode]['baseline']['best_model']
                for mode in self.character_modes
            }
        
        # Compare character counts
        if all('character_frequency' in self.results.get(mode, {}) for mode in self.character_modes):
            print("\nCharacter Counts:")
            for mode in self.character_modes:
                overview = self.results[mode]['character_frequency']['overview']
                print(f"  {mode}: {overview['total_characters']} total, "
                      f"{overview['main_characters']} main")
            
            comparison['comparisons']['character_counts'] = {
                mode: self.results[mode]['character_frequency']['overview']
                for mode in self.character_modes
            }
        
        # Compare killer statistics
        if all('killer_stats' in self.results.get(mode, {}) for mode in self.character_modes):
            print("\nKiller Statistics:")
            for mode in self.character_modes:
                overview = self.results[mode]['killer_stats']['overview']
                print(f"  {mode}: {overview.get('avg_killers_per_episode', 0):.2f} killers/episode, "
                      f"speaking ratio={overview.get('avg_killer_speaking_ratio', 0):.3f}")
            
            comparison['comparisons']['killer_stats'] = {
                mode: self.results[mode]['killer_stats']['overview']
                for mode in self.character_modes
            }
        
        # Save comparative report
        comparison_path = self.output_dir / "comparative_analysis.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nComparative report saved to {comparison_path}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a high-level summary of all analyses."""
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_directory': str(self.data_dir),
            'output_directory': str(self.output_dir),
            'character_modes': self.character_modes,
            'key_findings': {}
        }
        
        # Extract key findings from each mode
        for mode in self.character_modes:
            if mode not in self.results:
                continue
                
            mode_findings = {}
            
            # Baseline performance
            if 'baseline' in self.results[mode]:
                best_model = self.results[mode]['baseline']['best_model']
                mode_findings['best_baseline_model'] = {
                    'name': best_model['Model'],
                    'f1_score': best_model['F1']
                }
            
            # Character statistics
            if 'character_frequency' in self.results[mode]:
                overview = self.results[mode]['character_frequency']['overview']
                mode_findings['character_statistics'] = {
                    'total': overview['total_characters'],
                    'main': overview['main_characters'],
                    'recurring': overview['recurring_characters']
                }
            
            # Killer patterns
            if 'killer_stats' in self.results[mode]:
                overview = self.results[mode]['killer_stats']['overview']
                mode_findings['killer_patterns'] = {
                    'avg_per_episode': overview.get('avg_killers_per_episode', 0),
                    'speaking_ratio': overview.get('avg_killer_speaking_ratio', 0)
                }
            
            summary['key_findings'][mode] = mode_findings
        
        # Save summary
        summary_path = self.output_dir / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary report saved to {summary_path}")
        
        return summary


def main():
    """Run unified analysis from command line."""
    parser = argparse.ArgumentParser(description="Run all CSI analyses")
    parser.add_argument('--data-dir', type=Path, default=Path("data/original"),
                       help='Directory containing TSV files')
    parser.add_argument('--output-dir', type=Path, default=Path("analysis/unified_results"),
                       help='Directory to save all results')
    parser.add_argument('--modes', nargs='+', 
                       choices=['episode-isolated', 'cross-episode'],
                       default=['episode-isolated', 'cross-episode'],
                       help='Character modes to analyze')
    parser.add_argument('--modules', nargs='+',
                       choices=['baseline', 'killer_stats', 'character_frequency',
                               'sentence_patterns', 'episode_structure'],
                       help='Specific modules to run (default: all)')
    parser.add_argument('--plots', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Create and run unified analysis
    runner = UnifiedAnalysisRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        character_modes=args.modes
    )
    
    results = runner.run_all_analyses(
        modules=args.modules,
        generate_plots=args.plots
    )
    
    # Generate final summary
    runner.generate_summary_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All results saved to: {args.output_dir}")
    print("\nKey output files:")
    print(f"  - {args.output_dir}/analysis_summary.json")
    print(f"  - {args.output_dir}/comparative_analysis.json")
    for mode in args.modes:
        print(f"  - {args.output_dir}/{mode}/ (mode-specific results)")


if __name__ == "__main__":
    main()