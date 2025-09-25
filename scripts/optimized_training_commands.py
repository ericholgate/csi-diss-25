#!/usr/bin/env python3
"""
Optimized Training Commands for GPU Experiments
===============================================

Generates and executes optimized training commands to minimize GPU time.
Includes parallelization strategies and efficient resource usage.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from datetime import datetime
import os


class OptimizedTrainingRunner:
    """Generate and run optimized training commands."""
    
    def __init__(self, gpu_device: int = 0):
        """Initialize training runner.
        
        Args:
            gpu_device: CUDA device ID to use
        """
        self.gpu_device = gpu_device
        self.base_dir = Path(__file__).parent.parent
        self.experiments_dir = self.base_dir / 'experiments'
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Training configurations
        self.configs = {
            'sequential_cv': {
                'script': 'src/models/train_sequential_cv.py',
                'batch_size': 32,  # Optimized for T4 GPU memory
                'epochs': 10,
                'learning_rate': 2e-5,
                'warmup_steps': 100,
                'gradient_accumulation': 2  # Effective batch size = 64
            },
            'parallel_cv': {
                'script': 'src/models/train_parallel_cv.py',
                'batch_size': 32,
                'epochs': 10,
                'learning_rate': 2e-5,
                'warmup_steps': 100,
                'gradient_accumulation': 2
            }
        }
        
        # Character modes to test
        self.character_modes = ['episode-isolated', 'cross-episode']
        
        # Optimization settings
        self.optimizations = {
            'mixed_precision': True,  # Use FP16 for faster training
            'gradient_checkpointing': False,  # T4 has enough memory
            'dataloader_workers': 4,  # Parallel data loading
            'pin_memory': True,  # Faster GPU transfer
            'cudnn_benchmark': True  # Optimize convolutions
        }
    
    def generate_command(self, 
                         training_type: str,
                         character_mode: str,
                         output_dir: str) -> List[str]:
        """Generate optimized training command.
        
        Args:
            training_type: 'sequential_cv' or 'parallel_cv'
            character_mode: 'episode-isolated' or 'cross-episode'
            output_dir: Directory for outputs
            
        Returns:
            Command as list of arguments
        """
        config = self.configs[training_type]
        
        cmd = [
            'python', '-u',  # Unbuffered output
            str(self.base_dir / config['script']),
            '--data-dir', 'data/original',
            '--output-dir', output_dir,
            '--character-mode', character_mode,
            '--batch-size', str(config['batch_size']),
            '--epochs', str(config['epochs']),
            '--learning-rate', str(config['learning_rate']),
            '--warmup-steps', str(config['warmup_steps']),
            '--gradient-accumulation-steps', str(config['gradient_accumulation']),
            '--num-workers', str(self.optimizations['dataloader_workers']),
            '--seed', '42'  # For reproducibility
        ]
        
        # Add optimization flags
        if self.optimizations['mixed_precision']:
            cmd.append('--fp16')
        if self.optimizations['pin_memory']:
            cmd.append('--pin-memory')
        if self.optimizations['cudnn_benchmark']:
            cmd.append('--cudnn-benchmark')
        
        # Add GPU device
        cmd.extend(['--gpu', str(self.gpu_device)])
        
        return cmd
    
    def run_single_experiment(self, 
                            training_type: str,
                            character_mode: str) -> Dict:
        """Run a single training experiment.
        
        Args:
            training_type: 'sequential_cv' or 'parallel_cv'
            character_mode: 'episode-isolated' or 'cross-episode'
            
        Returns:
            Experiment results and timing
        """
        experiment_name = f"{training_type}_{character_mode}"
        output_dir = str(self.experiments_dir / experiment_name)
        
        print(f"\n{'='*60}")
        print(f"Starting: {experiment_name}")
        print(f"{'='*60}")
        
        # Generate command
        cmd = self.generate_command(training_type, character_mode, output_dir)
        
        # Set environment for optimal GPU usage
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_device)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Reduce fragmentation
        
        # Run training
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse output for metrics
            metrics = self.parse_training_output(result.stdout)
            
            return {
                'experiment': experiment_name,
                'success': True,
                'elapsed_time': elapsed_time,
                'metrics': metrics,
                'output_dir': output_dir
            }
            
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            print(f"Error in {experiment_name}: {e.stderr}")
            
            return {
                'experiment': experiment_name,
                'success': False,
                'elapsed_time': elapsed_time,
                'error': str(e),
                'output_dir': output_dir
            }
    
    def parse_training_output(self, output: str) -> Dict:
        """Parse training output for key metrics.
        
        Args:
            output: Training script stdout
            
        Returns:
            Extracted metrics
        """
        metrics = {}
        
        # Extract final metrics (customize based on actual output format)
        lines = output.split('\n')
        for line in lines:
            if 'final_accuracy' in line.lower():
                try:
                    metrics['accuracy'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'final_f1' in line.lower():
                try:
                    metrics['f1'] = float(line.split(':')[-1].strip())
                except:
                    pass
        
        return metrics
    
    def run_all_experiments(self, parallel: bool = False) -> List[Dict]:
        """Run all experiment combinations.
        
        Args:
            parallel: If True, run experiments in parallel (requires multiple GPUs)
            
        Returns:
            List of experiment results
        """
        results = []
        total_start = time.time()
        
        experiments = [
            ('sequential_cv', 'episode-isolated'),
            ('sequential_cv', 'cross-episode'),
            ('parallel_cv', 'episode-isolated'),
            ('parallel_cv', 'cross-episode')
        ]
        
        print(f"\nRunning {len(experiments)} experiments")
        print(f"Estimated time: {len(experiments) * 2} hours")
        
        for training_type, character_mode in experiments:
            result = self.run_single_experiment(training_type, character_mode)
            results.append(result)
            
            # Save intermediate results
            self.save_results(results)
            
            print(f"\nCompleted: {result['experiment']}")
            print(f"Time: {result['elapsed_time']/3600:.2f} hours")
            if result['success'] and result.get('metrics'):
                print(f"Metrics: {result['metrics']}")
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*60}")
        print(f"ALL EXPERIMENTS COMPLETE")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Total cost estimate: ${total_time/3600 * 0.526:.2f}")
        print(f"{'='*60}")
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save experiment results to file.
        
        Args:
            results: List of experiment results
        """
        results_file = self.experiments_dir / 'training_results.json'
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'experiments': results,
                'gpu_device': self.gpu_device,
                'optimizations': self.optimizations
            }, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def generate_quick_test_commands(self) -> List[str]:
        """Generate commands for quick testing (1 epoch, small batch).
        
        Returns:
            List of test commands
        """
        test_commands = []
        
        for training_type in ['sequential_cv', 'parallel_cv']:
            for character_mode in self.character_modes:
                config = self.configs[training_type]
                output_dir = f"experiments/test_{training_type}_{character_mode}"
                
                cmd = [
                    'python', '-u',
                    str(self.base_dir / config['script']),
                    '--data-dir', 'data/original',
                    '--output-dir', output_dir,
                    '--character-mode', character_mode,
                    '--batch-size', '8',  # Smaller for testing
                    '--epochs', '1',  # Just 1 epoch
                    '--learning-rate', str(config['learning_rate']),
                    '--warmup-steps', '10',  # Minimal warmup
                    '--gradient-accumulation-steps', '1',
                    '--num-workers', '2',
                    '--seed', '42',
                    '--gpu', str(self.gpu_device)
                ]
                
                if self.optimizations['mixed_precision']:
                    cmd.append('--fp16')
                
                test_commands.append(' '.join(cmd))
        
        return test_commands


def create_launch_script():
    """Create a simple bash script to launch training."""
    script_content = """#!/bin/bash
# Optimized GPU Training Launch Script

echo "CSI Character Embedding Training"
echo "================================"

# Activate virtual environment
source venv/bin/activate

# Start cost tracking in background
python scripts/aws_cost_tracker.py --monitor &
COST_PID=$!

# Run optimized training
python scripts/optimized_training_commands.py --run-all

# Stop cost tracking
kill $COST_PID

# Final cost summary
python scripts/aws_cost_tracker.py --summary

echo "Training complete!"
"""
    
    script_path = Path('scripts/launch_training.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"Launch script created: {script_path}")
    return script_path


def main():
    """Run optimized training from command line."""
    parser = argparse.ArgumentParser(description='Run optimized GPU training')
    parser.add_argument('--run-all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--test', action='store_true',
                       help='Generate test commands (1 epoch)')
    parser.add_argument('--single', type=str,
                       help='Run single experiment (e.g., sequential_cv_episode-isolated)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--create-launch-script', action='store_true',
                       help='Create bash launch script')
    
    args = parser.parse_args()
    
    runner = OptimizedTrainingRunner(gpu_device=args.gpu)
    
    if args.create_launch_script:
        create_launch_script()
        
    elif args.test:
        print("Test Commands (1 epoch each):")
        print("="*60)
        for cmd in runner.generate_quick_test_commands():
            print(f"\n{cmd}\n")
            
    elif args.run_all:
        results = runner.run_all_experiments()
        
        # Print summary
        print("\nFINAL SUMMARY")
        print("="*60)
        for result in results:
            status = "✓" if result['success'] else "✗"
            print(f"{status} {result['experiment']}: "
                  f"{result['elapsed_time']/3600:.2f}h")
            
    elif args.single:
        # Parse single experiment
        parts = args.single.split('_')
        if len(parts) >= 3:
            training_type = '_'.join(parts[:-1])
            character_mode = parts[-1]
            result = runner.run_single_experiment(training_type, character_mode)
            print(f"\nResult: {result}")
        else:
            print(f"Invalid experiment name: {args.single}")
            print("Format: training_type_character-mode")
            print("Example: sequential_cv_episode-isolated")
    
    else:
        # Show available commands
        print("Optimized Training Commands")
        print("="*60)
        
        for training_type in ['sequential_cv', 'parallel_cv']:
            for character_mode in ['episode-isolated', 'cross-episode']:
                output_dir = f"experiments/{training_type}_{character_mode}"
                cmd = runner.generate_command(training_type, character_mode, output_dir)
                
                print(f"\n# {training_type} - {character_mode}")
                print(' '.join(cmd))
        
        print("\n" + "="*60)
        print("To run all experiments: python scripts/optimized_training_commands.py --run-all")
        print("To test (1 epoch): python scripts/optimized_training_commands.py --test")
        print("To create launch script: python scripts/optimized_training_commands.py --create-launch-script")


if __name__ == "__main__":
    main()