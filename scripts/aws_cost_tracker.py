#!/usr/bin/env python3
"""
AWS Cost Tracker for GPU Training
==================================

Track and monitor AWS costs during GPU training experiments.
Provides real-time cost estimates and alerts.
"""

import boto3
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import argparse


class AWSCostTracker:
    """Track AWS costs for GPU instances."""
    
    def __init__(self, instance_id: Optional[str] = None):
        """Initialize cost tracker.
        
        Args:
            instance_id: EC2 instance ID to track (optional)
        """
        self.instance_id = instance_id
        self.start_time = None
        self.instance_type = 'g4dn.xlarge'
        self.hourly_rate = 0.526  # On-demand price
        
        # Initialize AWS clients
        try:
            self.ec2_client = boto3.client('ec2')
            self.ce_client = boto3.client('ce')  # Cost Explorer
            self.cloudwatch = boto3.client('cloudwatch')
        except Exception as e:
            print(f"Note: AWS clients not configured. Using offline mode. ({e})")
            self.ec2_client = None
            self.ce_client = None
            self.cloudwatch = None
    
    def start_tracking(self):
        """Start cost tracking timer."""
        self.start_time = datetime.now()
        log_file = Path('experiments/cost_log.json')
        log_file.parent.mkdir(exist_ok=True)
        
        log_entry = {
            'session_start': self.start_time.isoformat(),
            'instance_type': self.instance_type,
            'hourly_rate': self.hourly_rate,
            'instance_id': self.instance_id
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        print(f"Cost tracking started at {self.start_time}")
        print(f"Instance type: {self.instance_type} (${self.hourly_rate}/hour)")
    
    def get_current_cost(self) -> Dict[str, float]:
        """Calculate current session cost."""
        if not self.start_time:
            self.start_time = datetime.now()
        
        elapsed = datetime.now() - self.start_time
        hours = elapsed.total_seconds() / 3600
        current_cost = hours * self.hourly_rate
        
        return {
            'elapsed_hours': round(hours, 2),
            'current_cost': round(current_cost, 2),
            'hourly_rate': self.hourly_rate,
            'projected_daily': round(self.hourly_rate * 24, 2)
        }
    
    def get_aws_costs(self, days: int = 7) -> Optional[Dict]:
        """Get actual AWS costs from Cost Explorer.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Cost breakdown by service
        """
        if not self.ce_client:
            return None
        
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': str(start_date),
                    'End': str(end_date)
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )
            
            costs = {}
            for result in response['ResultsByTime']:
                date = result['TimePeriod']['Start']
                for group in result['Groups']:
                    service = group['Keys'][0]
                    amount = float(group['Metrics']['UnblendedCost']['Amount'])
                    
                    if service not in costs:
                        costs[service] = {}
                    costs[service][date] = round(amount, 2)
            
            return costs
            
        except Exception as e:
            print(f"Could not fetch AWS costs: {e}")
            return None
    
    def check_instance_status(self) -> Optional[str]:
        """Check if instance is running."""
        if not self.ec2_client or not self.instance_id:
            return None
        
        try:
            response = self.ec2_client.describe_instances(
                InstanceIds=[self.instance_id]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    return instance['State']['Name']
            
        except Exception as e:
            print(f"Could not check instance status: {e}")
            return None
    
    def estimate_remaining_cost(self, estimated_hours: float) -> Dict[str, float]:
        """Estimate remaining cost for training.
        
        Args:
            estimated_hours: Estimated hours remaining
            
        Returns:
            Cost estimates
        """
        current = self.get_current_cost()
        remaining_cost = estimated_hours * self.hourly_rate
        total_cost = current['current_cost'] + remaining_cost
        
        return {
            'current_cost': current['current_cost'],
            'estimated_remaining': round(remaining_cost, 2),
            'estimated_total': round(total_cost, 2),
            'estimated_hours_remaining': estimated_hours
        }
    
    def print_summary(self):
        """Print cost tracking summary."""
        print("\n" + "="*60)
        print("AWS COST TRACKING SUMMARY")
        print("="*60)
        
        # Current session
        current = self.get_current_cost()
        print(f"\nCurrent Session:")
        print(f"  Elapsed: {current['elapsed_hours']:.1f} hours")
        print(f"  Cost: ${current['current_cost']:.2f}")
        print(f"  Rate: ${current['hourly_rate']}/hour")
        
        # Instance status
        status = self.check_instance_status()
        if status:
            print(f"  Instance Status: {status}")
            if status != 'running':
                print(f"  ⚠️  Instance not running - no charges accruing")
        
        # Recent AWS costs
        aws_costs = self.get_aws_costs(days=7)
        if aws_costs:
            print(f"\nRecent AWS Costs (7 days):")
            for service, daily_costs in aws_costs.items():
                if 'EC2' in service:
                    total = sum(daily_costs.values())
                    print(f"  {service}: ${total:.2f}")
        
        # Projections
        print(f"\nCost Projections:")
        print(f"  If run 24 hours: ${current['projected_daily']:.2f}")
        print(f"  If run 48 hours: ${current['projected_daily'] * 2:.2f}")
        print(f"  If run 1 week: ${current['projected_daily'] * 7:.2f}")
        
        print("="*60)
    
    def save_checkpoint(self, experiment_name: str, metrics: Dict = None):
        """Save cost checkpoint with experiment progress.
        
        Args:
            experiment_name: Name of current experiment
            metrics: Optional training metrics to log
        """
        checkpoint_file = Path('experiments/cost_checkpoints.json')
        
        # Load existing checkpoints
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoints = json.load(f)
        else:
            checkpoints = []
        
        # Add new checkpoint
        current = self.get_current_cost()
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'experiment': experiment_name,
            'elapsed_hours': current['elapsed_hours'],
            'cost': current['current_cost'],
            'metrics': metrics
        }
        
        checkpoints.append(checkpoint)
        
        # Save
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoints, f, indent=2)
        
        print(f"Checkpoint saved: {experiment_name} - ${current['current_cost']:.2f}")


def monitor_training_cost(check_interval: int = 300):
    """Monitor costs during training.
    
    Args:
        check_interval: Seconds between cost checks
    """
    tracker = AWSCostTracker()
    tracker.start_tracking()
    
    print(f"Starting cost monitoring (checking every {check_interval}s)")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            time.sleep(check_interval)
            tracker.print_summary()
            
            # Alert if costs exceed threshold
            current = tracker.get_current_cost()
            if current['current_cost'] > 50:
                print("⚠️  WARNING: Costs exceed $50!")
            
    except KeyboardInterrupt:
        print("\nStopping cost monitor")
        tracker.print_summary()
        
        # Save final summary
        final_log = Path('experiments/cost_final.json')
        with open(final_log, 'w') as f:
            json.dump(tracker.get_current_cost(), f, indent=2)
        
        print(f"Final cost data saved to {final_log}")


def main():
    """Run cost tracker from command line."""
    parser = argparse.ArgumentParser(description='Track AWS GPU training costs')
    parser.add_argument('--monitor', action='store_true',
                       help='Start continuous monitoring')
    parser.add_argument('--summary', action='store_true',
                       help='Print current cost summary')
    parser.add_argument('--instance-id', type=str,
                       help='EC2 instance ID to track')
    parser.add_argument('--estimate', type=float,
                       help='Estimate cost for N hours of training')
    parser.add_argument('--checkpoint', type=str,
                       help='Save checkpoint with experiment name')
    
    args = parser.parse_args()
    
    tracker = AWSCostTracker(instance_id=args.instance_id)
    
    if args.monitor:
        monitor_training_cost()
    elif args.summary:
        tracker.start_tracking()
        tracker.print_summary()
    elif args.estimate:
        estimate = tracker.estimate_remaining_cost(args.estimate)
        print(f"Estimated cost for {args.estimate} hours: ${estimate['estimated_remaining']:.2f}")
    elif args.checkpoint:
        tracker.save_checkpoint(args.checkpoint)
    else:
        # Default: show summary
        tracker.start_tracking()
        tracker.print_summary()


if __name__ == "__main__":
    main()