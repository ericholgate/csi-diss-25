#!/usr/bin/env python3
"""
Process Killer Verification Results
====================================

Converts the verified CSV into a JSON configuration for dataset creation.
"""

import pandas as pd
import json
from pathlib import Path

def process_verification(csv_file: Path = Path("experiments/verification/killer_verification_template.csv")):
    """Process verified results into JSON configuration."""
    
    if not csv_file.exists():
        print(f"Verification file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # Create configuration
    config = {
        'version': '1.0',
        'verification_date': pd.Timestamp.now().isoformat(),
        'episodes': {}
    }
    
    for _, row in df.iterrows():
        episode = row['episode']
        
        # Parse killers (semicolon-separated)
        verified_killers = str(row.get('verified_killers', '')).strip()
        if not verified_killers or verified_killers == 'nan':
            # Use automated if not verified
            verified_killers = row['auto_killer']
        
        # Split multiple killers
        if ';' in verified_killers:
            killer_list = [k.strip() for k in verified_killers.split(';')]
        else:
            killer_list = [verified_killers.strip()]
        
        # Parse boundary
        verified_boundary = row.get('verified_boundary', '')
        if pd.notna(verified_boundary) and str(verified_boundary).strip():
            try:
                boundary = int(verified_boundary)
            except:
                boundary = row['suggested_boundary']
        else:
            boundary = row['suggested_boundary']
        
        # Handle -1 as "no holdout needed"
        if boundary == -1:
            boundary = None
        
        config['episodes'][episode] = {
            'killers': killer_list,
            'reveal_boundary': boundary,
            'total_sentences': int(row['total_sentences']) if pd.notna(row['total_sentences']) else None,
            'auto_killer': row['auto_killer'],
            'auto_confidence': row['auto_confidence'],
            'verified': verified_killers != row['auto_killer'],
            'notes': str(row.get('notes', '')) if pd.notna(row.get('notes', '')) else ''
        }
    
    # Save configuration
    output_file = Path("experiments/verification/killer_labels.json")
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Print summary
    print("Verification Processing Complete")
    print("=" * 60)
    print(f"Total episodes: {len(config['episodes'])}")
    
    # Count verifications
    verified = sum(1 for ep in config['episodes'].values() if ep['verified'])
    print(f"Manually verified: {verified}")
    print(f"Using automated: {len(config['episodes']) - verified}")
    
    # Count multiple killers
    multi_killer = sum(1 for ep in config['episodes'].values() if len(ep['killers']) > 1)
    print(f"Episodes with multiple killers: {multi_killer}")
    
    # Holdout statistics
    with_holdout = sum(1 for ep in config['episodes'].values() if ep['reveal_boundary'] is not None)
    print(f"Episodes with reveal boundary: {with_holdout}")
    
    if with_holdout > 0:
        avg_boundary = sum(ep['reveal_boundary'] for ep in config['episodes'].values() 
                          if ep['reveal_boundary'] is not None) / with_holdout
        print(f"Average reveal boundary: {avg_boundary:.0f} sentences")
    
    print(f"\nConfiguration saved to: {output_file}")
    
    # Example of how to use in dataset creation
    print("\nUsage in dataset creation:")
    print("-" * 40)
    print("from pathlib import Path")
    print("import json")
    print("")
    print("# Load killer labels")
    print("with open('experiments/verification/killer_labels.json', 'r') as f:")
    print("    killer_config = json.load(f)")
    print("")
    print("# Get killers for an episode")
    print("episode_id = 's01e07'")
    print("killers = killer_config['episodes'][episode_id]['killers']")
    print("boundary = killer_config['episodes'][episode_id]['reveal_boundary']")
    print("")
    print("# Filter sentences for training")
    print("if boundary is not None:")
    print("    training_sentences = [s for s in sentences if s.sent_id < boundary]")
    
    return output_file

if __name__ == "__main__":
    process_verification()
