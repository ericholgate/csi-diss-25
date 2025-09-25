#!/usr/bin/env python3
"""
Generate Comprehensive Killer Verification Worksheet
====================================================

Creates a detailed worksheet for manual verification of killers and reveal boundaries.
Includes support for multiple killers and sentence number boundaries for holdout.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import Counter

def load_killer_identifications():
    """Load our automated killer identifications."""
    killer_file = Path("experiments/killer_identification_v2.txt")
    killers = {}
    
    if killer_file.exists():
        with open(killer_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    episode, killer, method, confidence = parts
                    if killer != 'UNKNOWN':
                        killers[episode] = {
                            'killer': killer,
                            'method': method,
                            'confidence': confidence
                        }
    return killers

def find_reveal_boundaries(episode_file: Path) -> Dict:
    """
    Find likely reveal boundaries in an episode.
    
    Returns:
        Dict with reveal statistics and sentence boundaries
    """
    df = pd.read_csv(episode_file, sep='\t')
    
    # Find sentences with killer_gold = Y
    killer_mentions = df[df['killer_gold'] == 'Y'].copy()
    
    if len(killer_mentions) == 0:
        return {
            'first_mention': None,
            'last_mention': None,
            'peak_mention': None,
            'total_mentions': 0,
            'density_peak': None,
            'suggested_boundary': None
        }
    
    # Get sentence IDs with killer mentions
    mention_sents = killer_mentions['sentID'].unique()
    
    # Find first and last mentions
    first_mention = int(mention_sents.min())
    last_mention = int(mention_sents.max())
    
    # Find density peak (likely reveal location)
    # Group by windows of 20 sentences
    killer_mentions['sent_window'] = killer_mentions['sentID'] // 20
    mention_density = killer_mentions.groupby('sent_window').size()
    
    if len(mention_density) > 0:
        peak_window = mention_density.idxmax()
        peak_sent = peak_window * 20
        
        # Find the actual start of high-density mentions in that window
        window_mentions = killer_mentions[killer_mentions['sent_window'] == peak_window]
        reveal_start = int(window_mentions['sentID'].min())
    else:
        peak_sent = first_mention
        reveal_start = first_mention
    
    # Suggested boundary: Start of reveal scene minus buffer
    # We want to stop training BEFORE the reveal
    buffer = 50  # sentences before reveal
    suggested_boundary = max(0, reveal_start - buffer)
    
    # Get total sentences in episode
    total_sentences = df['sentID'].max()
    
    return {
        'first_mention': first_mention,
        'last_mention': last_mention,
        'peak_mention': peak_sent,
        'total_mentions': len(killer_mentions),
        'density_peak': reveal_start,
        'suggested_boundary': suggested_boundary,
        'total_sentences': total_sentences,
        'reveal_percentage': (reveal_start / total_sentences * 100) if total_sentences > 0 else 0
    }

def extract_reveal_context(episode_file: Path, 
                          reveal_sent: int,
                          window: int = 30) -> List[str]:
    """
    Extract dialogue context around reveal point.
    
    Returns:
        List of formatted dialogue lines
    """
    df = pd.read_csv(episode_file, sep='\t')
    
    # Get sentences around reveal
    start = max(0, reveal_sent - window // 2)
    end = reveal_sent + window // 2
    
    scene_df = df[(df['sentID'] >= start) & (df['sentID'] <= end)]
    
    # Group by sentence to reconstruct dialogue
    dialogue = []
    current_sent = None
    current_speaker = None
    current_words = []
    
    for _, row in scene_df.iterrows():
        if row['sentID'] != current_sent:
            # Save previous sentence
            if current_words and current_speaker and str(current_speaker) != 'None':
                speaker = str(current_speaker).upper().replace('_', ' ')
                text = ' '.join(current_words)
                dialogue.append(f"  [{current_sent:4d}] {speaker}: {text}")
            
            # Start new sentence
            current_sent = row['sentID']
            current_speaker = row['speaker']
            current_words = []
        
        # Add word
        if pd.notna(row['word']) and str(row['word']) != 'None':
            word = str(row['word'])
            if row['killer_gold'] == 'Y':
                word = f"*{word}*"  # Mark killer mentions
            current_words.append(word)
    
    # Don't forget last sentence
    if current_words and current_speaker and str(current_speaker) != 'None':
        speaker = str(current_speaker).upper().replace('_', ' ')
        text = ' '.join(current_words)
        dialogue.append(f"  [{current_sent:4d}] {speaker}: {text}")
    
    return dialogue

def generate_worksheet():
    """Generate comprehensive verification worksheet."""
    
    data_dir = Path("data/original")
    episodes = sorted(data_dir.glob("s*.tsv"))
    
    # Load automated identifications
    auto_killers = load_killer_identifications()
    
    # Create output directory
    output_dir = Path("experiments/verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Main worksheet file
    worksheet_file = output_dir / "killer_verification_worksheet.txt"
    
    # Also create a template CSV for easier editing
    csv_data = []
    
    with open(worksheet_file, 'w') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("CSI KILLER VERIFICATION WORKSHEET\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("INSTRUCTIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Review the reveal scene context for each episode\n")
        f.write("2. Verify or correct the identified killer(s)\n")
        f.write("3. For multiple killers, separate with semicolons (e.g., 'tinacollins;jesseoverton')\n")
        f.write("4. Specify the sentence boundary BEFORE the reveal (for training holdout)\n")
        f.write("5. The suggested boundary is typically 50 sentences before the reveal peak\n")
        f.write("6. Use 'NONE' if no killer can be identified\n")
        f.write("7. Use sentence number -1 if no holdout is needed (no clear reveal)\n\n")
        
        f.write("SENTENCE NUMBER GUIDE:\n")
        f.write("-" * 40 + "\n")
        f.write("- First mention: First sentence where killer is referenced\n")
        f.write("- Peak density: Where killer mentions are most concentrated (likely reveal)\n")
        f.write("- Suggested boundary: Recommended cutoff for training (before reveal)\n")
        f.write("- Total sentences: Total number of sentences in episode\n\n")
        
        f.write("=" * 100 + "\n\n")
        
        # Process each episode
        for episode_file in episodes:
            episode_id = episode_file.stem
            
            # Get automated identification
            auto_info = auto_killers.get(episode_id, {})
            auto_killer = auto_info.get('killer', 'UNKNOWN')
            auto_confidence = auto_info.get('confidence', 'none')
            
            # Find reveal boundaries
            boundaries = find_reveal_boundaries(episode_file)
            
            # Episode header
            f.write("=" * 100 + "\n")
            f.write(f"EPISODE: {episode_id.upper()}\n")
            f.write("=" * 100 + "\n\n")
            
            # Automated identification
            f.write("AUTOMATED IDENTIFICATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Killer: {auto_killer}\n")
            f.write(f"  Confidence: {auto_confidence}\n")
            f.write(f"  Method: {auto_info.get('method', 'unknown')}\n\n")
            
            # Reveal statistics
            f.write("REVEAL STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total sentences: {boundaries['total_sentences']}\n")
            f.write(f"  First killer mention: Sentence {boundaries['first_mention']}\n")
            f.write(f"  Last killer mention: Sentence {boundaries['last_mention']}\n")
            f.write(f"  Peak density: Sentence {boundaries['density_peak']}\n")
            f.write(f"  Reveal at: ~{boundaries['reveal_percentage']:.1f}% through episode\n")
            f.write(f"  Suggested training boundary: Sentence {boundaries['suggested_boundary']}\n")
            f.write(f"  Total killer mentions: {boundaries['total_mentions']}\n\n")
            
            # Extract reveal context if available
            if boundaries['density_peak']:
                f.write("REVEAL SCENE CONTEXT:\n")
                f.write("-" * 40 + "\n")
                dialogue = extract_reveal_context(episode_file, boundaries['density_peak'])
                
                # Show first 20 lines of context
                for line in dialogue[:20]:
                    f.write(line + "\n")
                
                if len(dialogue) > 20:
                    f.write(f"  ... ({len(dialogue) - 20} more lines)\n")
                
                f.write("\n")
            
            # Verification section
            f.write("MANUAL VERIFICATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Correct Killer(s): _________________________ (Current: {auto_killer})\n")
            f.write(f"Training Boundary: _________________________ (Suggested: {boundaries['suggested_boundary']})\n")
            f.write("Notes: _________________________________________________________________\n")
            f.write("       _________________________________________________________________\n")
            f.write("\n\n")
            
            # Add to CSV data
            csv_data.append({
                'episode': episode_id,
                'auto_killer': auto_killer,
                'auto_confidence': auto_confidence,
                'suggested_boundary': boundaries['suggested_boundary'],
                'first_mention': boundaries['first_mention'],
                'peak_density': boundaries['density_peak'],
                'total_sentences': boundaries['total_sentences'],
                'verified_killers': '',  # To be filled manually
                'verified_boundary': '',  # To be filled manually
                'notes': ''  # To be filled manually
            })
        
        # Footer with summary
        f.write("=" * 100 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total episodes: {len(episodes)}\n")
        f.write(f"High confidence: {sum(1 for k in auto_killers.values() if k['confidence'] == 'high')}\n")
        f.write(f"Medium confidence: {sum(1 for k in auto_killers.values() if k['confidence'] == 'medium')}\n")
        f.write(f"Low confidence: {sum(1 for k in auto_killers.values() if k['confidence'] == 'low')}\n")
        f.write("\nPlease save your verifications in the CSV file for processing.\n")
    
    # Save CSV template
    csv_file = output_dir / "killer_verification_template.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    print(f"Worksheet generated: {worksheet_file}")
    print(f"CSV template generated: {csv_file}")
    print("\nTo complete verification:")
    print("1. Review the worksheet: experiments/verification/killer_verification_worksheet.txt")
    print("2. Fill in the CSV: experiments/verification/killer_verification_template.csv")
    print("3. Run: python scripts/process_verification.py")
    
    return worksheet_file, csv_file

def main():
    """Generate the verification worksheet."""
    worksheet_file, csv_file = generate_worksheet()
    
    # Create the processing script
    create_processing_script()
    
    print("\nWorksheet generation complete!")

def create_processing_script():
    """Create script to process verified results into JSON."""
    
    script_content = '''#!/usr/bin/env python3
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
    
    print(f"\\nConfiguration saved to: {output_file}")
    
    # Example of how to use in dataset creation
    print("\\nUsage in dataset creation:")
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
'''
    
    script_file = Path("scripts/process_verification.py")
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    script_file.chmod(0o755)
    print(f"Processing script created: {script_file}")

if __name__ == "__main__":
    main()