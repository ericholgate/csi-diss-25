#!/usr/bin/env python3
"""
Extract Screenplay-Like Scripts from CSI Episodes
==================================================

Focuses on the final scenes where killer reveals typically occur.
Allows for human verification of killer identification.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import sys
import json

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

def format_timestamp(time_str):
    """Convert timestamp to readable format."""
    if pd.isna(time_str) or time_str == 'None':
        return ""
    
    try:
        # Format: HH:MM:SS.mmm -> MM:SS
        parts = str(time_str).split(':')
        if len(parts) >= 3:
            minutes = parts[1]
            seconds = parts[2].split('.')[0]
            return f"[{minutes}:{seconds}]"
    except:
        pass
    return ""

def create_screenplay(episode_file: Path, 
                     percentage: float = 0.2,
                     focus_killer: Optional[str] = None) -> str:
    """
    Create a screenplay-like representation of an episode.
    
    Args:
        episode_file: Path to TSV file
        percentage: Last X% of episode to extract (0.2 = last 20%)
        focus_killer: If provided, highlight this character's dialogue
        
    Returns:
        Formatted screenplay text
    """
    df = pd.read_csv(episode_file, sep='\t')
    episode_id = episode_file.stem
    
    # Group by sentence ID to reconstruct full sentences
    sentences = []
    current_sent_id = None
    current_speaker = None
    current_words = []
    current_time = None
    has_killer_gold = False
    
    for _, row in df.iterrows():
        if row['sentID'] != current_sent_id:
            # Save previous sentence
            if current_words and current_speaker and current_speaker != 'None':
                sentences.append({
                    'sent_id': current_sent_id,
                    'speaker': current_speaker,
                    'text': ' '.join(current_words),
                    'time': current_time,
                    'has_killer_gold': has_killer_gold
                })
            
            # Start new sentence
            current_sent_id = row['sentID']
            current_speaker = row['speaker']
            current_words = []
            current_time = row['medion_time']
            has_killer_gold = False
        
        # Add word to current sentence
        if pd.notna(row['word']) and row['word'] != 'None':
            current_words.append(str(row['word']))
            if row['killer_gold'] == 'Y':
                has_killer_gold = True
    
    # Don't forget last sentence
    if current_words and current_speaker and current_speaker != 'None':
        sentences.append({
            'sent_id': current_sent_id,
            'speaker': current_speaker,
            'text': ' '.join(current_words),
            'time': current_time,
            'has_killer_gold': has_killer_gold
        })
    
    # Extract last X% of episode
    total_sentences = len(sentences)
    start_idx = int(total_sentences * (1 - percentage))
    final_scenes = sentences[start_idx:]
    
    # Create screenplay format
    screenplay = []
    screenplay.append(f"{'='*80}")
    screenplay.append(f"EPISODE: {episode_id.upper()}")
    screenplay.append(f"FINAL SCENES (Last {int(percentage*100)}% of episode)")
    screenplay.append(f"Total sentences: {len(final_scenes)} of {total_sentences}")
    
    if focus_killer:
        screenplay.append(f"SUSPECTED KILLER: {focus_killer.upper()}")
    
    screenplay.append(f"{'='*80}\n")
    
    # Format dialogue
    last_speaker = None
    for sent in final_scenes:
        speaker = sent['speaker'].upper()
        
        # Clean up speaker name
        speaker = speaker.replace('_', ' ').strip()
        
        # Add timestamp if available
        timestamp = format_timestamp(sent['time'])
        
        # Highlight killer's dialogue
        if focus_killer and focus_killer.lower() in sent['speaker'].lower():
            speaker = f"→ {speaker} ←"  # Mark suspected killer
        
        # Mark lines that mention the killer
        killer_marker = " [K]" if sent['has_killer_gold'] else ""
        
        # Format based on speaker change
        if speaker != last_speaker:
            screenplay.append(f"\n{speaker}:")
            last_speaker = speaker
        
        screenplay.append(f"  {timestamp} {sent['text']}{killer_marker}")
    
    return '\n'.join(screenplay)

def extract_reveal_scene(episode_file: Path, 
                        focus_killer: Optional[str] = None,
                        window_size: int = 50) -> str:
    """
    Extract the likely reveal scene by finding where killer is mentioned most.
    
    Args:
        episode_file: Path to TSV file
        focus_killer: The suspected killer
        window_size: Number of sentences to include in scene
    """
    df = pd.read_csv(episode_file, sep='\t')
    
    # Find sentences with killer_gold = Y
    killer_mentions = df[df['killer_gold'] == 'Y'].copy()
    
    if len(killer_mentions) == 0:
        return "No killer mentions found in episode"
    
    # Find the densest cluster of killer mentions (likely the reveal)
    killer_mentions['sent_group'] = killer_mentions['sentID'] // 10  # Group by 10 sentences
    mention_density = killer_mentions.groupby('sent_group').size()
    
    if len(mention_density) > 0:
        peak_group = mention_density.idxmax()
        peak_sent_id = peak_group * 10
        
        # Extract window around peak
        start_sent = max(0, peak_sent_id - window_size // 2)
        end_sent = peak_sent_id + window_size // 2
        
        # Build screenplay for this specific scene
        scene_df = df[(df['sentID'] >= start_sent) & (df['sentID'] <= end_sent)]
        
        # Create screenplay for just this scene
        screenplay = []
        screenplay.append(f"{'='*80}")
        screenplay.append(f"LIKELY REVEAL SCENE (Sentences {start_sent}-{end_sent})")
        screenplay.append(f"Peak killer mentions at sentence ~{peak_sent_id}")
        screenplay.append(f"{'='*80}\n")
        
        current_speaker = None
        current_words = []
        
        for _, row in scene_df.iterrows():
            speaker = row['speaker']
            # Handle NaN/None speakers
            if pd.isna(speaker) or speaker == 'None':
                speaker = None
            else:
                speaker = str(speaker)
            
            if speaker != current_speaker:
                if current_words and current_speaker and current_speaker != 'None':
                    speaker_display = str(current_speaker).upper().replace('_', ' ')
                    if focus_killer and focus_killer.lower() in str(current_speaker).lower():
                        speaker_display = f"→ {speaker_display} ←"
                    screenplay.append(f"\n{speaker_display}:")
                    screenplay.append(f"  {' '.join(current_words)}")
                
                current_speaker = speaker
                current_words = []
            
            if pd.notna(row['word']) and row['word'] != 'None':
                word = str(row['word'])
                if row['killer_gold'] == 'Y':
                    word = f"**{word}**"  # Highlight killer mentions
                current_words.append(word)
        
        # Don't forget last speaker
        if current_words and current_speaker and current_speaker != 'None':
            speaker_display = str(current_speaker).upper().replace('_', ' ')
            if focus_killer and focus_killer.lower() in str(current_speaker).lower():
                speaker_display = f"→ {speaker_display} ←"
            screenplay.append(f"\n{speaker_display}:")
            screenplay.append(f"  {' '.join(current_words)}")
        
        return '\n'.join(screenplay)
    
    return "Could not identify reveal scene"

def main():
    """Generate screenplay extracts for killer verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract CSI episode screenplays for killer verification")
    parser.add_argument('--episode', type=str, help='Specific episode (e.g., s01e07)')
    parser.add_argument('--all', action='store_true', help='Process all episodes')
    parser.add_argument('--percentage', type=float, default=0.2, 
                       help='Percentage of episode to extract (default: 0.2 = last 20%)')
    parser.add_argument('--reveal-only', action='store_true',
                       help='Extract only the likely reveal scene')
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/screenplays'),
                       help='Output directory for screenplay files')
    
    args = parser.parse_args()
    
    # Load killer identifications
    killers = load_killer_identifications()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.episode:
        # Process single episode
        episode_file = Path(f"data/original/{args.episode}.tsv")
        if not episode_file.exists():
            print(f"Episode file not found: {episode_file}")
            return
        
        killer_info = killers.get(args.episode, {})
        killer_name = killer_info.get('killer', None)
        
        if args.reveal_only:
            screenplay = extract_reveal_scene(episode_file, killer_name)
        else:
            screenplay = create_screenplay(episode_file, args.percentage, killer_name)
        
        # Print to console
        print(screenplay)
        
        # Save to file
        output_file = args.output_dir / f"{args.episode}_screenplay.txt"
        with open(output_file, 'w') as f:
            f.write(screenplay)
        print(f"\nScreenplay saved to {output_file}")
        
    elif args.all:
        # Process all episodes
        data_dir = Path("data/original")
        episodes = sorted(data_dir.glob("s*.tsv"))
        
        # Create verification file
        verification_file = args.output_dir / "killer_verification.txt"
        
        with open(verification_file, 'w') as vf:
            vf.write("KILLER VERIFICATION WORKSHEET\n")
            vf.write("="*80 + "\n")
            vf.write("Instructions: Review each screenplay excerpt and verify the killer.\n")
            vf.write("Mark: ✓ (correct), ✗ (incorrect), ? (unclear)\n")
            vf.write("="*80 + "\n\n")
            
            for episode_file in episodes:
                episode_id = episode_file.stem
                killer_info = killers.get(episode_id, {})
                killer_name = killer_info.get('killer', 'UNKNOWN')
                confidence = killer_info.get('confidence', 'none')
                
                print(f"Processing {episode_id}...")
                
                # Extract screenplay
                if args.reveal_only:
                    screenplay = extract_reveal_scene(episode_file, killer_name)
                else:
                    screenplay = create_screenplay(episode_file, args.percentage, killer_name)
                
                # Save individual screenplay
                output_file = args.output_dir / f"{episode_id}_screenplay.txt"
                with open(output_file, 'w') as f:
                    f.write(screenplay)
                
                # Add to verification worksheet
                vf.write(f"\n{'='*80}\n")
                vf.write(f"Episode: {episode_id}\n")
                vf.write(f"Identified Killer: {killer_name} (confidence: {confidence})\n")
                vf.write(f"Verification: [ ] ✓ Correct  [ ] ✗ Incorrect  [ ] ? Unclear\n")
                vf.write(f"Correct Killer (if different): _______________________\n")
                vf.write(f"Notes: _____________________________________________\n")
                vf.write(f"See: {output_file.name}\n")
        
        print(f"\nAll screenplays saved to {args.output_dir}")
        print(f"Verification worksheet: {verification_file}")
    
    else:
        print("Please specify --episode or --all")
        print("\nExamples:")
        print("  python scripts/extract_episode_screenplay.py --episode s01e07")
        print("  python scripts/extract_episode_screenplay.py --episode s01e07 --reveal-only")
        print("  python scripts/extract_episode_screenplay.py --all --percentage 0.15")

if __name__ == "__main__":
    main()