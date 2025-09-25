#!/usr/bin/env python3
"""
Interactive Killer Verification Tool
=====================================

Allows scrolling through full episode screenplays to identify killers and reveal boundaries.
"""

import pandas as pd
from pathlib import Path
import json
import os
import sys
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_automated_killers():
    """Load automated killer identifications."""
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

def generate_episode_screenplay(episode_file: Path) -> List[Dict]:
    """
    Generate screenplay representation with sentence numbers.
    
    Returns:
        List of dialogue entries with metadata
    """
    df = pd.read_csv(episode_file, sep='\t')
    
    # Group by sentence ID to reconstruct full sentences
    sentences = []
    current_sent_id = None
    current_speaker = None
    current_words = []
    has_killer_gold = False
    word_killer_mentions = []
    
    for _, row in df.iterrows():
        if row['sentID'] != current_sent_id:
            # Save previous sentence
            if current_words and current_speaker and str(current_speaker) != 'None':
                sentences.append({
                    'sent_id': current_sent_id,
                    'speaker': str(current_speaker),
                    'text': ' '.join(current_words),
                    'has_killer_gold': has_killer_gold,
                    'killer_words': word_killer_mentions
                })
            
            # Start new sentence
            current_sent_id = row['sentID']
            current_speaker = row['speaker']
            current_words = []
            has_killer_gold = False
            word_killer_mentions = []
        
        # Add word to current sentence
        if pd.notna(row['word']) and str(row['word']) != 'None':
            word = str(row['word'])
            current_words.append(word)
            if row['killer_gold'] == 'Y':
                has_killer_gold = True
                word_killer_mentions.append(word)
    
    # Don't forget last sentence
    if current_words and current_speaker and str(current_speaker) != 'None':
        sentences.append({
            'sent_id': current_sent_id,
            'speaker': str(current_speaker),
            'text': ' '.join(current_words),
            'has_killer_gold': has_killer_gold,
            'killer_words': word_killer_mentions
        })
    
    return sentences

def find_high_density_regions(sentences: List[Dict], window_size: int = 20) -> List[Tuple[int, int]]:
    """
    Find regions with high killer mention density.
    
    Returns:
        List of (start_idx, end_idx) tuples for high-density regions
    """
    if not sentences:
        return []
    
    # Count killer mentions in sliding windows
    densities = []
    for i in range(0, len(sentences), window_size // 2):
        window_end = min(i + window_size, len(sentences))
        window = sentences[i:window_end]
        
        mention_count = sum(1 for s in window if s['has_killer_gold'])
        if mention_count > 3:  # Threshold for "high density"
            densities.append((i, window_end, mention_count))
    
    # Merge overlapping regions
    if not densities:
        return []
    
    merged = []
    current_start, current_end, _ = densities[0]
    
    for start, end, _ in densities[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged.append((current_start, current_end))
    return merged

def display_screenplay_page(sentences: List[Dict], 
                           start_idx: int, 
                           page_size: int,
                           high_density_regions: List[Tuple[int, int]],
                           search_term: Optional[str] = None) -> int:
    """
    Display a page of the screenplay with formatting.
    
    Returns:
        Number of sentences displayed
    """
    end_idx = min(start_idx + page_size, len(sentences))
    
    # Check if we're in a high-density region
    in_high_density = False
    for region_start, region_end in high_density_regions:
        if start_idx >= region_start and start_idx < region_end:
            in_high_density = True
            print(f"\n{Colors.YELLOW}{'='*80}")
            print(f"⚠️  HIGH KILLER MENTION DENSITY REGION (likely reveal scene)")
            print(f"{'='*80}{Colors.END}\n")
            break
    
    displayed = 0
    last_speaker = None
    
    for i in range(start_idx, end_idx):
        sent = sentences[i]
        sent_id = sent['sent_id']
        speaker = sent['speaker'].upper().replace('_', ' ')
        text = sent['text']
        
        # Highlight search term if provided
        if search_term and search_term.lower() in speaker.lower():
            speaker = f"{Colors.CYAN}→ {speaker} ←{Colors.END}"
        
        # Highlight killer mentions in text
        if sent['has_killer_gold']:
            for word in sent['killer_words']:
                text = text.replace(word, f"{Colors.RED}*{word}*{Colors.END}")
        
        # Format output
        if speaker != last_speaker:
            print(f"\n{Colors.BOLD}{speaker}:{Colors.END}")
            last_speaker = speaker
        
        # Add sentence number and killer indicator
        killer_mark = f"{Colors.YELLOW}[K]{Colors.END}" if sent['has_killer_gold'] else "   "
        print(f"  {Colors.GREEN}[{sent_id:4d}]{Colors.END} {killer_mark} {text}")
        
        displayed += 1
    
    # Check if exiting high-density region
    if in_high_density:
        for region_start, region_end in high_density_regions:
            if end_idx >= region_end and start_idx < region_end:
                print(f"\n{Colors.YELLOW}{'='*80}")
                print(f"⚠️  END OF HIGH DENSITY REGION")
                print(f"{'='*80}{Colors.END}\n")
                break
    
    return displayed

def interactive_verification(episode_file: Path, auto_killer: Optional[Dict] = None):
    """
    Interactive verification for a single episode.
    
    Returns:
        Dict with verified killer and boundary information
    """
    episode_id = episode_file.stem
    sentences = generate_episode_screenplay(episode_file)
    high_density_regions = find_high_density_regions(sentences)
    
    # Initial display
    clear_screen()
    print(f"{Colors.HEADER}{'='*80}")
    print(f"EPISODE: {episode_id.upper()}")
    print(f"Total sentences: {len(sentences)}")
    if auto_killer:
        print(f"Automated identification: {auto_killer['killer']} (confidence: {auto_killer['confidence']})")
    print(f"High-density regions found: {len(high_density_regions)}")
    print(f"{'='*80}{Colors.END}\n")
    
    # Navigation state
    current_idx = 0
    page_size = 30
    search_term = auto_killer['killer'] if auto_killer else None
    
    # If we have high-density regions, offer to jump to them
    if high_density_regions:
        print(f"{Colors.YELLOW}Found {len(high_density_regions)} potential reveal scene(s):{Colors.END}")
        for i, (start, end) in enumerate(high_density_regions):
            start_sent = sentences[start]['sent_id']
            end_sent = sentences[min(end-1, len(sentences)-1)]['sent_id']
            print(f"  {i+1}. Sentences {start_sent}-{end_sent} ({end-start} lines)")
        print()
    
    # Main interaction loop
    while True:
        # Display current page
        display_screenplay_page(sentences, current_idx, page_size, high_density_regions, search_term)
        
        # Show navigation info
        print(f"\n{Colors.BLUE}{'='*80}")
        print(f"Page: Sentences {sentences[current_idx]['sent_id']} - {sentences[min(current_idx+page_size-1, len(sentences)-1)]['sent_id']} of {sentences[-1]['sent_id']}")
        print(f"Position: {current_idx+1}-{min(current_idx+page_size, len(sentences))} of {len(sentences)} lines")
        print(f"{'='*80}{Colors.END}")
        
        # Show commands
        print("\nCommands:")
        print("  [n]ext page / [p]revious page / [t]op / [b]ottom")
        print("  [j]ump to sentence number / [r]egion <1,2,3...> to jump to high-density region")
        print("  [s]earch for character name / [c]lear search")
        print("  [v]erify killer and boundary / [q]uit without saving")
        print()
        
        command = input("Command: ").strip().lower()
        
        if command in ['n', 'next', '']:
            current_idx = min(current_idx + page_size, len(sentences) - 1)
            clear_screen()
        elif command in ['p', 'prev', 'previous']:
            current_idx = max(0, current_idx - page_size)
            clear_screen()
        elif command in ['t', 'top']:
            current_idx = 0
            clear_screen()
        elif command in ['b', 'bottom']:
            current_idx = max(0, len(sentences) - page_size)
            clear_screen()
        elif command.startswith('j'):
            try:
                target = int(command.split()[1] if len(command.split()) > 1 else input("Jump to sentence: "))
                # Find sentence with this ID
                for i, sent in enumerate(sentences):
                    if sent['sent_id'] >= target:
                        current_idx = max(0, i - 5)  # Show some context before
                        break
                clear_screen()
            except (ValueError, IndexError):
                print("Invalid sentence number")
        elif command.startswith('r'):
            try:
                region_num = int(command.split()[1] if len(command.split()) > 1 else input("Region number: "))
                if 1 <= region_num <= len(high_density_regions):
                    current_idx = high_density_regions[region_num - 1][0]
                    clear_screen()
                else:
                    print(f"Invalid region number (1-{len(high_density_regions)})")
            except (ValueError, IndexError):
                print("Invalid region number")
        elif command.startswith('s'):
            search_term = command[1:].strip() if len(command) > 1 else input("Search for character: ").strip()
            clear_screen()
            print(f"Searching for: {search_term}")
        elif command in ['c', 'clear']:
            search_term = None
            clear_screen()
        elif command in ['v', 'verify']:
            # Verification dialog
            print(f"\n{Colors.GREEN}{'='*60}")
            print("VERIFICATION")
            print(f"{'='*60}{Colors.END}")
            
            if auto_killer:
                print(f"Automated: {auto_killer['killer']}")
            
            # Get killer(s)
            killers_input = input("\nEnter killer(s) [comma-separated, or ENTER to keep automated]: ").strip()
            if not killers_input and auto_killer:
                killers = [auto_killer['killer']]
            elif killers_input:
                killers = [k.strip() for k in killers_input.split(',')]
            else:
                killers = ['UNKNOWN']
            
            # Get reveal boundary
            print("\nEnter sentence number where reveal BEGINS (training should stop BEFORE this)")
            print("Enter -1 if no clear reveal, or ENTER to auto-detect")
            boundary_input = input("Reveal boundary sentence: ").strip()
            
            if boundary_input == '-1':
                boundary = None
            elif boundary_input:
                try:
                    boundary = int(boundary_input)
                except ValueError:
                    boundary = None
            else:
                # Auto-detect based on first high-density region
                if high_density_regions:
                    boundary = sentences[high_density_regions[0][0]]['sent_id']
                    print(f"Auto-detected boundary: {boundary}")
                else:
                    boundary = None
            
            # Get optional notes
            notes = input("Notes (optional): ").strip()
            
            # Confirm
            print(f"\n{Colors.YELLOW}Verification Summary:{Colors.END}")
            print(f"  Killers: {', '.join(killers)}")
            print(f"  Reveal boundary: {boundary if boundary else 'None (no holdout)'}")
            print(f"  Notes: {notes if notes else 'None'}")
            
            confirm = input("\nConfirm? [y/n]: ").strip().lower()
            if confirm == 'y':
                return {
                    'episode': episode_id,
                    'killers': killers,
                    'reveal_boundary': boundary,
                    'total_sentences': len(sentences),
                    'notes': notes,
                    'verified': True
                }
            else:
                clear_screen()
        elif command in ['q', 'quit']:
            confirm = input("Quit without saving? [y/n]: ").strip().lower()
            if confirm == 'y':
                return None
        else:
            print(f"Unknown command: {command}")

def main():
    """Main interactive verification process."""
    data_dir = Path("data/original")
    episodes = sorted(data_dir.glob("s*.tsv"))
    
    # Load automated killers
    auto_killers = load_automated_killers()
    
    # Load existing verifications if any
    verification_file = Path("experiments/verification/killer_labels.json")
    if verification_file.exists():
        with open(verification_file, 'r') as f:
            existing = json.load(f)
            verified_episodes = existing.get('episodes', {})
    else:
        verified_episodes = {}
    
    # Main menu
    while True:
        clear_screen()
        print(f"{Colors.HEADER}{'='*80}")
        print("CSI KILLER VERIFICATION TOOL")
        print(f"{'='*80}{Colors.END}\n")
        
        # Status
        total = len(episodes)
        verified = sum(1 for ep in verified_episodes.values() if ep.get('verified', False))
        print(f"Episodes: {verified}/{total} verified")
        
        # List episodes with status
        print("\nEpisodes:")
        for i, ep_file in enumerate(episodes):
            ep_id = ep_file.stem
            status = "✓" if ep_id in verified_episodes and verified_episodes[ep_id].get('verified') else " "
            auto = auto_killers.get(ep_id, {})
            auto_killer = auto.get('killer', 'UNKNOWN')
            
            # Show verified killer if different
            if ep_id in verified_episodes:
                ver_killers = verified_episodes[ep_id].get('killers', [])
                if ver_killers and ver_killers != [auto_killer]:
                    killer_display = f"{', '.join(ver_killers)} (was: {auto_killer})"
                else:
                    killer_display = auto_killer
            else:
                killer_display = auto_killer
            
            print(f"  [{status}] {i+1:2d}. {ep_id}: {killer_display}")
        
        print(f"\n{Colors.BLUE}Commands:{Colors.END}")
        print("  Enter episode number (1-39) to verify")
        print("  [a]ll - verify all episodes in sequence")
        print("  [u]nverified - verify only unverified episodes")
        print("  [s]ave - save current verifications")
        print("  [e]xport - export to JSON and exit")
        print("  [q]uit - exit without saving")
        print()
        
        command = input("Command: ").strip().lower()
        
        if command.isdigit():
            idx = int(command) - 1
            if 0 <= idx < len(episodes):
                ep_file = episodes[idx]
                ep_id = ep_file.stem
                auto = auto_killers.get(ep_id)
                
                result = interactive_verification(ep_file, auto)
                if result:
                    verified_episodes[ep_id] = result
                    print(f"\n{Colors.GREEN}Verification saved for {ep_id}{Colors.END}")
                    input("Press ENTER to continue...")
        
        elif command == 'a':
            for ep_file in episodes:
                ep_id = ep_file.stem
                auto = auto_killers.get(ep_id)
                
                result = interactive_verification(ep_file, auto)
                if result:
                    verified_episodes[ep_id] = result
                else:
                    # User quit, ask if they want to continue
                    cont = input(f"\nContinue with next episode? [y/n]: ").strip().lower()
                    if cont != 'y':
                        break
        
        elif command == 'u':
            for ep_file in episodes:
                ep_id = ep_file.stem
                if ep_id not in verified_episodes or not verified_episodes[ep_id].get('verified'):
                    auto = auto_killers.get(ep_id)
                    
                    result = interactive_verification(ep_file, auto)
                    if result:
                        verified_episodes[ep_id] = result
                    else:
                        cont = input(f"\nContinue with next episode? [y/n]: ").strip().lower()
                        if cont != 'y':
                            break
        
        elif command in ['s', 'save']:
            # Save current progress
            output_dir = Path("experiments/verification")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output = {
                'version': '1.0',
                'verification_date': pd.Timestamp.now().isoformat(),
                'episodes': verified_episodes
            }
            
            with open(verification_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\n{Colors.GREEN}Verifications saved to {verification_file}{Colors.END}")
            input("Press ENTER to continue...")
        
        elif command in ['e', 'export']:
            # Export and exit
            output_dir = Path("experiments/verification")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Fill in missing episodes with automated values
            for ep_file in episodes:
                ep_id = ep_file.stem
                if ep_id not in verified_episodes:
                    auto = auto_killers.get(ep_id, {})
                    verified_episodes[ep_id] = {
                        'episode': ep_id,
                        'killers': [auto.get('killer', 'UNKNOWN')],
                        'reveal_boundary': None,
                        'total_sentences': None,
                        'notes': 'Using automated identification',
                        'verified': False
                    }
            
            output = {
                'version': '1.0',
                'verification_date': pd.Timestamp.now().isoformat(),
                'episodes': verified_episodes
            }
            
            with open(verification_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\n{Colors.GREEN}Final verifications exported to {verification_file}{Colors.END}")
            
            # Print summary
            verified_count = sum(1 for ep in verified_episodes.values() if ep.get('verified'))
            print(f"\nSummary:")
            print(f"  Total episodes: {len(episodes)}")
            print(f"  Manually verified: {verified_count}")
            print(f"  Using automated: {len(episodes) - verified_count}")
            
            break
        
        elif command in ['q', 'quit']:
            confirm = input("Exit without saving? [y/n]: ").strip().lower()
            if confirm == 'y':
                break

if __name__ == "__main__":
    main()