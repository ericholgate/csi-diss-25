#!/usr/bin/env python3
"""
Analyze killer_gold labels to identify actual killers (Version 2)
==================================================================

Better strategy: Look for character names (not pronouns) that are 
frequently mentioned with killer_gold = Y.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Known investigators to exclude
INVESTIGATORS = {
    'grissom', 'catherine', 'nick', 'warrick', 'sara', 'brass', 
    'detective', 'det', 'officer', 'greg', 'hodges', 'archie',
    'david', 'doc', 'robbins', 'ecklie', 'sofia', 'henry'
}

# Common pronouns and non-name words to filter
NON_NAMES = {
    'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
    'they', 'them', 'their', 'we', 'us', 'our', 'i', 'me', 'my',
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'that', 'this',
    'guy', 'man', 'woman', 'person', 'people', 'someone', 'anyone',
    'gunman', 'killer', 'suspect', 'victim', 'hostess', 'driver', 'shooter',
    'brother', 'sister', 'father', 'mother', 'family', 'wife', 'husband',
    'intruder', 'manager', 'mrs', 'mr', 'miss', 'ms'
}

def is_likely_name(word):
    """Check if a word is likely a character name."""
    word = word.lower().strip()
    
    # Filter out non-names
    if word in NON_NAMES or word in INVESTIGATORS:
        return False
    
    # Filter out very short words
    if len(word) < 3:
        return False
    
    # Filter out words that are all lowercase common words
    if word in ['was', 'were', 'been', 'being', 'have', 'has', 'had', 'having']:
        return False
    
    return True

def analyze_episode_v2(filepath):
    """Analyze a single episode to find the killer using improved strategy."""
    df = pd.read_csv(filepath, sep='\t')
    episode_id = Path(filepath).stem
    
    # Find all rows where killer_gold = Y
    killer_refs = df[df['killer_gold'] == 'Y']
    
    if len(killer_refs) == 0:
        return None, "No killer_gold=Y found", {}
    
    # Collect all potential killer names
    name_mentions = Counter()
    speaker_mentions = Counter()
    
    for _, row in killer_refs.iterrows():
        word = str(row['word']).lower().strip()
        speaker = str(row['speaker']).lower().strip()
        
        # Count word mentions if it's likely a name
        if is_likely_name(word):
            name_mentions[word] += 1
        
        # Track who mentions the killer
        if speaker != 'none' and speaker not in INVESTIGATORS:
            speaker_mentions[speaker] += 1
    
    # Strategy 1: Find self-references (speaker mentions themselves)
    self_refs = {}
    for _, row in killer_refs.iterrows():
        speaker = str(row['speaker']).lower().strip()
        word = str(row['word']).lower().strip()
        
        if speaker != 'none' and is_likely_name(speaker):
            # Check for self-reference
            if speaker == word or word in speaker or speaker in word:
                if len(speaker) > 3:  # Avoid short matches
                    self_refs[speaker] = self_refs.get(speaker, 0) + 1
    
    # Strategy 2: Find the most mentioned character name
    if name_mentions:
        top_names = name_mentions.most_common(5)
        
        # Filter to get the best candidate
        best_candidate = None
        for name, count in top_names:
            # Check if this is a full name (contains first+last)
            if len(name) > 6 and count >= 3:
                best_candidate = name
                break
        
        if not best_candidate and top_names:
            # Take the most mentioned name
            best_candidate = top_names[0][0]
    else:
        best_candidate = None
    
    # Determine the killer
    if self_refs:
        # Prefer self-reference if found
        killer = max(self_refs.items(), key=lambda x: x[1])[0]
        confidence = "high" if self_refs[killer] >= 2 else "medium"
        method = "self-reference"
    elif best_candidate:
        killer = best_candidate
        confidence = "medium" if name_mentions[killer] >= 5 else "low"
        method = "name-frequency"
    else:
        # Last resort: speaker who mentions killer most
        if speaker_mentions:
            non_inv_speakers = {k: v for k, v in speaker_mentions.items() 
                              if k not in INVESTIGATORS and is_likely_name(k)}
            if non_inv_speakers:
                killer = max(non_inv_speakers.items(), key=lambda x: x[1])[0]
                confidence = "low"
                method = "speaker-frequency"
            else:
                killer = None
                confidence = "none"
                method = "not-found"
        else:
            killer = None
            confidence = "none"
            method = "not-found"
    
    debug_info = {
        'self_refs': dict(self_refs) if self_refs else {},
        'top_names': dict(name_mentions.most_common(5)) if name_mentions else {},
        'method': method,
        'confidence': confidence
    }
    
    return killer, method, debug_info

def main():
    """Analyze all episodes to identify killers."""
    data_dir = Path("data/original")
    episodes = sorted(data_dir.glob("s*.tsv"))
    
    print("KILLER IDENTIFICATION ANALYSIS V2")
    print("=" * 80)
    print("\nStrategy: Identify character names mentioned with killer_gold = Y")
    print("-" * 80)
    
    results = {}
    identified_killers = {}
    unresolved_episodes = []
    
    for episode_file in episodes:
        episode_id = episode_file.stem
        killer, method, debug_info = analyze_episode_v2(episode_file)
        
        if killer:
            identified_killers[episode_id] = (killer, method, debug_info['confidence'])
            
            # Format output based on confidence
            if debug_info['confidence'] == 'high':
                symbol = "✓✓"
            elif debug_info['confidence'] == 'medium':
                symbol = "✓"
            else:
                symbol = "?"
            
            print(f"\n{episode_id}: {symbol} {killer} ({method}, confidence: {debug_info['confidence']})")
            
            # Show debug info for transparency
            if debug_info['self_refs']:
                print(f"  Self-refs: {debug_info['self_refs']}")
            if debug_info['top_names']:
                print(f"  Top names: {list(debug_info['top_names'].keys())[:3]}")
        else:
            unresolved_episodes.append(episode_id)
            print(f"\n{episode_id}: ❌ Could not identify killer")
            if debug_info['top_names']:
                print(f"  Top names mentioned: {list(debug_info['top_names'].keys())[:3]}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal episodes: {len(episodes)}")
    print(f"Episodes with identified killer: {len(identified_killers)}")
    print(f"  - High confidence: {sum(1 for _, (_, _, conf) in identified_killers.items() if conf == 'high')}")
    print(f"  - Medium confidence: {sum(1 for _, (_, _, conf) in identified_killers.items() if conf == 'medium')}")
    print(f"  - Low confidence: {sum(1 for _, (_, _, conf) in identified_killers.items() if conf == 'low')}")
    print(f"Unresolved episodes: {len(unresolved_episodes)}")
    
    # Check for duplicates (multiple killers per episode shouldn't happen)
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    # Each episode should have exactly one killer
    print(f"✓ One killer per episode: {len(identified_killers) == len(set(identified_killers.keys()))}")
    
    # No investigators should be killers
    investigator_killers = []
    for ep, (killer, _, _) in identified_killers.items():
        if any(inv in killer for inv in INVESTIGATORS):
            investigator_killers.append((ep, killer))
    
    if investigator_killers:
        print(f"⚠️  WARNING: Investigators as killers: {investigator_killers}")
    else:
        print("✓ No main investigators identified as killers")
    
    # List all identified killers
    print("\n" + "=" * 80)
    print("IDENTIFIED KILLERS")
    print("=" * 80)
    
    for ep in sorted(identified_killers.keys()):
        killer, method, confidence = identified_killers[ep]
        print(f"{ep}: {killer:20} ({method:15} confidence: {confidence})")
    
    if unresolved_episodes:
        print("\nUnresolved episodes:")
        for ep in unresolved_episodes:
            print(f"  {ep}")
    
    # Save results
    output_file = Path("experiments/killer_identification_v2.txt")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Episode,Killer,Method,Confidence\n")
        for ep, (killer, method, confidence) in identified_killers.items():
            f.write(f"{ep},{killer},{method},{confidence}\n")
        for ep in unresolved_episodes:
            f.write(f"{ep},UNKNOWN,not-found,none\n")
    
    print(f"\nResults saved to {output_file}")
    
    return identified_killers, unresolved_episodes

if __name__ == "__main__":
    main()