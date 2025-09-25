#!/usr/bin/env python3
"""
Analyze killer_gold labels to identify actual killers
======================================================

Test the hypothesis that killer_gold marks references to the killer,
and killers can be identified when they reference themselves.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import sys

def analyze_episode(filepath):
    """Analyze a single episode to find the killer."""
    df = pd.read_csv(filepath, sep='\t')
    
    # Find all rows where killer_gold = Y
    killer_refs = df[df['killer_gold'] == 'Y']
    
    if len(killer_refs) == 0:
        return None, "No killer_gold=Y found"
    
    # Strategy: Find when speaker == word (self-reference) with killer_gold = Y
    self_refs = []
    for _, row in killer_refs.iterrows():
        speaker = str(row['speaker']).lower().strip()
        word = str(row['word']).lower().strip()
        
        # Check for self-reference (accounting for variations)
        if speaker != 'none' and word != 'none':
            # Direct match
            if speaker == word:
                self_refs.append(speaker)
            # Check if word is part of speaker name or vice versa
            elif word in speaker.split() or speaker in word:
                self_refs.append(speaker)
            # Check for first name match (e.g., "jesseoverton" and "jesse")
            elif len(word) > 3 and word in speaker:
                self_refs.append(speaker)
    
    # Count self-references
    if self_refs:
        self_ref_counts = Counter(self_refs)
        return self_ref_counts, "self-reference"
    
    # Fallback: Look at who is mentioned most in killer_gold words
    words_mentioned = []
    speakers_when_mentioned = []
    for _, row in killer_refs.iterrows():
        word = str(row['word']).lower().strip()
        speaker = str(row['speaker']).lower().strip()
        if word != 'none':
            words_mentioned.append(word)
        if speaker != 'none':
            speakers_when_mentioned.append(speaker)
    
    # Try to find a non-investigator who is mentioned
    word_counts = Counter(words_mentioned)
    
    # Filter out common words and investigators
    investigators = ['grissom', 'catherine', 'nick', 'warrick', 'sara', 'brass', 'detective', 'det', 'officer']
    filtered_words = {}
    for word, count in word_counts.items():
        is_investigator = any(inv in word for inv in investigators)
        if not is_investigator and len(word) > 2:  # Skip short words
            filtered_words[word] = count
    
    if filtered_words:
        return filtered_words, "word-frequency"
    
    return word_counts, "all-words"

def main():
    """Analyze all episodes to identify killers."""
    data_dir = Path("data/original")
    episodes = sorted(data_dir.glob("s*.tsv"))
    
    print("KILLER IDENTIFICATION ANALYSIS")
    print("=" * 80)
    print("\nStrategy: Find when speaker references themselves with killer_gold = Y")
    print("-" * 80)
    
    results = {}
    self_ref_killers = {}
    problem_episodes = []
    
    for episode_file in episodes:
        episode_id = episode_file.stem
        result, method = analyze_episode(episode_file)
        
        if result:
            if method == "self-reference":
                # Found self-references
                if len(result) == 1:
                    killer = list(result.keys())[0]
                    self_ref_killers[episode_id] = killer
                    print(f"\n{episode_id}: ✓ {killer} (self-referenced {result[killer]} times)")
                else:
                    # Multiple self-references
                    print(f"\n{episode_id}: ⚠️  Multiple self-refs: {dict(result)}")
                    # Take the most frequent
                    killer = result.most_common(1)[0][0]
                    self_ref_killers[episode_id] = killer
                    problem_episodes.append((episode_id, "multiple-self-refs", result))
            else:
                # No self-reference found
                print(f"\n{episode_id}: ❌ No self-reference found")
                if isinstance(result, dict):
                    # Show top mentioned words
                    top_words = sorted(result.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"  Top mentioned: {top_words}")
                    problem_episodes.append((episode_id, "no-self-ref", top_words))
        else:
            print(f"\n{episode_id}: ❌ {result[1]}")
            problem_episodes.append((episode_id, "no-killer-gold", None))
        
        results[episode_id] = (result, method)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal episodes: {len(episodes)}")
    print(f"Episodes with self-reference killer: {len(self_ref_killers)}")
    print(f"Problem episodes: {len(problem_episodes)}")
    
    if self_ref_killers:
        print("\nIdentified killers (self-reference method):")
        for ep, killer in sorted(self_ref_killers.items()):
            print(f"  {ep}: {killer}")
    
    if problem_episodes:
        print("\nProblem episodes requiring investigation:")
        for ep, issue, data in problem_episodes:
            print(f"  {ep}: {issue}")
            if data and issue == "no-self-ref":
                print(f"    Consider: {data[0][0]} (mentioned {data[0][1]} times)")
    
    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    # Check 1: Each episode should have exactly one killer
    episodes_with_one_killer = sum(1 for ep, (result, method) in results.items() 
                                  if method == "self-reference" and len(result) == 1)
    print(f"\n✓ Episodes with exactly 1 killer: {episodes_with_one_killer}/{len(episodes)}")
    
    # Check 2: All episodes should have a killer
    episodes_with_killer = len(self_ref_killers)
    print(f"✓ Episodes with identified killer: {episodes_with_killer}/{len(episodes)}")
    
    # Check 3: No investigators should be killers
    investigators = {'grissom', 'catherine', 'nick', 'warrick', 'sara', 'brass'}
    investigator_killers = [k for k in self_ref_killers.values() if k in investigators]
    if investigator_killers:
        print(f"⚠️  WARNING: Investigators identified as killers: {investigator_killers}")
    else:
        print(f"✓ No main investigators identified as killers")
    
    # Save results
    output_file = Path("experiments/killer_identification_analysis.txt")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Episode,Killer,Method,Confidence\n")
        for ep, killer in self_ref_killers.items():
            f.write(f"{ep},{killer},self-reference,high\n")
        for ep, issue, data in problem_episodes:
            if issue == "no-self-ref" and data:
                f.write(f"{ep},{data[0][0]},word-frequency,low\n")
    
    print(f"\nResults saved to {output_file}")
    
    return self_ref_killers, problem_episodes

if __name__ == "__main__":
    main()