"""
Data preprocessing for CSI episode transcripts.

Handles TSV file parsing, character name normalization, and episode construction
from word-level transcript data.
"""

import csv
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter

from data.models import Character, Sentence, Episode

logger = logging.getLogger(__name__)


def normalize_character_name(raw_name: str) -> str:
    """
    Normalize character name from raw TSV format.
    
    Handles cases like:
    - "tinacollins" -> "tina collins"
    - "dr.robbins" -> "dr robbins"
    - "GRISSOM" -> "grissom"
    
    Args:
        raw_name: Raw character name from TSV
    
    Returns:
        Normalized character name
    """
    if not raw_name or raw_name.lower() == 'none':
        return None
    
    name = raw_name.strip()
    
    # Handle concatenated names by inserting spaces before capitals
    # tinacollins -> tina collins
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    
    # Convert to lowercase after handling capitals
    name = name.lower()
    
    # Clean up punctuation
    name = re.sub(r'[^\w\s]', ' ', name)
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name


def parse_episode_tsv(file_path: Path) -> Tuple[List[Sentence], List[Character]]:
    """
    Parse a single episode TSV file into sentences and characters.
    
    Args:
        file_path: Path to episode TSV file
    
    Returns:
        Tuple of (sentences, characters) for the episode
    """
    episode_id = file_path.stem  # e.g., "s01e07"
    
    logger.info(f"Loading episode {episode_id} from {file_path}")
    
    # Read TSV data
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    
    # Group words by sentence (caseID, sentID)
    sentence_data = defaultdict(list)
    for row in rows:
        case_id = row['caseID']
        sent_id = row['sentID']
        sentence_key = f"{case_id}:{sent_id}"
        sentence_data[sentence_key].append(row)
    
    # Process sentences
    sentences = []
    characters_seen = {}  # Track unique characters
    
    for sentence_key, words in sentence_data.items():
        # Skip sentences with no speaker
        speaker_name = words[0].get('speaker', '').strip()
        if not speaker_name or speaker_name.lower() == 'none':
            continue
        
        # Aggregate sentence text
        sentence_text = ' '.join(word['word'] for word in words if word.get('word'))
        if not sentence_text.strip():
            continue
        
        # Create or get character
        normalized_name = normalize_character_name(speaker_name)
        
        # Additional safety check for normalized name
        if normalized_name is None or not normalized_name.strip():
            continue
            
        char_key = f"{episode_id}:{normalized_name}"
        
        if char_key not in characters_seen:
            character = Character(
                raw_name=speaker_name,
                normalized_name=normalized_name,
                episode_id=episode_id
            )
            characters_seen[char_key] = character
        else:
            character = characters_seen[char_key]
        
        # Extract metadata
        first_word = words[0]
        gold_labels = {
            'killer_gold': first_word.get('killer_gold'),
            'other_gold': first_word.get('other_gold'),
            'suspect_gold': first_word.get('suspect_gold')
        }
        
        # Extract timing data
        timing_data = {
            'medion_time': first_word.get('medion_time'),
            'start_time': first_word.get('start_time'),
            'end_time': first_word.get('end_time')
        }
        
        # Add individual timing markers
        for i in range(1, 6):
            timing_key = f'i{i}_time'
            if timing_key in first_word:
                timing_data[timing_key] = first_word[timing_key]
        
        # Create sentence
        sentence = Sentence(
            case_id=first_word['caseID'],
            sent_id=first_word['sentID'],
            speaker=character,
            text=sentence_text,
            gold_labels=gold_labels,
            human_guess=first_word.get('human_guess'),
            timing_data=timing_data
        )
        
        sentences.append(sentence)
        
        # Check for malformed sentence data
        try:
            sentence_words = [word['word'] for word in words if word.get('word')]
            if len(sentence_words) != len(words):
                logger.warning(f"Skipping malformed sentence {sentence_key} in {episode_id}: Inconsistent word data for sentence {sentence_key}")
                sentences.pop()  # Remove the malformed sentence
                continue
        except Exception:
            logger.warning(f"Skipping malformed sentence {sentence_key} in {episode_id}: Data validation failed")
            sentences.pop()  # Remove the malformed sentence
            continue
    
    characters = list(characters_seen.values())
    
    logger.info(f"Loaded episode {episode_id}: {len(sentences)} sentences, {len(characters)} characters")
    
    return sentences, characters


def load_csi_data_complete(data_directory: Path, character_mode: str = 'episode-isolated') -> Dict[str, Any]:
    """
    Load complete CSI dataset from directory of TSV files.
    
    Args:
        data_directory: Directory containing episode TSV files
        character_mode: Either 'episode-isolated' or 'cross-episode'
    
    Returns:
        Dictionary containing episodes and metadata
    """
    data_directory = Path(data_directory)
    
    # Find all TSV files
    tsv_files = sorted(data_directory.glob('s*.tsv'))
    
    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in {data_directory}")
    
    logger.info(f"Loading {len(tsv_files)} episodes from {data_directory}")
    
    episodes = []
    all_characters = {}  # For cross-episode mode
    
    for tsv_file in tsv_files:
        try:
            sentences, episode_characters = parse_episode_tsv(tsv_file)
            
            if character_mode == 'cross-episode':
                # Consolidate characters across episodes
                consolidated_characters = []
                character_mapping = {}  # Map episode chars to consolidated chars
                
                for ep_char in episode_characters:
                    global_key = ep_char.normalized_name.lower()
                    
                    if global_key not in all_characters:
                        # Create new consolidated character
                        global_char = Character(
                            raw_name=ep_char.raw_name,
                            normalized_name=ep_char.normalized_name,
                            episode_id=None  # No specific episode for cross-episode mode
                        )
                        all_characters[global_key] = global_char
                    
                    character_mapping[ep_char] = all_characters[global_key]
                    consolidated_characters.append(all_characters[global_key])
                
                # Update sentence speakers to use consolidated characters
                for sentence in sentences:
                    sentence.speaker = character_mapping[sentence.speaker]
                
                episode_characters = list(set(consolidated_characters))
            
            episode = Episode(
                episode_id=tsv_file.stem,
                sentences=sentences,
                characters=episode_characters
            )
            
            episodes.append(episode)
            
        except Exception as e:
            logger.error(f"Failed to load episode {tsv_file.stem}: {e}")
            continue
    
    if not episodes:
        raise ValueError("No episodes were successfully loaded")
    
    logger.info("Successfully loaded {} episodes".format(len(episodes)))
    
    # Sort episodes chronologically
    episodes.sort(key=lambda ep: ep.episode_id)
    logger.info("Sorted {} episodes chronologically".format(len(episodes)))
    
    # Calculate summary statistics
    total_sentences = sum(len(ep.sentences) for ep in episodes)
    if character_mode == 'cross-episode':
        unique_characters = len(all_characters)
        total_character_instances = sum(len(ep.characters) for ep in episodes)
    else:
        all_chars = set()
        for ep in episodes:
            for char in ep.characters:
                all_chars.add(char.get_unique_id('episode-isolated'))
        unique_characters = len(all_chars)
        total_character_instances = sum(len(ep.characters) for ep in episodes)
    
    avg_sentences_per_episode = total_sentences / len(episodes) if episodes else 0
    avg_characters_per_episode = total_character_instances / len(episodes) if episodes else 0
    
    summary_stats = {
        'character_mode': character_mode,
        'total_episodes': len(episodes),
        'total_sentences': total_sentences,
        'total_characters_instances': total_character_instances,
        'unique_characters': unique_characters,
        'avg_sentences_per_episode': avg_sentences_per_episode,
        'avg_characters_per_episode': avg_characters_per_episode
    }
    
    logger.info("Data loading complete: {}".format(summary_stats))
    
    return {
        'episodes': episodes,
        'summary_stats': summary_stats,
        'character_mode': character_mode
    }