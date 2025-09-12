#!/usr/bin/env python3
"""
Verify that the repository has been fully restored from memory.

Checks all critical files and provides a comprehensive status report.
"""

import os
from pathlib import Path

def check_file_or_dir(path, description, required=True):
    """Check if file or directory exists and report status."""
    path = Path(path)
    exists = path.exists()
    
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = " (required)" if required else " (optional)"
    
    if path.is_dir() and exists:
        count = len(list(path.iterdir()))
        print(f"{status} {path} - {description}{req_text} ({count} items)")
    else:
        print(f"{status} {path} - {description}{req_text}")
    
    return exists

def main():
    """Main verification function."""
    print("🔍 REPOSITORY RESTORATION VERIFICATION")
    print("=====================================")
    print()
    
    all_good = True
    
    # Core directories
    print("📁 CORE DIRECTORIES:")
    all_good &= check_file_or_dir("src", "Source code directory")
    all_good &= check_file_or_dir("src/data", "Data processing modules")
    all_good &= check_file_or_dir("src/data/models", "Data model classes")
    all_good &= check_file_or_dir("src/model", "ML model modules")
    all_good &= check_file_or_dir("scratch", "Scratch/example scripts")
    check_file_or_dir("data", "Data directory", required=False)
    check_file_or_dir("data/original", "Original TSV files", required=False)
    check_file_or_dir("experiments", "Experiment results", required=False)
    check_file_or_dir("notes", "Planning notes", required=False)
    print()
    
    # Data models
    print("🏗️ DATA MODELS:")
    all_good &= check_file_or_dir("src/data/models/__init__.py", "Data models package init")
    all_good &= check_file_or_dir("src/data/models/character.py", "Character model")
    all_good &= check_file_or_dir("src/data/models/sentence.py", "Sentence model") 
    all_good &= check_file_or_dir("src/data/models/episode.py", "Episode model")
    print()
    
    # Data processing
    print("📊 DATA PROCESSING:")
    all_good &= check_file_or_dir("src/data/__init__.py", "Data package init")
    all_good &= check_file_or_dir("src/data/preprocessing.py", "Data preprocessing")
    all_good &= check_file_or_dir("src/data/dataset.py", "PyTorch dataset")
    print()
    
    # Model architecture
    print("🧠 MODEL ARCHITECTURE:")
    all_good &= check_file_or_dir("src/model/__init__.py", "Model package init")
    all_good &= check_file_or_dir("src/model/architecture.py", "Neural network architecture")
    all_good &= check_file_or_dir("src/model/trainer.py", "Training system")
    all_good &= check_file_or_dir("src/model/experiment.py", "Experiment management")
    print()
    
    # Experimental scripts
    print("🧪 EXPERIMENTAL SCRIPTS:")
    all_good &= check_file_or_dir("run_experiments.sh", "Main experiment runner")
    all_good &= check_file_or_dir("resume_experiments.sh", "Resume incomplete experiments")
    all_good &= check_file_or_dir("monitor_experiments.sh", "Progress monitoring")
    all_good &= check_file_or_dir("quick_progress.sh", "Quick progress check")
    all_good &= check_file_or_dir("scratch/run_single_experiment.py", "Single experiment runner")
    all_good &= check_file_or_dir("scratch/example_training.py", "Example training script")
    print()
    
    # Configuration and documentation
    print("📋 CONFIGURATION & DOCS:")
    all_good &= check_file_or_dir("CLAUDE.md", "Claude Code project instructions")
    all_good &= check_file_or_dir("README.md", "Project documentation")
    all_good &= check_file_or_dir(".gitignore", "Git ignore rules")
    check_file_or_dir("requirements.txt", "Python dependencies", required=False)
    print()
    
    # Restoration scripts
    print("🚨 EMERGENCY RESTORATION:")
    all_good &= check_file_or_dir("restore_repository.py", "Repository restoration script")
    print()
    
    # Check script permissions
    print("🔒 SCRIPT PERMISSIONS:")
    scripts = [
        "run_experiments.sh",
        "resume_experiments.sh", 
        "monitor_experiments.sh",
        "quick_progress.sh"
    ]
    
    for script in scripts:
        if Path(script).exists():
            is_executable = os.access(script, os.X_OK)
            status = "✅" if is_executable else "⚠️"
            print(f"{status} {script} - {'Executable' if is_executable else 'Not executable (run chmod +x)'}")
        else:
            print(f"❌ {script} - Missing")
    print()
    
    # Final status
    if all_good:
        print("🎉 REPOSITORY FULLY RESTORED!")
        print("============================")
        print()
        print("✅ All critical files are present")
        print("✅ Directory structure is complete")
        print("✅ Ready for experiments")
        print()
        print("Next steps:")
        print("1. Create virtual environment: python3 -m venv venv")
        print("2. Activate environment: source venv/bin/activate")
        print("3. Install dependencies: pip install torch transformers")
        print("4. Add data files to data/original/")
        print("5. Run experiments: ./run_experiments.sh")
        print()
        print("Monitor progress: ./monitor_experiments.sh")
        print("Resume experiments: ./resume_experiments.sh")
        
    else:
        print("⚠️ RESTORATION INCOMPLETE")
        print("========================")
        print()
        print("Some critical files are missing. Please:")
        print("1. Check the error messages above")
        print("2. Re-run restore_repository.py if needed")
        print("3. Manually create any missing files")

if __name__ == "__main__":
    main()