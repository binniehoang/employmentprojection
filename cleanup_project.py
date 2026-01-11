"""
Cleanup script to remove unnecessary files from the employment projection project.
This script safely removes duplicate files, cache directories, and intermediate processing files.
"""

import os
import shutil
import sys
from pathlib import Path

def safe_remove_file(file_path):
    """Safely remove a file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Removed file: {file_path}")
            return True
        else:
            print(f"⚠️  File not found (already deleted?): {file_path}")
            return False
    except Exception as e:
        print(f"❌ Error removing file {file_path}: {e}")
        return False

def safe_remove_directory(dir_path):
    """Safely remove a directory if it exists."""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"✅ Removed directory: {dir_path}")
            return True
        else:
            print(f"⚠️  Directory not found (already deleted?): {dir_path}")
            return False
    except Exception as e:
        print(f"❌ Error removing directory {dir_path}: {e}")
        return False

def main():
    """Main cleanup function."""
    print("🧹 Starting Employment Projection Project Cleanup...")
    print("=" * 60)
    
    # Get project root directory
    project_root = os.getcwd()
    print(f"Project root: {project_root}")
    print()
    
    # Files to remove
    files_to_remove = [
        # Duplicate training log
        "model_data/logs/training_log.json",
        
        # Intermediate processing files
        "data/encoded_features.csv",
        "data/scaled_features.csv", 
        "model_data/full_feature_columns.txt"
    ]
    
    # Directories to remove (cache directories)
    dirs_to_remove = [
        "src/__pycache__",
        "src/model/__pycache__",
        "tests/__pycache__",
        ".pytest_cache"
    ]
    
    # Optional files (ask user)
    optional_files = [
        "src/eda.py",
        "src/utils.py"
    ]
    
    removed_count = 0
    
    print("1️⃣ Removing duplicate and intermediate files...")
    print("-" * 40)
    for file_path in files_to_remove:
        full_path = os.path.join(project_root, file_path)
        if safe_remove_file(full_path):
            removed_count += 1
    
    print(f"\n2️⃣ Removing cache directories...")
    print("-" * 40)
    for dir_path in dirs_to_remove:
        full_path = os.path.join(project_root, dir_path)
        if safe_remove_directory(full_path):
            removed_count += 1
    
    # Also remove the empty logs directory if it exists
    logs_dir = os.path.join(project_root, "model_data", "logs")
    if os.path.exists(logs_dir) and not os.listdir(logs_dir):  # If directory is empty
        if safe_remove_directory(logs_dir):
            removed_count += 1
    
    print(f"\n3️⃣ Optional cleanup (potentially unused modules)...")
    print("-" * 40)
    print("The following files are potentially unused:")
    for file_path in optional_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"  - {file_path}")
    
    response = input(f"\nDo you want to remove these optional files? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        for file_path in optional_files:
            full_path = os.path.join(project_root, file_path)
            if safe_remove_file(full_path):
                removed_count += 1
    else:
        print("⏭️  Skipping optional file removal")
    
    # Calculate space saved (rough estimate)
    print(f"\n🎉 Cleanup Complete!")
    print("=" * 60)
    print(f"📁 Items removed: {removed_count}")
    print(f"💾 Cache directories will be auto-regenerated when needed")
    print(f"🔧 Your project is now cleaner and more organized!")
    
    print(f"\n📋 Remaining essential files:")
    essential_dirs = ["data", "model_data", "plots", "src", "tests"]
    for dir_name in essential_dirs:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            file_count = sum(1 for _ in Path(dir_path).rglob('*') if _.is_file())
            print(f"  - {dir_name}/: {file_count} files")
    
    print(f"\n✨ Project cleanup successful! Your workspace is now optimized.")

if __name__ == "__main__":
    main()