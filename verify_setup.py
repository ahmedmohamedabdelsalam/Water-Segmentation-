"""Simple verification script to check project setup."""

import sys
from pathlib import Path

def check_structure():
    """Check if all required directories and files exist."""
    required_dirs = [
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "scripts",
        "configs",
    ]
    
    required_files = [
        "requirements.txt",
        "README.md",
        ".gitignore",
        "configs/default.yaml",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/run_experiments.py",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py",
    ]
    
    print("="*80)
    print("PROJECT STRUCTURE VERIFICATION")
    print("="*80)
    
    all_good = True
    
    print("\nChecking directories...")
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists() and full_path.is_dir():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING!")
            all_good = False
    
    print("\nChecking files...")
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists() and full_path.is_file():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING!")
            all_good = False
    
    print("\n" + "="*80)
    if all_good:
        print("✅ All checks passed! Project structure is correct.")
    else:
        print("❌ Some checks failed. Please review the missing items.")
    print("="*80)
    
    return all_good


def check_imports():
    """Check if key modules can be imported."""
    print("\n" + "="*80)
    print("IMPORT VERIFICATION")
    print("="*80)
    
    imports_to_test = [
        ("src.data", "create_dataloaders"),
        ("src.models", "create_model"),
        ("src.training", "Trainer"),
        ("src.evaluation", "Evaluator"),
        ("src.utils", "set_seed"),
    ]
    
    all_good = True
    
    for module_name, obj_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[obj_name])
            obj = getattr(module, obj_name)
            print(f"  ✓ from {module_name} import {obj_name}")
        except Exception as e:
            print(f"  ✗ from {module_name} import {obj_name} - ERROR: {str(e)[:50]}")
            all_good = False
    
    print("="*80)
    if all_good:
        print("✅ All imports successful!")
    else:
        print("❌ Some imports failed. Check error messages above.")
    print("="*80)
    
    return all_good


def main():
    """Run all verification checks."""
    print("\n🔍 Water Segmentation Project Verification\n")
    
    structure_ok = check_structure()
    
    if structure_ok:
        imports_ok = check_imports()
        
        if imports_ok:
            print("\n🎉 PROJECT READY TO USE!")
            print("\nNext steps:")
            print("  1. Install dependencies: pip install -r requirements.txt")
            print("  2. Quick test: python scripts/train.py --quick-test")
            print("  3. Full training: python scripts/train.py")
            return 0
        else:
            print("\n⚠️  Structure OK but imports failed.")
            print("Make sure dependencies are installed: pip install -r requirements.txt")
            return 1
    else:
        print("\n⚠️  Project structure incomplete.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
