#!/usr/bin/env python
"""
EETA Setup Verification Script

Run this to verify your installation is working correctly.
Usage: python test_setup.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, '.')

def test_section(name):
    """Print section header."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print('='*50)

def check_import(module_path, item=None):
    """Try to import a module."""
    try:
        if item:
            module = __import__(module_path, fromlist=[item])
            getattr(module, item)
        else:
            __import__(module_path)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print("EETA Setup Verification")
    print("=" * 50)
    
    errors = []
    warnings = []
    
    # Test 1: Directory structure
    test_section("Directory Structure")
    required_dirs = [
        "configs",
        "src",
        "src/agents",
        "src/rl",
        "src/environment",
        "src/data",
        "src/training",
        "src/evaluation",
        "src/risk",
        "src/utils",
        "data/processed",
        "experiments/checkpoints",
    ]
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            errors.append(f"Missing directory: {dir_path}")
    
    # Test 2: Required files
    test_section("Required Files")
    required_files = [
        ("configs/default.yaml", "Config file"),
        ("requirements.txt", "Dependencies"),
        ("run.py", "Entry point"),
        ("src/__init__.py", "Package init"),
        ("src/agents/__init__.py", "Agents init"),
        ("src/rl/__init__.py", "RL init"),
        ("src/data/__init__.py", "Data init"),
    ]
    
    for filepath, desc in required_files:
        if os.path.isfile(filepath):
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath} ({desc} missing)")
            errors.append(f"Missing file: {filepath}")
    
    # Test 3: Python imports
    test_section("Python Imports")
    
    imports_to_test = [
        ("src.utils.config", "get_config"),
        ("src.utils.logging", "setup_logging"),
        ("src.data.sources", "DataSourceManager"),
        ("src.data.preprocessor", "StatePreprocessor"),
        ("src.rl.dqn", "DQNAgent"),
        ("src.rl.thompson", "ThompsonSampler"),
        ("src.rl.replay_buffer", "ReplayBuffer"),
        ("src.environment.trading_env", "EarningsTradingEnv"),
        ("src.training.train", "EETATrainer"),
        ("src.risk.controller", "RiskController"),
    ]
    
    for module, item in imports_to_test:
        success, error = check_import(module, item)
        if success:
            print(f"✅ from {module} import {item}")
        else:
            print(f"❌ from {module} import {item}")
            print(f"   Error: {error}")
            errors.append(f"Import failed: {module}.{item}")
    
    # Test 4: Dependencies
    test_section("Dependencies")
    
    deps = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("torch", None),
        ("sklearn", "scikit-learn"),
        ("gymnasium", None),
        ("yfinance", "yf"),
        ("yaml", "pyyaml"),
    ]
    
    for dep, alias in deps:
        try:
            __import__(dep)
            print(f"✅ {alias or dep}")
        except ImportError:
            print(f"❌ {alias or dep} (not installed)")
            errors.append(f"Missing dependency: {alias or dep}")
    
    # Test 5: Config loading
    test_section("Configuration")
    
    try:
        from src.utils.config import get_config
        config = get_config()
        print(f"✅ Config loaded successfully")
        print(f"   State dim: {config.dqn.state_dim}")
        print(f"   Action dim: {config.dqn.action_dim}")
        print(f"   Hidden dims: {config.dqn.hidden_dims}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        errors.append(f"Config error: {e}")
    
    # Test 6: Data source
    test_section("Data Sources")
    
    try:
        from src.data.sources import DataSourceManager
        dm = DataSourceManager()
        print(f"✅ DataSourceManager initialized")
        
        # Try to get tickers (doesn't require network)
        tickers = dm.get_sp500_tickers()[:5]
        print(f"✅ S&P 500 tickers accessible: {tickers}")
    except Exception as e:
        print(f"⚠️  Data source warning: {e}")
        warnings.append(f"Data source issue: {e}")
    
    # Test 7: DQN Agent
    test_section("DQN Agent")
    
    try:
        from src.rl.dqn import DQNAgent
        agent = DQNAgent(state_dim=43, action_dim=5)
        print(f"✅ DQN Agent created")
        print(f"   Epsilon: {agent.epsilon:.2f}")
        print(f"   Device: {agent.device}")
    except Exception as e:
        print(f"❌ DQN Agent failed: {e}")
        errors.append(f"DQN error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if not errors and not warnings:
        print("✅ All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. python run.py build-data --max-tickers 10")
        print("  2. python run.py train --episodes 20")
        return 0
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} warnings:")
        for w in warnings:
            print(f"   - {w}")
    
    if errors:
        print(f"\n❌ {len(errors)} errors found:")
        for e in errors:
            print(f"   - {e}")
        print("\nPlease fix these issues before running the project.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
