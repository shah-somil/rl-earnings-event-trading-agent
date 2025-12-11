#!/usr/bin/env python
"""
EETA - Main Entry Point

Demonstrates the complete earnings trading agent workflow:
1. Build dataset
2. Train agent with curriculum learning
3. Evaluate with walk-forward validation
4. Run ablation studies
5. Compare to benchmarks
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import get_config
from src.utils.logging import setup_logging, TrainingLogger

# suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_dataset(args):
    """Build the earnings dataset."""
    from src.data import build_dataset as build_ds
    from src.data.sources import DataSourceManager
    
    logging.info("Building earnings dataset...")
    
    # Get tickers
    source_manager = DataSourceManager()
    tickers = source_manager.yfinance.get_sp500_tickers()
    
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    
    logging.info(f"Processing {len(tickers)} tickers...")
    
    dataset = build_ds(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_path=args.cache_path
    )

    if dataset.empty:
        logging.error("Dataset build returned 0 events. No file was written; check data source availability/network and your date range.")
        return dataset
    
    if 'earnings_date' not in dataset.columns:
        logging.error("Dataset missing expected column 'earnings_date'; inspect data source output with --log-level DEBUG.")
        return dataset
    
    logging.info(f"Dataset built with {len(dataset)} events")
    logging.info(f"Date range: {dataset['earnings_date'].min()} to {dataset['earnings_date'].max()}")
    
    return dataset


def train(args):
    """Train the agent."""
    import pandas as pd
    from src.training import EETATrainer
    
    logging.info("Starting training...")
    
    # Load dataset
    if Path(args.data_path).exists():
        dataset = pd.read_parquet(args.data_path)
        logging.info(f"Loaded dataset with {len(dataset)} events")
    else:
        logging.error(f"Dataset not found: {args.data_path}")
        logging.info("Run with --build-data first")
        return
    
    # Get config
    config = get_config()
    
    # Create trainer
    trainer = EETATrainer(
        data=dataset,
        config=config._raw,
        experiment_name=args.experiment_name
    )
    
    # Train
    results = trainer.train(
        n_episodes=args.episodes,
        eval_frequency=args.eval_freq,
        checkpoint_frequency=args.checkpoint_freq
    )
    
    logging.info(f"Training complete!")
    logging.info(f"Final epsilon: {results['final_epsilon']:.3f}")
    logging.info(f"Best Sharpe: {results['best_sharpe']:.2f}")




# def validate(args):
#     """Run walk-forward validation."""
#     import pandas as pd
#     from src.training import (
#         WalkForwardValidator, 
#         create_agent_factory,
#         train_fold,
#         test_fold
#     )
    
#     logging.info("Starting walk-forward validation...")
    
#     # Load dataset
#     if Path(args.data_path).exists():
#         dataset = pd.read_parquet(args.data_path)
#     else:
#         logging.error(f"Dataset not found: {args.data_path}")
#         return
    
#     # Get config
#     config = get_config()
    
#     # Create validator
#     validator = WalkForwardValidator(
#         data=dataset,
#         min_train_years=args.min_train_years,
#         test_years=args.test_years
#     )
    
#     # Run validation
#     results = validator.run_validation(
#         agent_factory=create_agent_factory(config._raw),
#         train_fn=train_fold,
#         test_fn=test_fold,
#         config=config._raw
#     )
    
#     # Print results
#     logging.info("=" * 50)
#     logging.info("WALK-FORWARD VALIDATION RESULTS")
#     logging.info("=" * 50)
    
#     agg = results['aggregate']
#     logging.info(f"Number of Folds: {agg['n_folds']}")
#     logging.info(f"Mean Test Return: {agg['mean_return']:.2%}")
#     logging.info(f"Std Test Return: {agg['std_return']:.2%}")
#     logging.info(f"Mean Sharpe Ratio: {agg['mean_sharpe']:.2f}")
#     logging.info(f"Mean Win Rate: {agg['mean_win_rate']:.1%}")

def validate(args):
    """Run walk-forward validation."""
    import pandas as pd
    import json
    from datetime import datetime
    from src.training import (
        WalkForwardValidator, 
        create_agent_factory,
        train_fold,
        test_fold
    )
    
    print("Starting walk-forward validation...")
    
    # Load dataset
    if Path(args.data_path).exists():
        dataset = pd.read_parquet(args.data_path)
        print(f"Loaded dataset with {len(dataset)} events")
    else:
        print(f"ERROR: Dataset not found: {args.data_path}")
        return
    
    # Get config
    config = get_config()
    
    # Create validator
    validator = WalkForwardValidator(
        data=dataset,
        min_train_years=args.min_train_years,
        test_years=args.test_years
    )
    
    folds = validator.get_folds()
    print(f"Running {len(folds)} folds...")
    for f in folds:
        print(f"  Fold {f['fold_id']}: Train {f['train_years']} ({f['train_size']} samples), Test {f['test_year']} ({f['test_size']} samples)")
    
    # Run validation
    results = validator.run_validation(
        agent_factory=create_agent_factory(config._raw),
        train_fn=train_fold,
        test_fn=test_fold,
        config=config._raw
    )
    
    # Print results
    print("=" * 50)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 50)
    
    agg = results['aggregate']
    print(f"Number of Folds: {agg['n_folds']}")
    print(f"Mean Test Return: {agg['mean_return']:.2%}")
    print(f"Std Test Return: {agg['std_return']:.2%}")
    print(f"Mean Sharpe Ratio: {agg['mean_sharpe']:.2f}")
    print(f"Mean Win Rate: {agg['mean_win_rate']:.1%}")
    print(f"Total Test Trades: {agg['total_test_trades']}")
    
    print("\nPer-Fold Results:")
    fold_results = []
    for fold in results['folds']:
        test = fold['test_results']
        train = fold['train_results']
        print(f"  Fold {fold['fold_id']} ({fold['test_year']}): "
              f"Return={test['total_return']:.2%}, "
              f"Sharpe={test['sharpe_ratio']:.2f}, "
              f"Trades={test['n_trades']}, "
              f"Win Rate={test['win_rate']:.1%}")
        
        fold_results.append({
            'fold_id': fold['fold_id'],
            'train_years': fold['train_years'],
            'test_year': fold['test_year'],
            'train_size': fold['train_size'],
            'test_size': fold['test_size'],
            'train_episodes': train['episodes'],
            'train_mean_reward': float(train['mean_reward']),
            'test_return': float(test['total_return']),
            'test_sharpe': float(test['sharpe_ratio']),
            'test_trades': test['n_trades'],
            'test_win_rate': float(test['win_rate'])
        })
    
    # === SAVE RESULTS ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON
    summary = {
        'timestamp': timestamp,
        'dataset_size': len(dataset),
        'n_folds': agg['n_folds'],
        'mean_return': float(agg['mean_return']),
        'std_return': float(agg['std_return']),
        'mean_sharpe': float(agg['mean_sharpe']),
        'mean_win_rate': float(agg['mean_win_rate']),
        'total_test_trades': agg['total_test_trades'],
        'fold_returns': [float(x) for x in agg.get('fold_returns', [])],
        'fold_sharpes': [float(x) for x in agg.get('fold_sharpes', [])]
    }
    
    json_path = results_dir / f"validation_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {json_path}")
    
    # Save detailed CSV
    csv_path = results_dir / f"validation_{timestamp}_folds.csv"
    pd.DataFrame(fold_results).to_csv(csv_path, index=False)
    print(f"Fold details saved to: {csv_path}")
    
    print("=" * 50)
    print("Validation complete!")
    
    return results

# def evaluate(args):
#     """Run evaluation with benchmarks."""
#     import pandas as pd
#     from src.evaluation import run_benchmark_comparison
#     from src.rl.dqn import DQNAgent
    
#     logging.info("Running evaluation...")
    
#     # Load dataset
#     if Path(args.data_path).exists():
#         dataset = pd.read_parquet(args.data_path)
#     else:
#         logging.error(f"Dataset not found: {args.data_path}")
#         return
    
#     # For demo, use random returns
#     import numpy as np
#     agent_returns = list(np.random.randn(100) * 0.02)
    
#     # Run benchmark comparison
#     comparison = run_benchmark_comparison(agent_returns, dataset)
    
#     print("\nBenchmark Comparison:")
#     print(comparison.to_string())

def evaluate(args):
    """Run evaluation with benchmarks."""
    import pandas as pd
    import numpy as np
    from src.evaluation import run_benchmark_comparison
    from src.rl.dqn import DQNAgent
    from src.data.preprocessor import StatePreprocessor, FEATURE_NAMES
    from src.environment.trading_env import EarningsTradingEnv
    
    logging.info("Running evaluation...")
    
    # Load dataset
    if Path(args.data_path).exists():
        dataset = pd.read_parquet(args.data_path)
    else:
        logging.error(f"Dataset not found: {args.data_path}")
        return
    
    # Load model
    # Load model
    if args.model_path is None:
        # Find most recent checkpoint
        import glob
        checkpoints = glob.glob('experiments/checkpoints/*/final_model.pt')
        if checkpoints:
            args.model_path = max(checkpoints)
            logging.info(f"Using most recent model: {args.model_path}")
        else:
            logging.error("No model path specified and no checkpoints found. Use --model-path")
            return

    if not Path(args.model_path).exists():
        logging.error(f"Model not found: {args.model_path}")
        return
    
    # Setup preprocessor
    preprocessor = StatePreprocessor()
    feature_cols = [c for c in dataset.columns if c in FEATURE_NAMES]
    preprocessor.fit(dataset[feature_cols])
    
    # Load agent
    agent = DQNAgent(state_dim=len(FEATURE_NAMES), action_dim=5)
    agent.load(args.model_path)
    logging.info(f"Loaded model from {args.model_path}")
    
    # Run agent through all events
    env = EarningsTradingEnv(data=dataset, preprocessor=preprocessor, shuffle=False)
    state = env.reset()
    
    agent_returns = []
    while True:
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action)
        agent_returns.append(info.get('pnl_pct', 0))
        state = next_state
        if done:
            break
    
    logging.info(f"Agent made {len(agent_returns)} trades")
    
    # Run benchmark comparison
    comparison = run_benchmark_comparison(agent_returns, dataset)
    
    print("\nBenchmark Comparison:")
    print(comparison.to_string())

def ablation(args):
    """Run ablation studies."""
    import pandas as pd
    import torch
    from src.evaluation.ablation import run_ablation_study
    from src.rl.dqn import DQNAgent
    from src.data.preprocessor import StatePreprocessor
    
    print("="*60)
    print("EETA ABLATION STUDY")
    print("="*60)
    
    # Load dataset
    if Path(args.data_path).exists():
        dataset = pd.read_parquet(args.data_path)
        print(f"✓ Loaded dataset with {len(dataset)} events")
    else:
        print(f"✗ Dataset not found: {args.data_path}")
        return
    
    # Find model
    if args.model_path is None:
        import glob
        checkpoints = glob.glob('experiments/checkpoints/*/checkpoint_ep49.pt')
        if not checkpoints:
            checkpoints = glob.glob('experiments/checkpoints/*/final_model.pt')
        if checkpoints:
            args.model_path = max(checkpoints)
        else:
            print("✗ No model found. Run training first.")
            return
    
    print(f"✓ Loading model: {args.model_path}")
    
    # Get config
    config = get_config()
    dqn_config = config._raw.get('dqn', {})
    
    # Create agent
    agent = DQNAgent(
        state_dim=dqn_config.get('state_dim', 43),
        action_dim=dqn_config.get('action_dim', 5),
        hidden_dims=dqn_config.get('hidden_dims', [128, 64]),
        learning_rate=dqn_config.get('learning_rate', 0.001),
        gamma=dqn_config.get('gamma', 0.99),
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        batch_size=dqn_config.get('batch_size', 32),
        buffer_size=dqn_config.get('replay_buffer_size', 10000)
    )
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    agent.q_network.load_state_dict(checkpoint['q_network'])
    agent.epsilon = 0.0
    
    # Create preprocessor
    preprocessor = StatePreprocessor()
    preprocessor.fit(dataset)
    
    # Run ablation
    print(f"\nRunning {args.n_episodes} episodes per experiment...")
    
    results = run_ablation_study(
        agent=agent,
        test_data=dataset,
        preprocessor=preprocessor,
        config={'n_episodes': args.n_episodes}
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\n📊 Summary Table:\n")
    summary = results['summary']
    print(summary.to_string(index=False))
    
    print("\n📈 Component Importance (Sharpe drop when removed):\n")
    for component, value in sorted(results['component_importance'].items(), 
                                   key=lambda x: x[1], reverse=True):
        bar = "█" * int(abs(value) * 10)
        print(f"  {component:20s}: {value:+.2f} {bar}")
    
    # Save results
    from datetime import datetime
    output_dir = Path('experiments/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'ablation_{timestamp}.csv'
    summary.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EETA - Earnings Event Trading Agent"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Build data command
    build_parser = subparsers.add_parser('build-data', help='Build earnings dataset')
    build_parser.add_argument('--start-date', default='2019-01-01', help='Start date')
    build_parser.add_argument('--end-date', default='2024-12-01', help='End date')
    build_parser.add_argument('--max-tickers', type=int, default=50, help='Max tickers to process')
    build_parser.add_argument('--cache-path', default='data/processed/earnings_dataset.parquet')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument('--data-path', default='data/processed/earnings_dataset.parquet')
    train_parser.add_argument('--experiment-name', default=None, help='Experiment name')
    train_parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    train_parser.add_argument('--eval-freq', type=int, default=25, help='Evaluation frequency')
    train_parser.add_argument('--checkpoint-freq', type=int, default=10, help='Checkpoint frequency')
    
    # Validate command
    val_parser = subparsers.add_parser('validate', help='Run walk-forward validation')
    val_parser.add_argument('--data-path', default='data/processed/earnings_dataset.parquet')
    val_parser.add_argument('--min-train-years', type=int, default=3)
    val_parser.add_argument('--test-years', type=int, default=1)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    eval_parser.add_argument('--data-path', default='data/processed/earnings_dataset.parquet')
    eval_parser.add_argument('--model-path', default=None, help='Path to trained model')
    

    # Ablation command
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation studies')
    ablation_parser.add_argument('--data-path', default='data/processed/earnings_dataset.parquet')
    ablation_parser.add_argument('--model-path', default=None, help='Path to trained model')
    ablation_parser.add_argument('--n-episodes', type=int, default=5, help='Episodes per experiment')

    # Common arguments
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    # args = parser.parse_args()
    
    # if args.model_path is None:
    # # Find most recent checkpoint
    #     import glob
    #     checkpoints = glob.glob('experiments/checkpoints/*/final_model.pt')
    #     if checkpoints:
    #         args.model_path = max(checkpoints)
    #         logging.info(f"Using most recent model: {args.model_path}")
    #     else:
    #         logging.error("No model path specified and no checkpoints found")
    #         return
   
    args = parser.parse_args()

    # Only check model_path for evaluate command
    if args.command == 'evaluate' and (not hasattr(args, 'model_path') or args.model_path is None):
        import glob
        checkpoints = glob.glob('experiments/checkpoints/*/final_model.pt')
        if checkpoints:
            args.model_path = max(checkpoints)
            logging.info(f"Using most recent model: {args.model_path}")
        else:
            logging.error("No model path specified and no checkpoints found. Use --model-path")
            return
        
    # Setup logging
    setup_logging('eeta', level=args.log_level)
    
    if args.command == 'build-data':
        build_dataset(args)
    elif args.command == 'train':
        train(args)
    elif args.command == 'validate':
        validate(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'ablation':
        ablation(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
