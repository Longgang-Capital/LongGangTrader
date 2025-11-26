import os
import sys
import shutil
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

def setup_paths_and_imports():
    """Add project root to sys.path and import custom modules."""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to sys.path")
    
    # Now we can import our custom modules
    from longgang_trader.alpha.dl_model_factor import inference_on_test_set
    from longgang_trader.optimize.optimizer import LayeredOptimizer, EqualWeightOptimizer
    from longgang_trader.backtesting.backtester import Backtester
    from longgang_trader.backtesting.backtester import BaseStrategy as TopNFactorStrategy # Alias for compatibility

    # Return imported modules to be used in main
    return {
        "inference_on_test_set": inference_on_test_set,
        "LayeredOptimizer": LayeredOptimizer,
        "EqualWeightOptimizer": EqualWeightOptimizer,
        "Backtester": Backtester,
        "TopNFactorStrategy": TopNFactorStrategy
    }

def main():
    """Main function to run the end-to-end workflow."""
    
    # --- 0. Setup and Configuration ---
    print("--- Step 0: Setup and Configuration ---")
    
    # Dynamically import modules after setting path
    modules = setup_paths_and_imports()
    inference_on_test_set = modules["inference_on_test_set"]
    LayeredOptimizer = modules["LayeredOptimizer"]
    EqualWeightOptimizer = modules["EqualWeightOptimizer"]
    Backtester = modules["Backtester"]
    TopNFactorStrategy = modules["TopNFactorStrategy"]

    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RESULTS_DIR = DATA_DIR / 'results'
    MARKET_DATA_PATH = PROJECT_ROOT / 'notebooks' / 'test_data.parquet'

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load full data metadata
    with open(DATA_DIR / 'test_features.bin.json', 'r') as f:
        features_meta = json.load(f)
    
    DI, II, N_FEAT = features_meta['shape']
    START_DATE = str(features_meta['start_date'])
    END_DATE = str(features_meta['end_date'])

    CONFIG = {
        "feature_path": str(DATA_DIR / 'test_features.bin'),
        "label_path": str(DATA_DIR / 'test_labels.bin'), # Dummy file
        "model_path": str(PROJECT_ROOT / 'notebooks' / 'AttentionGRU.pt'),
        "inference_output_path": str(RESULTS_DIR / 'preds_full.npy'),
        "dl_model_config": {
            "seq_len": 60,
            "hidden_dim": 128,
            "num_layers": 1,
            "dropout": 0.1,
            "batch_size": 1024,
            "num_workers": 0,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        },
        "optimizer_config": { "n_groups": 5 },
        "backtest_config": {
            "initial_capital": 1_000_000,
            "transaction_cost": 0.001,
            "top_n": 10,
            "date_col": "date",
            "symbol_col": "order_book_id",
            "weight_col": "target_weight",
            "close_col": "close",
        }
    }
    print("Configuration loaded.")

    # --- Steps 1 & 2: Factor Calculation (DL Model Inference) ---
    inference_output_path = Path(CONFIG["inference_output_path"])
    
    if inference_output_path.exists():
        print(f"\n--- Steps 1 & 2: Skipping inference, result file already exists ---")
        print(f"Loading existing predictions from {inference_output_path}")
        predictions = np.load(inference_output_path)
    else:
        # --- Step 1: Create dummy label file for inference ---
        print("\n--- Step 1: Creating dummy label file for inference ---")
        dummy_labels = np.zeros((DI, II), dtype=np.float32)
        labels_path = Path(CONFIG["label_path"])
        dummy_labels.tofile(labels_path)
        with open(f"{labels_path}.json", "w") as f:
            json.dump({"dtype": "float32", "shape": [DI, II]}, f)
        print(f"Dummy labels created at {labels_path} and {labels_path}.json")

        # --- Step 2: Run inference ---
        print("\n--- Step 2: Running inference on the full test set ---")
        inference_on_test_set(
            model_path=CONFIG["model_path"],
            feature_path=CONFIG["feature_path"],
            label_path=CONFIG["label_path"],
            output_file=CONFIG["inference_output_path"],
            **CONFIG["dl_model_config"]
        )
        predictions = np.load(inference_output_path)
        
        # --- Cleanup dummy files ---
        print("\nCleaning up dummy label files...")
        if labels_path.exists():
            os.remove(labels_path)
            os.remove(f"{labels_path}.json")
            print(f"Removed dummy label files.")

    print(f"Factor predictions ready, shape: {predictions.shape}")


    # --- Step 3. Portfolio Optimization ---
    print("\n--- Step 3: Running portfolio optimization ---")
    n_dates_preds, n_instruments_preds = predictions.shape
    
    # Use real date range and instrument count from metadata
    dates_preds = pd.date_range(start=START_DATE, periods=n_dates_preds, freq='B')
    instruments_preds = [f"INS_{i:04d}" for i in range(n_instruments_preds)]

    df_preds = pd.DataFrame(predictions, index=dates_preds, columns=instruments_preds)
    factor_data_pd = df_preds.stack().reset_index()
    factor_data_pd.columns = ['date', 'order_book_id', 'factor_value']
    factor_data_pd.dropna(inplace=True)
    factor_data_pl = pl.from_pandas(factor_data_pd)
    
    # Load market data from parquet file
    print(f"Loading market data from {MARKET_DATA_PATH}")
    stock_data_pl = pl.read_parquet(MARKET_DATA_PATH)
    
    # Ensure market data has 'date', 'order_book_id', 'close'
    required_market_cols = ['date', 'order_book_id', 'close']
    if not all(col in stock_data_pl.columns for col in required_market_cols):
        raise ValueError(f"Market data from {MARKET_DATA_PATH} missing required columns: {required_market_cols}")

    layered_optimizer = LayeredOptimizer(
        optimizer=EqualWeightOptimizer(),
        config=CONFIG["optimizer_config"]
    )
    
    optimized_weights_pl = layered_optimizer.optimize_layered_portfolio(
        stock_data=stock_data_pl, # Pass real market data
        factor_data=factor_data_pl,
        weight_col=CONFIG['backtest_config']['weight_col']
    )
    
    # Save optimized weights
    weights_output_path = RESULTS_DIR / 'optimized_weights.parquet'
    optimized_weights_pl.write_parquet(weights_output_path)
    print(f"Optimization complete. Weights saved to {weights_output_path}")

    # --- 4. Backtesting ---
    print("\n--- Step 4: Running backtest ---")

    class PrecomputedWeightStrategy(TopNFactorStrategy):
        def generate_signals_for_all_dates(self):
            return self.factor_data[['date', 'order_book_id', 'target_weight']]

    strategy = PrecomputedWeightStrategy(
        factor_data=optimized_weights_pl.to_pandas(), 
        config=CONFIG['backtest_config']
    )
    
    # Join the weights with market data to ensure backtest runs only on valid data
    backtest_market_data_pd = stock_data_pl.to_pandas()
    
    backtester = Backtester(
        strategy=strategy,
        data=backtest_market_data_pd,
        config=CONFIG['backtest_config']
    )
    
    backtester.run_backtest()
    print("Backtest finished.")
    
    # Save results
    portfolio_history = backtester.get_portfolio_history()
    if portfolio_history is not None and not portfolio_history.empty:
        history_output_path = RESULTS_DIR / 'portfolio_history.csv'
        portfolio_history.to_csv(history_output_path, index=False)
        print(f"Portfolio history saved to {history_output_path}")
    else:
        print("Backtest did not produce any history.")

    print("\nWorkflow finished successfully!")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()