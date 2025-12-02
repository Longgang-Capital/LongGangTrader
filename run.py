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
    from longgang_trader.optimize.optimizer import LayeredOptimizer, EqualWeightOptimizer, MeanVarianceOptimizer, RiskParityOptimizer, TopNOptimizer
    from longgang_trader.backtesting.backtester import Backtester, TopNFactorStrategy, GroupedFactorStrategy

    # Return imported modules to be used in main
    return {
        "inference_on_test_set": inference_on_test_set,
        "LayeredOptimizer": LayeredOptimizer,
        "EqualWeightOptimizer": EqualWeightOptimizer,
        "MeanVarianceOptimizer": MeanVarianceOptimizer,
        "RiskParityOptimizer": RiskParityOptimizer,
        "TopNOptimizer": TopNOptimizer,
        "Backtester": Backtester,
        "TopNFactorStrategy": TopNFactorStrategy,
        "GroupedFactorStrategy": GroupedFactorStrategy
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
    MeanVarianceOptimizer = modules["MeanVarianceOptimizer"]
    RiskParityOptimizer = modules["RiskParityOptimizer"]
    TopNOptimizer = modules["TopNOptimizer"]
    Backtester = modules["Backtester"]
    TopNFactorStrategy = modules["TopNFactorStrategy"]
    GroupedFactorStrategy = modules["GroupedFactorStrategy"]

    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RESULTS_DIR = DATA_DIR / 'results'
    MARKET_DATA_PATH = DATA_DIR / 'baostock_data_converted.parquet'

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
            "symbol_col": "code", # Use 'code' as the symbol column
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
    symbol_col_name = CONFIG['backtest_config']['symbol_col']
    date_col_name = CONFIG['backtest_config']['date_col']

    # Use real date range and instrument count from metadata
    with open("./data/test_ticker_list.json","r") as f:
        instruments_preds = json.load(f)
    with open("./data/test_tradedate_list.json","r") as f:
        dates_preds = json.load(f)
    # 正确转换日期格式：YYYYMMDD -> datetime
    dates_preds = pd.to_datetime(dates_preds, format='%Y%m%d')

    df_preds = pd.DataFrame(predictions, index=dates_preds, columns=instruments_preds)
    factor_data_pd = df_preds.stack().reset_index()
    factor_data_pd.columns = [date_col_name, symbol_col_name, 'factor_value']
    factor_data_pd.dropna(inplace=True)
    factor_data_pl = pl.from_pandas(factor_data_pd)
    del df_preds, factor_data_pd

    print(f"Loading market data from {MARKET_DATA_PATH}")

    # Read the pre-converted market data
    stock_data_pl = pl.read_parquet(MARKET_DATA_PATH)

    required_market_cols = [date_col_name, symbol_col_name, 'close']
    if not all(col in stock_data_pl.columns for col in required_market_cols):
        raise ValueError(f"Market data from {MARKET_DATA_PATH} missing required columns: {required_market_cols}")

    # Use TopNOptimizer for concentrated weights
    top_n_config = CONFIG["optimizer_config"].copy()
    top_n_config.update({
        "top_n": 50,  # Select top 50 stocks in each group
        "top_percentage": 0.2  # Select top 20% stocks based on factor value
    })

    layered_optimizer = LayeredOptimizer(
        optimizer=RiskParityOptimizer(top_n_config),
        config=CONFIG["optimizer_config"]
    )
    
    optimized_weights_pl = layered_optimizer.optimize_layered_portfolio(
        stock_data=stock_data_pl,
        factor_data=factor_data_pl,
        date_col=date_col_name,
        symbol_col=symbol_col_name,
        weight_col=CONFIG['backtest_config']['weight_col']
    )

    # Convert symbol format from standard (688069.SH) to baostock (sh.688069)
    def convert_symbol_format(symbol):
        '''Convert from standard format (688069.SH) to baostock format (sh.688069)'''
        if '.' in symbol:
            code, exchange = symbol.split('.')
            if exchange == 'SH':
                return f'sh.{code}'
            elif exchange == 'SZ':
                return f'sz.{code}'
        return symbol

    optimized_weights_pl = optimized_weights_pl.with_columns([
        pl.col(symbol_col_name).map_elements(convert_symbol_format, return_dtype=pl.Utf8).alias(symbol_col_name)
    ])

    weights_output_path = RESULTS_DIR / 'optimized_weights.parquet'
    optimized_weights_pl.write_parquet(weights_output_path)
    print(f"Optimization complete. Weights saved to {weights_output_path}")

    # --- 4. Backtesting ---
    print("\n--- Step 4: Running grouped backtest ---")
    
    backtest_config = CONFIG['backtest_config'].copy()
    # Let the strategy know which column the optimizer used for weights
    backtest_config['optimized_weight_col'] = CONFIG['backtest_config']['weight_col']

    strategy = GroupedFactorStrategy(
        factor_data=optimized_weights_pl.to_pandas(), 
        config=backtest_config
    )
    
    backtester = Backtester(
        strategy=strategy,
        data_path=str(MARKET_DATA_PATH),  # Use the converted data file
        config=backtest_config
    )
    
    group_results = backtester.run_grouped_backtest(save_plots=True, results_dir=str(RESULTS_DIR))

    if group_results:
        for group, results in group_results.items():
            portfolio_history = results['portfolio_history']
            metrics = results['metrics']
            
            print(f"\n----- Group {group} Metrics -----")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            if portfolio_history is not None and not portfolio_history.empty:
                history_output_path = RESULTS_DIR / f'portfolio_history_group_{group}.csv'
                portfolio_history.to_csv(history_output_path, index=False)
                print(f"Portfolio history for group {group} saved to {history_output_path}")
        print("\nGrouped backtest finished successfully!")
    else:
        print("Grouped backtest did not produce any results.")

    print("\nWorkflow finished successfully!")



if __name__ == "__main__":
    main()