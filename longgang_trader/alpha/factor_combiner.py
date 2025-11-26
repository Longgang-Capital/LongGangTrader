import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def load_data(preds_path: str, returns_path: str, univ_path: str, tradedates_path: str, instruments: list, end_di: int, ii_max: int):
    """
    Load all necessary data for weight calculation.
    """
    # Load predictions (assuming .npy format from dl_model_factor.py)
    preds_full = np.load(preds_path)
    
    # Pad predictions matrix to match full instrument dimension
    new_preds = np.full((preds_full.shape[0], ii_max), np.nan, dtype=float)
    new_preds[:, :preds_full.shape[1]] = preds_full

    # Load trading dates
    tradedate_df = pd.read_csv(tradedates_path, skiprows=[1])
    tradedate_list = tradedate_df[(tradedate_df['TradingDate'] <= 20250926)]['TradingDate'].to_list()
    
    # Ensure the length of dates matches the predictions
    tradedate_list = tradedate_list[-preds_full.shape[0]:]

    # Create predictions DataFrame
    df_preds = pd.DataFrame(new_preds, index=tradedate_list, columns=instruments)

    # Load returns
    # WARNING: The original notebook used sarray.load. We are assuming the .bin file
    # is a flat binary file that can be read by np.fromfile and reshaped.
    # This might fail if the file has a different structure.
    try:
        y_returns = np.fromfile(returns_path, dtype=np.float32).reshape(len(tradedate_list), ii_max)
    except Exception as e:
        print(f"ERROR: Could not load returns file '{returns_path}'. np.fromfile failed: {e}")
        print("This script requires returns to be in a flat binary format of float32.")
        raise

    df_returns = pd.DataFrame(y_returns, index=tradedate_list, columns=instruments)

    # Load universe
    # WARNING: Assuming the universe is a flat binary file of booleans.
    try:
        univ_tradable = np.fromfile(univ_path, dtype=bool).reshape(len(tradedate_list), ii_max)
    except Exception as e:
        print(f"ERROR: Could not load universe file '{univ_path}'. np.fromfile failed: {e}")
        print("This script requires the universe to be in a flat binary format of booleans.")
        raise
    
    # Apply universe to returns
    df_returns[~univ_tradable] = np.nan
    
    return df_preds, df_returns

def calculate_daily_weights(df_preds: pd.DataFrame, rolling_window: int = 5, std_dev_multiplier: float = 1.5):
    """
    Calculate daily weights based on prediction scores.
    """
    # Calculate rolling mean and std dev
    df_preds_rolling5 = df_preds.rolling(rolling_window).mean()
    df_preds_rolling5_std = df_preds.rolling(rolling_window).std()

    # 1) Create mask for stocks to trade
    mask = (df_preds > df_preds_rolling5 + std_dev_multiplier * df_preds_rolling5_std) & (df_preds > 0)

    # 2) Set non-tradable predictions to 0 for sum calculation
    positive_preds = df_preds.where(mask, 0.0)

    # 3) Sum predictions per day
    row_sum = positive_preds.sum(axis=1)
    
    # Avoid division by zero
    row_sum = row_sum.replace(0, np.nan)

    # 4) Calculate weights
    weights = positive_preds.div(row_sum, axis=0)

    # 5) Set non-tradable weights back to NaN
    weights = weights.where(mask, np.nan)
    
    return weights

def calculate_performance_metrics(daily_returns, trading_days_per_year=252):
    """
    Calculates performance metrics for a portfolio.
    """
    equity_curve = (1 + daily_returns).cumprod()
    
    n_days = len(daily_returns)
    if n_days == 0:
        return {
            "Annualized Return": 0,
            "Annualized Vol": 0,
            "Sharpe Ratio": 0,
            "Max Drawdown": 0,
            "Calmar Ratio": 0,
        }

    # Annualized return
    annual_return = equity_curve.iloc[-1]**(trading_days_per_year / n_days) - 1

    # Annualized volatility
    annual_vol = daily_returns.std() * np.sqrt(trading_days_per_year)

    # Sharpe ratio
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan

    # Max drawdown
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    return {
        "Annualized Return": annual_return,
        "Annualized Vol": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate portfolio weights and performance from model predictions.")
    parser.add_argument("--preds_path", type=str, required=True, help="Path to the predictions file (e.g., preds_full.npy)")
    parser.add_argument("--returns_path", type=str, required=True, help="Path to the returns file (e.g., v2v_ret_5d.bin)")
    parser.add_argument("--univ_path", type=str, required=True, help="Path to the tradable universe file (e.g., valid.bin)")
    parser.add_argument("--tradedates_path", type=str, required=True, help="Path to the tradedates file (e.g., tradedates.txt)")
    # These would come from a sim environment, so we pass them as args
    parser.add_argument("--iimax", type=int, required=True, help="Maximum number of instruments (iiMax)")
    parser.add_argument("--enddi", type=int, required=True, help="End date index (enddi)")

    args = parser.parse_args()

    # A mock instrument list is needed. In a real scenario, this would be loaded.
    instruments = [f'ins_{i}' for i in range(args.iimax)]

    # Load data
    df_preds, df_returns = load_data(
        preds_path=args.preds_path,
        returns_path=args.returns_path,
        univ_path=args.univ_path,
        tradedates_path=args.tradedates_path,
        instruments=instruments,
        end_di=args.enddi,
        ii_max=args.iimax,
    )

    # Calculate weights
    weights = calculate_daily_weights(df_preds)
    
    # Calculate portfolio daily returns
    portfolio_daily_returns = (weights * df_returns).sum(axis=1)

    # Filter for a specific period for performance calculation, e.g., from 2023
    start_date = '20230101'
    portfolio_daily_returns_filtered = portfolio_daily_returns.loc[portfolio_daily_returns.index.astype(str) >= start_date]

    # Calculate and print performance
    perf_metrics = calculate_performance_metrics(portfolio_daily_returns_filtered)
    
    for metric, value in perf_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plotting can be added here if needed, e.g. using matplotlib
    # (portfolio_daily_returns_filtered.cumsum() + 1).plot(figsize=(20,6), title="Equity Curve")
    # import matplotlib.pyplot as plt
    # plt.show()


if __name__ == "__main__":
    # This is an example of how to run it.
    # The paths are placeholders and would need to be provided via command line.
    # python factor_combiner.py --preds_path /path/to/preds_full.npy --returns_path /path/to/v2v_ret_5d.bin \
    # --univ_path /path/to/valid.bin --tradedates_path /path/to/tradedates.txt --iimax 5500 --enddi 1392
    main()