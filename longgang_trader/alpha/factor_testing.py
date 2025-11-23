import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FactorTester:

    def __init__(self,
                 factor_data: pd.DataFrame,
                 return_data: pd.DataFrame,
                 industry_data: Optional[pd.DataFrame] = None,
                 weight_data: Optional[pd.DataFrame] = None):
        """
        Parameters
        ----------
        factor_data : pd.DataFrame
            index: [datetime, instrument], columns: ['factor']
        return_data : pd.DataFrame
            收益率，columns: ['ret_next']
        industry_data : pd.DataFrame, optional
            行业分类
        weight_data : pd.DataFrame, optional
            已有持仓权重（用于计算实际换手率）
        """
        self.factor = factor_data.copy()
        self.ret = return_data.copy()
        self.industry = industry_data.copy() if industry_data is not None else None
        self.weight = weight_data.copy() if weight_data is not None else None

        # 合并主数据
        self.data = self.factor.join(self.ret, how='inner')
        if self.mv is not None:
            self.data = self.data.join(self.mv, how='left')
        if self.industry is not None:
            self.data = self.data.join(self.industry, how='left')

        self.dates = sorted(self.data.index.get_level_values(0).unique())
        print(f"因子检验初始化完成，共 {len(self.dates)} 个交易日，{self.data.index.get_level_values(1).nunique()} 只股票")

    def neutralize(self, factor_series: pd.Series) -> pd.Series:
        """行业 + 市值中性化（截面回归去残差）"""
        df = pd.DataFrame({'factor': factor_series})
        if 'industry' in self.data.columns:
            df = df.join(self.data['industry'])
            dummies = pd.get_dummies(df['industry'], drop_first=True)
            X = pd.concat([df[['mv']].apply(lambda x: np.log(x + 1)), dummies], axis=1)
        else:
            X = df[['mv']].apply(np.log1p)

        X = sm.add_constant(X)
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = df.loc[X.index, 'factor']

        if len(X) < 10 or X.shape[1] >= len(X):
            return factor_series  # 数据不足不中性

        model = sm.OLS(y, X, missing='drop').fit()
        residual = y - model.predict(X)
        result = factor_series.copy()
        result.loc[residual.index] = residual
        result = result.fillna(method='ffill').fillna(0)
        return result

    def calculate_ic(self, neutral: bool = False) -> Dict[str, Any]:
        """计算IC和RankIC序列"""
        ic_list = []
        rank_ic_list = []

        for date in self.dates:
            df_day = self.data.xs(date, level=0).dropna(subset=['factor', 'ret_next'])

            if len(df_day) < 50:
                continue

            factor = df_day['factor']
            ret = df_day['ret_next']

            if neutral and 'mv' in df_day.columns:
                factor = self.neutralize(factor)

            ic = pearsonr(factor, ret)[0]
            rank_ic = spearmanr(factor, ret)[0]

            ic_list.append(ic)
            rank_ic_list.append(rank_ic if not np.isnan(rank_ic) else 0)

        ic_series = pd.Series(ic_list, index=self.dates[:len(ic_list)])
        rank_ic_series = pd.Series(rank_ic_list, index=self.dates[:len(rank_ic_list)])

        return {
            'ic': ic_series,
            'rank_ic': rank_ic_series,
            'ic_mean': ic_series.mean(),
            'ic_std': ic_series.std(),
            'ic_ir': ic_series.mean() / ic_series.std() if ic_series.std() != 0 else np.nan,
            'rank_ic_mean': rank_ic_series.mean(),
            'rank_ic_ir': rank_ic_series.mean() / rank_ic_series.std() if rank_ic_series.std() != 0 else np.nan,
        }

    def layered_backtest(self, n_groups: int = 5, neutral: bool = True) -> Dict[str, Any]:
        """分层回测（等权或市值加权）"""
        group_rets = {i: [] for i in range(n_groups)}
        long_short_rets = []

        for date in self.dates:
            df_day = self.data.xs(date, level=0).dropna(subset=['factor', 'ret_next'])

            if len(df_day) < n_groups * 10:
                continue

            factor = df_day['factor']
            if neutral and 'mv' in df_day.columns:
                factor = self.neutralize(factor)

            df_day = df_day.loc[factor.index]
            df_day['group'] = pd.qcut(factor, n_groups, labels=False, duplicates='drop')

            for g in range(n_groups):
                sub = df_day[df_day['group'] == g]
                if len(sub) == 0:
                    continue
                ret_eq = sub['ret_next'].mean()
                group_rets[g].append(ret_eq)

            # 多空组合
            ret_long = df_day[df_day['group'] == n_groups - 1]['ret_next'].mean()
            ret_short = df_day[df_day['group'] == 0]['ret_next'].mean()
            long_short_rets.append(ret_long - ret_short)

        results = {}
        for g in group_rets:
            rets = pd.Series(group_rets[g])
            cum = (1 + rets).cumprod()
            results[f'group_{g+1}'] = {
                'ret_series': rets,
                'cum_ret': cum,
                'annual_ret': (cum.iloc[-1] ** (252 / len(rets)) - 1) if len(rets) > 0 else np.nan,
                'sharpe': rets.mean() / rets.std() * np.sqrt(252) if rets.std() != 0 else np.nan,
                'maxdrawdown': (cum / cum.cummax() - 1).min()
            }

        ls_ret = pd.Series(long_short_rets)
        ls_cum = (1 + ls_ret).cumprod()
        results['long_short'] = {
            'ret_series': ls_ret,
            'cum_ret': ls_cum,
            'annual_ret': (ls_cum.iloc[-1] ** (252 / len(ls_ret)) - 1),
            'sharpe': ls_ret.mean() / ls_ret.std() * np.sqrt(252),
            'maxdrawdown': (ls_cum / ls_cum.cummax() - 1).min()
        }

        return results

    def plot_ic_decay(self, max_lag: int = 20):
        """IC衰减图"""
        decay = {}
        for lag in range(1, max_lag + 1):
            shifted_ret = self.ret.groupby(level=1)['ret_next'].shift(-lag + 1)
            merged = self.factor.join(shifted_ret.rename('ret_lag'), how='inner')
            daily_ic = merged.groupby(level=0).apply(lambda x: spearmanr(x['factor'], x['ret_lag'])[0] if len(x) > 10 else np.nan)
            decay[lag] = daily_ic.mean()

        plt.figure(figsize=(12, 6))
        pd.Series(decay).plot(kind='bar')
        plt.title('因子IC衰减图（Rank IC）')
        plt.xlabel('预测未来N日')
        plt.ylabel('Rank IC')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_layered_returns(self, results: Dict, title: str = "分层回测累计收益"):
        """绘制分层收益曲线"""
        plt.figure(figsize=(14, 8))
        for i in range(5):
            if f'group_{i+1}' in results:
                cum = results[f'group_{i+1}']['cum_ret']
                plt.plot(cum.index, cum, label=f'Group {i+1}', linewidth=2)

        ls_cum = results['long_short']['cum_ret']
        plt.plot(ls_cum.index, ls_cum, label='Long-Short', color='black', linewidth=3, linestyle='--')

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def calculate_factor_returns(self, method: str = "ls_equal") -> pd.Series:
      
        factor_rets = []

        for date in self.dates:
            df_day = self.data.xs(date, level=0).dropna(subset=['factor', 'ret_next'])

            if len(df_day) < 100:  # 样本太少跳过
                factor_rets.append(0.0)
                continue

            factor = df_day['factor']

            # 可选：中性化处理（推荐）
            if 'mv' in df_day.columns and 'industry' in df_day.columns:
                factor = self.neutralize(factor)

            df_day = df_day.loc[factor.index]

            # 按因子值排序
            df_day = df_day.sort_values('factor')

            n = len(df_day)
            top_n = max(1, n // 10)   # 前10%
            bot_n = max(1, n // 10)   # 后10%

            if method == "ls_equal":
                ret_long = df_day.iloc[-top_n:]['ret_next'].mean()
                ret_short = df_day.iloc[:bot_n]['ret_next'].mean()
                daily_ret = ret_long - ret_short

            elif method == "ls_cap":
                mv = df_day['mv']
                ret_long = np.average(df_day.iloc[-top_n:]['ret_next'], weights=mv.iloc[-top_n:])
                ret_short = np.average(df_day.iloc[:bot_n]['ret_next'], weights=mv.iloc[:bot_n])
                daily_ret = ret_long - ret_short

            elif method == "long_only_ew":
                daily_ret = df_day.iloc[-top_n:]['ret_next'].mean()

            elif method == "long_only_cw":
                mv = df_day['mv']
                daily_ret = np.average(df_day.iloc[-top_n:]['ret_next'], weights=mv.iloc[-top_n:])

            else:
                raise ValueError(f"Unknown method: {method}")

            factor_rets.append(daily_ret)

        factor_ret_series = pd.Series(factor_rets, index=self.dates)

        # 缓存到实例，方便后续绘图
        self.factor_returns = factor_ret_series
        self.factor_cum_returns = (1 + factor_ret_series).cumprod()

        return factor_ret_series

    def plot_ic_series(self, neutral: bool = True, kind: str = "rank_ic"):
        """
        绘制IC时间序列图（带均值线、±1σ区间）
        """
        ic_result = self.calculate_ic(neutral=neutral)
        ic_series = ic_result['rank_ic'] if kind == "rank_ic" else ic_result['ic']
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()

        plt.figure(figsize=(14, 6))
        ic_series.plot(color='steelblue', alpha=0.8, linewidth=1)

        plt.axhline(mean_ic, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_ic:.4f}')
        plt.axhline(mean_ic + std_ic, color='gray', linestyle=':', alpha=0.7)
        plt.axhline(mean_ic - std_ic, color='gray', linestyle=':', alpha=0.7)

        plt.fill_between(ic_series.index,
                         mean_ic - std_ic,
                         mean_ic + std_ic,
                         color='gray', alpha=0.15, label='±1σ区间')

        plt.title(f"因子每日Rank IC序列 {'（行业市值中性）' if neutral else ''}", fontsize=16, fontweight='bold')
        plt.ylabel("Rank IC")
        plt.xlabel("日期")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"IC均值: {mean_ic:.4f}   |   IC_IR: {mean_ic/std_ic:.3f}")

    def plot_cumulative_factor_returns(self, method: str = "ls_equal", title: str = None):
        """
        绘制因子累计收益率曲线
        """
        if not hasattr(self, 'factor_returns') or self.factor_returns is None:
            self.calculate_factor_returns(method=method)

        cum_ret = self.factor_cum_returns

        # 绩效统计
        annual_ret = cum_ret.iloc[-1] ** (252 / len(cum_ret)) - 1
        sharpe = self.factor_returns.mean() / self.factor_returns.std() * np.sqrt(252)
        max_dd = (cum_ret / cum_ret.cummax() - 1).min()
        calmar = annual_ret / (-max_dd) if max_dd < 0 else np.nan

        plt.figure(figsize=(15, 8))
        cum_ret.plot(color='darkorange', linewidth=2.5)

        plt.title(title or f"因子累计收益率曲线\n"
                          f"年化收益: {annual_ret:.2%}  |  Sharpe: {sharpe:.2f}  |  MaxDD: {max_dd:.2%}  |  Calmar: {calmar:.2f}",
                  fontsize=16, fontweight='bold')
        plt.ylabel("累计收益")
        plt.xlabel("日期")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"因子收益统计（{method}）".center(50, "-"))
        print(f"年化收益率   : {annual_ret:.2%}")
        print(f"年化波动率   : {self.factor_returns.std() * np.sqrt(252):.2%}")
        print(f"Sharpe比率   : {sharpe:.2f}")
        print(f"最大回撤     : {max_dd:.2%}")
        print(f"Calmar比率   : {calmar:.2f}")
        print(f"多空年化收益 : {self.factor_returns.mean() * 252:.2%}")
    
    def run_all_tests(self,
                      neutral: bool = True,
                      n_groups: int = 5,
                      factor_ret_method: str = "ls_equal",
                      show_plots: bool = True) -> Dict[str, Any]:
        print("开始因子全面检验".center(60, "="))
        results = {}

        # 1. IC 分析（原始 + 中性）
        print(f"\n1. 计算 IC 与 RankIC {'（市值中性化）' if neutral else ''}")
        ic_raw = self.calculate_ic(neutral=False)
        ic_neu = self.calculate_ic(neutral=neutral)
        results['ic_raw'] = ic_raw
        results['ic_neutralized'] = ic_neu

        # 2. 分层回测
        print(f"\n2. 执行 {n_groups} 组分层回测")
        layer_result = self.layered_backtest(n_groups=n_groups, neutral=neutral)
        results['layered'] = layer_result

        # 3. 因子收益率（多空对冲）
        print(f"\n3. 计算因子收益率序列（{factor_ret_method}）")
        factor_ret_series = self.calculate_factor_returns(method=factor_ret_method)
        results['factor_returns'] = factor_ret_series
        results['factor_cum_returns'] = self.factor_cum_returns

        # 4. 可视化输出
        if show_plots:
            
            # 图1：IC时间序列
            self.plot_ic_series(neutral=neutral, kind="rank_ic")
            
            # 图2：分层收益曲线
            self.plot_layered_returns(layer_result, 
                                    title=f"AI多因子模型 - {n_groups}组分层回测 {'（行业市值中性）' if neutral else ''}")
            
            # 图3：因子累计收益（多空对冲）
            title = (f"因子多空对冲累计收益曲线\n"
                     f"方法: {factor_ret_method} | "
                     f"中性化: {'是' if neutral else '否'}")
            self.plot_cumulative_factor_returns(method=factor_ret_method, title=title)
            
            # 图4：IC衰减
            self.plot_ic_decay(max_lag=20)

        # 5. 输出核心绩效总表
        print("\n" + " 因子核心绩效总表 ".center(60, "="))
        
        ls_ann_ret = layer_result['long_short']['annual_ret']
        ls_sharpe = layer_result['long_short']['sharpe']
        ls_mdd = layer_result['long_short']['maxdrawdown']

        fr_ann_ret = (self.factor_cum_returns.iloc[-1] ** (252 / len(factor_ret_series)) - 1)
        fr_sharpe = factor_ret_series.mean() / factor_ret_series.std() * np.sqrt(252)
        fr_mdd = (self.factor_cum_returns / self.factor_cum_returns.cummax() - 1).min()

        print(f"{'指标':<15} {'原始RankIC':<12} {'中性RankIC':<12} {'多空年化':<10} {'多空Sharpe':<10} {'最大回撤':<10}")
        print("-" * 70)
        print(f"{'值':<15} "
              f"{ic_raw['rank_ic_mean']:.4f}     "
              f"{ic_neu['rank_ic_mean']:.4f}     "
              f"{fr_ann_ret:.2%}    "
              f"{fr_sharpe:.2f}     "
              f"{fr_mdd:.2%}")
        print(f"{'IR/t-value':<15} "
              f"{ic_raw['rank_ic_ir']:.2f}        "
              f"{ic_neu['rank_ic_ir']:.2f}        "
              f"{'-':<10} "
              f"{'-':<10} "
              f"{'-':<10}")

        print(f"\n多空组合（{factor_ret_method}）年度表现：")
        print(f"   年化收益     : {fr_ann_ret:.2%}")
        print(f"   Sharpe比率   : {fr_sharpe:.2f}")
        print(f"   最大回撤     : {fr_mdd:.2%}")
        print(f"   Calmar比率   : {fr_ann_ret / (-fr_mdd):.2f}" if fr_mdd < 0 else "   Calmar比率   : N/A")

        print("因子检验完成".center(60, "="))

        return results