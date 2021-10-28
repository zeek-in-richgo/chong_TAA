import numpy as np
import pandas as pd

""" Output Format
    Start                     2004-08-19 00:00:00
    End                       2013-03-01 00:00:00
    Duration                   3116 days 00:00:00
    Exposure Time [%]                     93.9944
    Equity Final [$]                      51959.9
    Equity Peak [$]                       75787.4
    Return [%]                            419.599
    Buy & Hold Return [%]                 703.458
    Return (Ann.) [%]                      21.328
    Volatility (Ann.) [%]                 36.5383
    Sharpe Ratio                         0.583718
    Sortino Ratio                         1.09239
    Calmar Ratio                         0.444518
    Max. Drawdown [%]                    -47.9801
    Avg. Drawdown [%]                    -5.92585
    Max. Drawdown Duration      584 days 00:00:00
    Avg. Drawdown Duration       41 days 00:00:00
    # Trades                                   65
    Win Rate [%]                          46.1538
    Best Trade [%]                         53.596
    Worst Trade [%]                      -18.3989
    Avg. Trade [%]                        2.35371
    Max. Trade Duration         183 days 00:00:00
    Avg. Trade Duration          46 days 00:00:00
    Profit Factor                         2.08802
    Expectancy [%]                        8.79171
    SQN                                  0.916893
    _strategy                            SmaCross
    _equity_curve                           Eq...
    _trades                       Size  EntryB...
    dtype: object
"""
def compute_drawdown_duration_peaks(dd: pd.Series):
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())
    df = df[df['iloc'] > df['prev'] + 1].astype(int)

    # If no drawdown since no trade, avoid below for pandas sake and return nan series
    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
    df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
    df = df.reindex(dd.index)
    return df['duration'], df['peak_dd']

def geometric_mean(returns: pd.Series) -> float:
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1

def get_stats(my_portfolio):
    risk_free_rate: float = 0
    equity  = my_portfolio
    idx = my_portfolio.index

    dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=idx))

    s = pd.Series(dtype=object)
    s.loc['Start'] = idx[0]
    s.loc['End'] = idx[-1]
    s.loc['Duration'] = s.End - s.Start
    s.loc['Return [%]'] = (equity[-1] - equity[0]) / equity[0] * 100


    day_returns = equity.resample('D').last().dropna().pct_change()
    gmean_day_return = geometric_mean(day_returns)
    annual_trading_days = float(
        365 if idx.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * .6 else
        252)

    annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1

    s.loc['Return (Ann.) [%]'] = annualized_return * 100
    s.loc['Volatility (Ann.) [%]'] = np.sqrt(
        (day_returns.var(ddof=int(bool(day_returns.shape))) + (1 + gmean_day_return) ** 2) ** annual_trading_days - (
                    1 + gmean_day_return) ** (2 * annual_trading_days)) * 100  # noqa: E501
    # s.loc['Return (Ann.) [%]'] = gmean_day_return * annual_trading_days * 100
    # s.loc['Risk (Ann.) [%]'] = day_returns.std(ddof=1) * np.sqrt(annual_trading_days) * 100

    # Our Sharpe mismatches `empyrical.sharpe_ratio()` because they use arithmetic mean return
    # and simple standard deviation
    s.loc['Sharpe Ratio'] = np.clip(
        (s.loc['Return (Ann.) [%]'] - risk_free_rate) / (s.loc['Volatility (Ann.) [%]'] or np.nan), 0,
        np.inf)  # noqa: E501
    # Our Sortino mismatches `empyrical.sortino_ratio()` because they use arithmetic mean return
    s.loc['Sortino Ratio'] = np.clip((annualized_return - risk_free_rate) / (
                np.sqrt(np.mean(day_returns.clip(-np.inf, 0) ** 2)) * np.sqrt(annual_trading_days)), 0,
                                     np.inf)  # noqa: E501
    max_dd = -np.nan_to_num(dd.max())
    s.loc['Calmar Ratio'] = np.clip(annualized_return / (-max_dd or np.nan), 0, np.inf)
    s.loc['Max. Drawdown [%]'] = max_dd * 100

    print()

