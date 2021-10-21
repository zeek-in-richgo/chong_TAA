from os.path import basename, dirname, join as p_join

import numpy as np
import pandas as pd

#from utils.data_manipulation import *

try:
    from xbbg import blp
    bloomberg_available = True
except:
    bloomberg_available = False


def get_eq(table_name, conn):
    table_len = get_sql_table_len(table_name, conn)
    if table_len == 0:
        eq_df = pd.read_sql('SELECT * FROM ' + table_name, conn)
        eq_df.set_index('etf_isin', inplace=True)
        return eq_df
    else:
        return pandabase.read_sql(table_name, conn)

def get_hdf5_key(ticker):
    return 'b_ticker_' + '_'.join(ticker.split())

def get_historical_data(ticker_prefix, data_dir):
    file_name = '_'.join(ticker_prefix.split()) + '.hdf5'
    hdf5_path_to_have_historical_data = p_join(data_dir, file_name)
    return pd.read_hdf(hdf5_path_to_have_historical_data, get_hdf5_key(ticker_prefix))

def get_usdkrw(from_date, to_date_str, TZ_param, bloomberg_available, price_csv_dir):
    usd_ticker_x = 'USDKRW BOKR Curncy'
    usd_ticker_y = 'USDKRW Curncy'
    if bloomberg_available:
        x = blp.bdh(usd_ticker_x, 'px_last', from_date, to_date_str, TZ_param)
        y = blp.bdh(usd_ticker_y, 'px_last', from_date, to_date_str, TZ_param)
        x.columns = x.columns.get_level_values(0)
        y.columns = y.columns.get_level_values(0)
    else:
        path_x = p_join(price_csv_dir, usd_ticker_x + '.csv')
        path_y = p_join(price_csv_dir, usd_ticker_y + '.csv')
        x = pd.read_csv(path_x, index_col=0)[usd_ticker_x]
        y = pd.read_csv(path_y, index_col=0)[usd_ticker_y]
        x.index = pd.to_datetime(x.index)
        y.index = pd.to_datetime(y.index)
        x = pd.DataFrame({usd_ticker_x: x})
        y = pd.DataFrame({usd_ticker_y: y})
    z = pd.concat([x,y], axis=1)
    merged = z.apply(lambda x: x[usd_ticker_y] if np.isnan(x[usd_ticker_x]) else x[usd_ticker_x], axis=1)
    merged.sort_index(inplace=True)
    return pd.DataFrame(merged).rename(columns={0: usd_ticker_y})

def get_new_FX(FX_tickers,
           from_date, to_date_str, TZ_param, korean_closing_time, bloomberg_available):
    base_fx = get_usdkrw(from_date, to_date_str, TZ_param, bloomberg_available)
    if base_fx.size == 0:
        return None
    else:
        base_fx.columns = base_fx.columns.get_level_values(0)
        base_fx.index = base_fx.index.tz_localize('US/Eastern')
        base_fx.index += korean_closing_time
        base_fx.rename(lambda x: x.split()[0], axis='columns', inplace=True)

        tickers = ['USD{} Curncy'.format(pair) for pair in FX_tickers['pairs']]
        pairs = []
        for ticker in tickers:
            # from 10 days before base_fx: to have enough data
            pair_fx = blp.bdh(ticker, 'px_last', from_date - timedelta(days=10), to_date_str, TZ_param)
            pair_fx.columns = pair_fx.columns.get_level_values(0)
            pair_fx.index = pair_fx.index.tz_localize('US/Eastern')
            pair_fx.index += time2timedelta(blp.bdp(ticker, 'TRADING_DAY_END_TIME_EOD').iloc[0, 0])
            pair_fx = pair_fx.apply(lambda x: 1 / x, axis=1)
            pair_fx.rename(columns={ticker: ticker[3:6] + ticker[:3]}, inplace=True)
            pairs.append(pair_fx)
        pairs_fx = pd.concat(pairs, axis=1).fillna(method='pad')
        concated = pd.concat([base_fx, pairs_fx], axis=1).fillna(method='pad')
        filtered = concated.filter(items=base_fx.index, axis=0)
        FXs = filtered.apply(lambda x: (x * filtered['USDKRW'])
        if x.name.endswith("USD") else x)
        FXs.rename(lambda fx_quote: fx_quote[:3] + 'KRW' if fx_quote.endswith('USD')
                                                        else fx_quote, axis=1, inplace=True)
        return FXs


def get_underlying_prices(etf_info, undl_historicals):
    undl_ticker = get_undl_ticker(etf_info)
    if etf_info['CRNCY'] == 'KRW':
        return undl_historicals.loc[undl_ticker, 'point']
    elif etf_info['CURRENCY_HEDGED_INDICATOR'] == 'N':
        return undl_historicals.loc[undl_ticker, 'point']


def get_underlying_prices(info, FXs, fill_method='backfill'):
    u_prices_by_its_crncy = pd.read_hdf(get_hdf_path(info['ETF_UNDL_INDEX_TICKER']))
    u_prices_by_its_crncy.columns = pd.Index(['underlying_prices'], dtype=str)
    currency = info['CRNCY']
    FX_quote = currency + 'KRW'
    is_hedged = info['CURRENCY_HEDGED_INDICATOR']
    if currency == 'KRW':
        return u_prices_by_its_crncy
    elif is_hedged == 'Y':
        return u_prices_by_its_crncy
    elif (FX_quote in FXs.columns) and (is_hedged == 'N'):
        concated = pd.concat([u_prices_by_its_crncy, FXs[FX_quote]], axis=1).fillna(method=fill_method)
        filtered = concated.filter(items=u_prices_by_its_crncy.index, axis=0)
        bm_krw = filtered.apply(lambda x: x['underlying_prices'] * x[FX_quote], axis=1)
        return pd.DataFrame({'underlying_prices': bm_krw}, index=filtered.index)
    else:
        print('Unsupported CRNCY', currency, ' in ', info['ETF_ticker'])
        raise Exception
