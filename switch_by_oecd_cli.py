import os
from datetime import timedelta, datetime, date
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

import yaml

from utils.data_load import *
from utils.data_manipulation import *
from utils.draw import *
from utils.econ_idex import *
from utils.strategy import *
from utils.stat_backtesting import *


def get_trade(df, my_money):
    my_quantity = pd.DataFrame()
    recession_periods = []
    recession_start_dt = None
    for i, (dt, row) in enumerate(df.iterrows()):
        if row['trade_boom'] == -1:
            my_money = number_to_buy_value * row['asset_in_boom']
        elif row['trade_recession'] == -1:
            my_money = number_to_buy_growth * row['asset_in_recession']

        if row['trade_recession'] == 1:
            number_to_buy_growth = my_money / row['asset_in_recession']
            number_to_buy_value = 0
            recession_start_dt = dt
        elif row['trade_boom'] == 1:
            number_to_buy_growth = 0
            number_to_buy_value = my_money / row['asset_in_boom']
            if recession_start_dt is not None:
                recession_periods.append((recession_start_dt, dt))
        my_quantity = pd.concat([my_quantity,
                                 pd.DataFrame({'quantity_boom': number_to_buy_value,
                                               'quantity_recession': number_to_buy_growth},
                                              index=[dt])],
                                axis=0)
    return my_quantity, recession_periods

def get_chong_switching(historicals, trade_signals, my_money):
    df = pd.concat([historicals, trade_signals], axis=1).sort_index()
    #df0 = df.fillna(columns=['asset_in_boom', 'asset_in_recession'])
    ## filling method를 bfill로 하는 이유는 OECD경기선행지수가 휴일에 나왔을 경우
    ## 휴일에 거래할 수 없으므로 원래대로라면 익 영업일에 거래해야히지만
    ## 구현상의 편의를 위해서 그냥 휴일에 익영업일 가격으로 거래했다고 해도 무방함, 그래서 bfill
    df0 = df[['asset_in_boom', 'asset_in_recession']].fillna(method='bfill')
    #df1 = df.dropna(subset=['trade_boom', 'trade_recession'])
    df1 = df.dropna(subset=['trade_boom', 'trade_recession'])
    historicals_trade_signals = adjust_to_100(df1, df1.index[0], subset=['asset_in_boom', 'asset_in_recession'])
    #tmp= pd.concat([df0, df1[['trade_boom', 'trade_recession']]], axis=1)
    #historicals_trade_signals = pd.concat([df0, df1[['trade_boom', 'trade_recession']]], axis=1).dropna()
    #historicals_trade_signals.to_csv('historicals_trade_signals.csv')
    initial_trade_date = historicals_trade_signals.index[0]
    positions, recession_periods = get_trade(historicals_trade_signals, my_money)
    #positions.to_csv('positions.csv')
    historicals_positions = pd.concat([historicals, positions], axis=1)
    historicals_positions = historicals_positions.loc[initial_trade_date:]
    historicals_positions = historicals_positions.fillna(method='pad')
    historicals_positions = adjust_to_100(historicals_positions, historicals_positions.index[0], subset=['asset_in_boom', 'asset_in_recession'])
    estimates = []
    for i, row in historicals_positions.iterrows():
        estimate = row['asset_in_boom'] * row['quantity_boom'] + \
                    row['asset_in_recession'] * row['quantity_recession']
        estimates.append(estimate)
    estimates = pd.Series(estimates, index=historicals_positions.index, name='my_portfolio')
    chong_taa = pd.concat([historicals_positions, estimates], axis=1)
    return chong_taa, initial_trade_date, recession_periods

##def asset2tickers_names():
def assets2infos():
    assets2infos = {
     'us_stock_us_bond':
        {'tickers': ['SPX Index', 'LUATTRUU Index'],
          'names': ['S&P500', 'US Treasury Total Return'],
         'FX_quotes': ['USDKRW', "USDKRW"],
         'class': ['asset_index', 'asset_index']
         }
    }
    return assets2infos

def get_FXs(usdkrw):
    return usdkrw.rename(lambda x: x.split()[0], axis='columns')

def get_foreigns_in_KRW(historical_df, FXs, FX_quote):
    column_name = historical_df.columns[0]
    if FX_quote == 'KRWKRW':
        historical_df['point_in_krw'] = historical_df.loc[:, column_name]
        historicals = historical_df
    else:
        #concated = pd.concat([historical_df, FXs[FX_quote]], axis=1).fillna(method='backfill')
        concated = pd.concat([historical_df, FXs[FX_quote]], axis=1).fillna(method='pad')
        filtered = concated.filter(items=historical_df.index, axis=0)
        historical_in_krw = filtered.apply(lambda x: x[column_name] * x[FX_quote], axis=1)
        historical_in_krw = pd.DataFrame({'point_in_krw': historical_in_krw}, index=filtered.index)  #.dropna()
        historicals = pd.concat([historical_df, historical_in_krw], axis=1).dropna()
    return historicals

def interest_rate2asset_idx(historicals, infos):
    new_names = get_new_names(infos)
    new_df = pd.DataFrame()
    for i, cc in enumerate(infos['class']):
        if cc == 'interest_rate':
            i_rate_column = historicals.columns[i]
            #new_column_name = i_rate_column + '_index'
            new_column_name = new_names[i]
            df = 1 + historicals[i_rate_column] / 100 / 365
            #df.rename('daily_rate', axis='columns', inplace=True)
            #concated = pd.concat([historicals, df], axis=1)
            new_df =  pd.concat([new_df, pd.DataFrame({new_column_name: df.cumprod(axis=0)})], axis=1)
        elif cc == 'asset_index':
            name = infos['names'][i]
            new_df = pd.concat([new_df, historicals[name]], axis=1)
    return new_df

def get_new_names(infos):
    new_names = []
    for i, cc in enumerate(infos['class']):
        if cc == 'asset_index':
            new_names.append(infos['names'][i])
        elif cc == 'interest_rate':
            new_names.append(infos['names'][i] + '_index')
    return new_names


def get_eco_indicator_name(oecd_cli_csv):
    if basename(oecd_cli_csv) == 'OECD전체.csv':
        return 'OECD CLI(OECD_Total)'
    elif basename(oecd_cli_csv) == '미국.csv':
        return 'OECD CLI(US)'

#def get_stats(my_portfolio):
#    dd = 1 - my_portfolio / np.maximum.accumulate(my_portfolio)
#    pass

if __name__ == '__main__':
    with open('config.yaml', "r", encoding='UTF8') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    from_date_str =  configuration['from_date_str']
    korean_closing_time = timedelta(hours=2, minutes=30)
    to_date_str = datetime.today().strftime('%Y-%m-%d')
    FX_tickers = configuration['FX_tickers']
    TZ_param = configuration['TZ_param']
    price_csv_dir = configuration['price_csv_dir_path']

    try:
        from xbbg import blp
        if blp.bdp('KOSPI2 Index', "name").index.size == 0:
            bloomberg_available = False
        else:
            bloomberg_available = True
    except:
        bloomberg_available = False

    #FXs = get_new_FX(FX_tickers,
    #                 date_str2datetime(from_date_str),
    #                 to_date_str, TZ_param, korean_closing_time,
    #                 bloomberg_available)
    usdkrw = get_usdkrw(from_date_str, to_date_str, TZ_param, bloomberg_available, price_csv_dir)
    FXs = get_FXs(usdkrw)

    # when add econ-indicator, should add "eco_indicator_tickers, eco_indicator_name, thresholds, added, period_freq"
    eco_indicator_ticker2info = get_eco_indicator_ticker2info()

    assets2infos = assets2infos()
    asset_keys = ['us_stock_us_bond']
    print(1)
    for asset_key in asset_keys[:1]:
    #for asset_key in asset_keys[1:2]:   #[:1]:
        print(2, asset_key)
        infos = assets2infos[asset_key]
        boom_idx_ticker, recession_idx_ticker = infos['tickers']
        boom_FX_quote, recession_FX_quote = infos['FX_quotes']
        boom_name, recession_name = get_new_names(infos)
        if bloomberg_available:
            historicals = blp.bdh([boom_idx_ticker, recession_idx_ticker], 'px_last',
                                  from_date_str, to_date_str, TZ_param)
            historicals.columns = historicals.columns.get_level_values(0)
        else:
            path_boom = p_join(price_csv_dir, boom_idx_ticker.split()[0] + '.csv')
            path_recession = p_join(price_csv_dir, recession_idx_ticker.split()[0] + '.csv')
            historicals_boom = pd.read_csv(path_boom, index_col=0)  #[usd_ticker_x]
            historicals_recession = pd.read_csv(path_recession, index_col=0)    #[usd_ticker_y]
            historicals = pd.concat([historicals_boom, historicals_recession], axis=1)
            historicals = historicals.sort_index().fillna(method='ffill')
        #historicals.index = historicals.index.tz_localize('US/Eastern')
        historicals.rename(columns={boom_idx_ticker: boom_name,
                                    recession_idx_ticker: recession_name}, inplace=True)
        historicals.index = pd.to_datetime(historicals.index)
        historicals = fill_index(historicals)
        historicals = interest_rate2asset_idx(historicals, assets2infos[asset_key])
        boom_df = get_foreigns_in_KRW(pd.DataFrame(historicals[boom_name]), FXs, boom_FX_quote)['point_in_krw']
        recession_df = get_foreigns_in_KRW(pd.DataFrame(historicals[recession_name]), FXs, recession_FX_quote)['point_in_krw']
        historicals_in_local_FX = historicals
        historicals = pd.DataFrame({'asset_in_boom':  boom_df,
                                    'asset_in_recession': recession_df})
        adjusted_historicals = adjust_to_100(historicals, '1997-01-01')
        historicals = adjusted_historicals
        print()

        #thresholds = [100, 50, 0, 0, 0, -1.3481, 0, 1.35]
        #for i, eco_indicator_ticker in enumerate(eco_indicator_ticker2info.keys()):
        for oecd_cli_csv in glob('./OECD_CLI/CLI/*.csv'):
            plt.rcParams['font.family'] = 'Malgun Gothic'
            fig = plt.figure(figsize=(20, 18))
            #eco_indicator_info = eco_indicator_ticker2info[eco_indicator_ticker]
            eco_indicator_name = get_eco_indicator_name(oecd_cli_csv)
            print(eco_indicator_name, oecd_cli_csv)
            upper_bound = 100
            lower_bound = 100


            df = pd.read_csv(oecd_cli_csv, index_col=0, parse_dates=True)
            df.index.name = 'release_date'
            trade_signals_1 =  strategy_oecd_cli_1(df['0'])
            trade_signals_2 =  strategy_oecd_cli_2(df[['0', '1', '2']])

            # execute trades
            my_money = 100
            chong_taa_1, initial_trade_date_1, recession_periods_1 = get_chong_switching(historicals, trade_signals_1, my_money)
            chong_taa_2, initial_trade_date_2, recession_periods_2 = get_chong_switching(historicals, trade_signals_2, my_money)
            initial_trade_date = initial_trade_date_2

            # compare to S&P 500
            bm_name = 'S&P 500 Index'
            bm_ticker = 'SPX Index'
            if bloomberg_available:
                benchmark_historicals = blp.bdh(bm_ticker, 'px_last', from_date_str, to_date_str, TZ_param)
                benchmark_historicals.columns = benchmark_historicals.columns.get_level_values(0)
            else:
                path_kospi2 = p_join(price_csv_dir, '{}.csv'.format(bm_ticker.split()[0]))
                benchmark_historicals = pd.read_csv(path_kospi2, index_col=0)
                benchmark_historicals.index = pd.to_datetime(benchmark_historicals.index)
            benchmark_historicals = get_foreigns_in_KRW(pd.DataFrame(benchmark_historicals), FXs, 'USDKRW')['point_in_krw']
            benchmark_historicals = pd.DataFrame({bm_ticker: benchmark_historicals})
            benchmark_historicals = fill_index(benchmark_historicals)

            kospi_chong_1 = pd.concat([chong_taa_1['my_portfolio'], benchmark_historicals], axis=1).dropna()
            kospi_chong_2 = pd.concat([chong_taa_2['my_portfolio'], benchmark_historicals], axis=1).dropna()
            kospi_chong_1 = adjust_to_100(kospi_chong_1, kospi_chong_1.index[0])
            kospi_chong_2 = adjust_to_100(kospi_chong_2, kospi_chong_2.index[0])

            bm_stats = get_stats(benchmark_historicals['SPX Index'].loc['2008-03-07':])
            str1_stats = get_stats(kospi_chong_1['my_portfolio'])
            str2_stats = get_stats(kospi_chong_2['my_portfolio'])
            print('--' * 10, "bm_stats")
            print(bm_stats)
            print('')
            print('--' * 10, "strategy 1 stats")
            print(str1_stats)
            print('')
            print('--' * 10, "strategy 2 stats")
            print(str2_stats)

            #### Draw
            draw_simple_general(fig, kospi_chong_1, recession_periods_1,
                        boom_name, recession_name,
                        eco_indicator_name,  upper_bound, lower_bound, asset_key,
                        bm_name = bm_name, bm_ticker = bm_ticker, strategy_id='1')
            plt.clf()
            csv_path = './csvs_oecd_cli/{}/taa_1_{}_strategy{}.csv'.format(eco_indicator_name, asset_key, '1')
            os.makedirs(dirname(csv_path), exist_ok=True)
            chong_taa_1.to_csv(csv_path, encoding="utf-8-sig")

            draw_simple_general(fig, kospi_chong_2, recession_periods_2,
                        boom_name, recession_name,
                        eco_indicator_name, upper_bound, lower_bound, asset_key,
                        bm_name=bm_name, bm_ticker=bm_ticker, strategy_id='2')
            plt.clf()
            csv_path = './csvs_oecd_cli/{}/taa_1_{}_strategy{}.csv'.format(eco_indicator_name, asset_key, '2')
            os.makedirs(dirname(csv_path), exist_ok=True)
            chong_taa_2.to_csv(csv_path, encoding="utf-8-sig")

            #draw_full(fig, chong_taa_1, chong_taa_2,
            #        recession_periods_1, recession_periods_2,
            #        boom_name, recession_name,
            #        df, eco_indicator_name, upper_bound, lower_bound, asset_key)
            plt.close()
            print()
    print()
