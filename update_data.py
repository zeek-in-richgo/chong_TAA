import os
from os.path import basename, dirname, join as p_join
from datetime import timedelta, datetime, date, time
from glob import glob
import argparse
import sys
from shutil import *

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sqlalchemy import insert
from sqlalchemy import Table, Column, Integer, String, Float, DateTime, MetaData
from sqlalchemy.orm import sessionmaker
from pangres import upsert
from xbbg import blp
import pytz
import matplotlib.pyplot as plt
import yaml
import wget
import pandabase

from utils.data_manipulation import *
from utils.data_load import *


def update_undl_by_fnguide(configuration, connect_to_my_own_db, korean_closing_time):
    csv_path = configuration['fnguide_idx_urls_csv_path']
    downloaded_excel_dir = configuration['fnguide_idx_excel_dir']
    symbol2url = pd.Series(pd.read_csv(csv_path, index_col=['symbols'])['urls'])
    if os.path.exists(downloaded_excel_dir):
        rmtree(downloaded_excel_dir)
    historicals_by_fnguide = pd.read_sql("""select e_v.* from underlying_variables as e_v  
	            join underlying_constants as e_c 		 on e_v.undl_ticker = e_c.undl_ticker
                join fundamental_undl_constants as f_u_c on e_c.fundamental_undl_ticker = f_u_c.fundamental_undl_ticker
                    where f_u_c.data_vendor = 'fnGuide';""",
                                         con=connect_to_my_own_db,
                                         index_col=['undl_ticker', 'base_dt'],
                                         parse_dates=["base_dt"])
                                         #parse_dates={"base_dt": {"format": "%y/%m/%d %h:%M:%s"}})
    fails_to_wget_urls = []
    fails_to_wget_symbols = []
    will_be_appended = pd.DataFrame()
    for i, symbol in enumerate(set(historicals_by_fnguide.index.get_level_values(0))):
        url = symbol2url[symbol]
        try:
            tmp_path = wget.download(url)
            xlsx_path = p_join(downloaded_excel_dir, symbol + '_' + tmp_path)
            os.makedirs(dirname(xlsx_path), exist_ok=True)
            move(tmp_path, xlsx_path)
            prices = pd.read_excel(xlsx_path, engine='openpyxl', sheet_name='IndexValue',
                                   header=4, index_col=0, parse_dates=True)
            prices = prices.apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
            prices.index += korean_closing_time
            #prices.index.name = historicals_by_fnguide.loc[symbol].index.name
            new_data = prices.join(historicals_by_fnguide.loc[symbol])[['종가', 'point', 'point_in_krw']]
            new_data = new_data.fillna(method='ffill', axis=1)[['point', 'point_in_krw']]
            new_data = make_index_multiple_with_ticker(new_data, symbol, ['undl_ticker', 'base_dt'])
            will_be_appended = pd.concat([will_be_appended, new_data])
            if will_be_appended.size > 0:
                upsert(engine=connect_to_my_own_db,
                       df=will_be_appended,
                       table_name='underlying_variables',
                       if_row_exists='update')
            else:
                print('!!! Nothing is updated for UNDL')
        except:
            fails_to_wget_symbols.append(symbol)
            fails_to_wget_urls.append(url)
    with open('./fails_to_wget_urls.csv', 'w') as fh:
        print('\n'.join(fails_to_wget_urls), file=fh)
    with open('./fails_to_wget_symbols.csv', 'w') as fh:
        print('\n'.join(fails_to_wget_symbols), file=fh)
    print()

def update_undl_by_bloomberg(undl_constant_infos,
                             configuration,
                             today_param, TZ_param, korean_closing_time):
    will_be_appended = pd.DataFrame()

    initial_dt = date_str2datetime(configuration['from_date_str'])
    FX_tickers = configuration['FX_tickers']
    FXs = get_new_FX(FX_tickers, initial_dt, today_param, TZ_param, korean_closing_time)
    if FXs is None:
        sys.exit()

    for undl_ticker, undl_infos in undl_constant_infos.iterrows():
        if undl_ticker in set_undl_last_dts.index:
            from_dt = set_undl_last_dts.loc[undl_ticker, 'last_dt'] + timedelta(days=1)
        else:
            from_dt = initial_dt
        try:
            this_closing_t = str2timedelta(undl2t_end_times[undl_ticker])
        except:
            print('Error, ', undl_ticker)
        today_param = datetime.combine(date.today(), time(0)) + this_closing_t

        df = blp.bdh(undl_ticker, "px_last", from_dt, today_param, configuration['TZ_param']).dropna()
        df.index += this_closing_t
        if df.size > 0:
            df.index = df.index.tz_localize('US/Eastern')
            df.columns = df.columns.get_level_values(1)
            FX_quote = undl2crncies[undl_ticker] + 'KRW'
            df.index.names = ['base_dt']
            df.rename(columns={'px_last': 'point'}, inplace=True)
            new_data = get_foreigns_in_KRW(df, FXs, FX_quote, undl_ticker)
            will_be_appended = pd.concat([will_be_appended, new_data])
    if will_be_appended.size > 0:
        upsert(engine=connect_to_my_own_db,
               df=will_be_appended,
               table_name='underlying_variables',
               if_row_exists='update')
    else:
        print('!!! Nothing is updated for UNDL')
    FXs.to_csv('./csvs/FXs.csv')


def update_etf(etf_constant_infos, undl_constant_infos, set_etf_last_dts):
    will_be_appended = pd.DataFrame()
    num_of_new_etf = 0

    for etf_isin, etf_info in etf_constant_infos.iterrows():
        etf_ticker = etf_info['etf_ticker']
        if etf_info['undl_ticker']  in undl_constant_infos.index:
            if etf_isin in set_etf_last_dts.index:
                from_dt = set_etf_last_dts.loc[etf_isin, 'last_dt'] + timedelta(days=1)
            else:
                from_dt = etf_info['etf_inception_dt']
                num_of_new_etf += 1
                print('New ETF {}! \t{}\t{}'.format(num_of_new_etf, etf_ticker, from_dt))

            new_data = blp.bdh(etf_ticker, ['FUND_NET_ASSET_VAL', "PX_LAST"],
                            from_dt, today_param, configuration['TZ_param']).dropna()
            if new_data.size > 0:
                new_data = preprocess_raw_etf_bdh(new_data, korean_closing_time, etf_isin)
                will_be_appended = pd.concat([will_be_appended, new_data])

    will_be_appended.to_csv('./update_data_etf.csv')
    if will_be_appended.size > 0:
        upsert(engine=connect_to_my_own_db,
               df=will_be_appended,
               table_name='etf_variables',
               if_row_exists='update')
    else:
        print('!!! Nothing is updated for ETF')

if __name__ == '__main__':
    with open(os.getcwd() + '/check_execution.txt', 'w') as fh:
        print(datetime.now(), file=fh)

    pymysql.install_as_MySQLdb()
    parser = argparse.ArgumentParser(description = "update new information")
    parser.add_argument('--config', type=str,   default='./config.yaml',   help='Config YAML file')
    parser.add_argument('--korean_closing_time', type=timedelta, default=timedelta(hours=2, minutes=30))
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r", encoding='UTF8') as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
    to_date_str = datetime.today().strftime('%Y-%m-%d')
    TZ_param = configuration['TZ_param']
    korean_closing_time = args.korean_closing_time

    #df = pd.read_csv(configuration['etf_infos_csv_path'])

    ### connect Database
    db_info = configuration['database_info']
    my_own_db_info = db_info['my_etf_db']
    connect_to_my_own_db = create_engine(get_sqlalchemy_conn_str(my_own_db_info, 'etf_management'), echo=True)

    today_param = datetime.now().strftime('%Y-%m-%d')
    #today_param = '2015-01-01'

    etf_constant_infos = pandabase.read_sql('etf_constants', con=connect_to_my_own_db)
    undl_constant_infos = pandabase.read_sql('underlying_constants', con=connect_to_my_own_db)
    fundamental_undl_constant_infos = pandabase.read_sql('fundamental_undl_constants', con=connect_to_my_own_db)

    ## append new ETF information and undls
    undl2infos = undl_constant_infos.loc[:, ['currency', 'trading_end_time']].to_dict()
    undl2crncies = undl2infos['currency']
    undl2t_end_times = undl2infos['trading_end_time']
    set_undl_last_dts = pd.read_sql('SELECT undl_ticker, max(base_dt) as last_dt \
                                   FROM etf_management.underlying_variables group by undl_ticker',
                                connect_to_my_own_db).set_index('undl_ticker')
    set_etf_last_dts = pd.read_sql('SELECT isin, max(base_dt) as last_dt \
                                   FROM etf_management.etf_variables group by isin',
                                connect_to_my_own_db).set_index('isin')

    groups_by_data_vendor = fundamental_undl_constant_infos.groupby('data_vendor').groups
    for data_vendor, fundamental_undls in groups_by_data_vendor.items():
        fundamental_undls = fundamental_undl_constant_infos.loc[fundamental_undls].index
        undl_constant_infos_from_this_data_vendor = fundamental_undls2undl_constant_infos(fundamental_undls,
                                                                                          undl_constant_infos)
        etf_constant_infos_from_this_data_vendor = fundamental_undls2etf_constant_infos(fundamental_undls,
                                                                                        undl_constant_infos,
                                                                                        etf_constant_infos)
        if data_vendor == 'bloomberg':
            update_undl_by_bloomberg(undl_constant_infos_from_this_data_vendor,
                                    configuration,
                                    today_param, TZ_param, korean_closing_time)
            update_etf(etf_constant_infos_from_this_data_vendor,
                       undl_constant_infos,
                       set_etf_last_dts)
            print()
        elif data_vendor == 'fnGuide':
            will_be_appended = pd.DataFrame()
            df = pd.read_csv('./csvs/fnguide_index_historicals_2001_to_2021-08-21.csv',
                             index_col=0,
                             dtype='Float64',
                             parse_dates=['date'],
                             dayfirst=False,
                             infer_datetime_format=True)    #.set_index('date')
            df.index += korean_closing_time
            df.index.names = ['base_dt']
            for i, undl_id in  enumerate(undl_constant_infos_from_this_data_vendor.index):

                this_data = pd.DataFrame({'point': df[undl_id].dropna(),
                                          'point_in_krw': df[undl_id].dropna()})
                this_data.index = pd.MultiIndex.from_tuples([(undl_id, idx) for idx in this_data.index],
                                                            names=['undl_ticker', 'base_dt'])
                if undl_id == 'FI00.WLT.EXP':
                    print()
                will_be_appended = pd.concat([will_be_appended, this_data])
            upsert(engine=connect_to_my_own_db,
                   df=will_be_appended,
                   table_name='underlying_variables',
                   if_row_exists='update')

            #update_undl_by_fnguide(configuration, connect_to_my_own_db, korean_closing_time)
            update_etf(etf_constant_infos_from_this_data_vendor,
                       undl_constant_infos,
                       set_etf_last_dts)
            print()
        elif data_vendor == 'krx':
            pass
        else:
            print('Invalid Data Vendor:', data_vendor)
            raise ValueError

#    will_be_appended = pd.DataFrame()
#    num_of_new_etf = 0
#
#    for etf_isin, etf_info in etf_constant_infos.iterrows():
#        etf_ticker = etf_info['etf_ticker']
#        if etf_info['undl_ticker']  in undl_constant_infos.index:
#        #if etf_info['undl_ticker'] + ' Index' in undl_constant_infos.index:
#            if etf_isin in set_etf_last_dts.index:
#                from_dt = set_etf_last_dts.loc[etf_ticker, 'last_dt'] + timedelta(days=1)
#            else:
#                from_dt = etf_info['etf_inception_dt']
#                num_of_new_etf += 1
#                print('New ETF {}! \t{}\t{}'.format(num_of_new_etf, etf_ticker, from_dt))
#
#            new_data = blp.bdh(etf_ticker, ['FUND_NET_ASSET_VAL', "PX_LAST"],
#                            from_dt, today_param, configuration['TZ_param']).dropna()
#            if new_data.size > 0:
#                new_data = preprocess_raw_etf_bdh(new_data, korean_closing_time, etf_isin)
#                will_be_appended = pd.concat([will_be_appended, new_data])
#
#    will_be_appended.to_csv('./update_data_etf.csv')
#    if will_be_appended.size > 0:
#        #pdbase_to_sql(will_be_appended, connect_to_my_own_db, 'etf_variables')
#        upsert(engine=connect_to_my_own_db,
#               df=will_be_appended,
#               table_name='etf_variables',
#               if_row_exists='ignore')
#    else:
#        print('!!! Nothing is updated for ETF')
    print()

#    ## append new underlying information
#    undl2infos = undl_constant_infos.loc[:, ['currency', 'trading_end_time']].to_dict()
#    undl2crncies = undl2infos['currency']
#    undl2t_end_times = undl2infos['trading_end_time']
#    set_undl_last_dts = pd.read_sql('SELECT ticker, max(base_dt) as last_dt \
#                                   FROM etf_management.underlying_variables group by ticker',
#                                connect_to_my_own_db).set_index('ticker')
#    will_be_appended = pd.DataFrame()
#
#    initial_dt = date_str2datetime(configuration['from_date_str'])
#    FX_tickers = configuration['FX_tickers']
#    FXs = get_new_FX(FX_tickers, initial_dt, today_param, TZ_param, korean_closing_time)
#    if FXs is None:
#        sys.exit()
#
#    for undl_ticker, undl_infos in undl_constant_infos.iterrows():
#        if undl_ticker in set_undl_last_dts.index:
#            from_dt = set_undl_last_dts.loc[undl_ticker, 'last_dt'] + timedelta(days=1)
#        else:
#            from_dt = initial_dt
#        try:
#            this_closing_t = str2timedelta(undl2t_end_times[undl_ticker])
#        except:
#            print('Error, ', undl_ticker)
#        today_param = datetime.combine(date.today(), time(0)) + this_closing_t
#
#        df = blp.bdh(undl_ticker, "px_last", from_dt, today_param, configuration['TZ_param']).dropna()
#        df.index += this_closing_t
#        if df.size > 0:
#            df.index = df.index.tz_localize('US/Eastern')
#            df.columns = df.columns.get_level_values(1)
#            FX_quote = undl2crncies[undl_ticker] + 'KRW'
#            df.index.names = ['base_dt']
#            df.rename(columns={'px_last': 'point'}, inplace=True)
#            new_data = get_foreigns_in_KRW(df, FXs, FX_quote, undl_ticker)
#            will_be_appended = pd.concat([will_be_appended, new_data])
#    #will_be_appended.to_csv('./undl_variables.csv')
#    if will_be_appended.size > 0:
#        #pdbase_to_sql(will_be_appended, connect_to_my_own_db, 'underlying_variables')
#        upsert(engine=connect_to_my_own_db,
#               df=will_be_appended,
#               table_name='underlying_variables',
#               if_row_exists='ignore')
#    else:
#        print('!!! Nothing is updated for UNDL')
#    FXs.to_csv('./csvs/FXs.csv')
    print()
