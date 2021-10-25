
from datetime import datetime
from datetime import timedelta, datetime, date
from urllib.parse import quote_plus

import pandas as pd
#import pandabase
from pytz import timezone, utc
#import sqlalchemy

#def pdbase_to_sql(df, conn, table_name):
#    assert isinstance(df, pd.DataFrame)
#    assert isinstance(conn, sqlalchemy.engine.base.Engine)
#    assert isinstance(table_name, str)
#    return pandabase.to_sql(df, con=conn, table_name=table_name, how='upsert')

def adjust_to_100(historicals, base_dt, subset=None):
    try:
        base_value = historicals.loc[base_dt]
    except:
        base_value = fill_index(historicals).loc[base_dt]
    adjusted_historicals = pd.DataFrame()
    if subset is None:
        set_columns = historicals.columns
    else:
        set_columns = subset
    for column in set_columns:
        adjusted_historicals[column] = historicals[column] / base_value[column] * 100
    for column in historicals.columns:
        if column not in set_columns:
            adjusted_historicals[column] = historicals[column]
    return adjusted_historicals

def fill_index(chong_taa_1):
    chong_taa_full_idx = pd.date_range(chong_taa_1.index[0], chong_taa_1.index[-1])
    return chong_taa_1.reindex(chong_taa_full_idx, method='pad')


def append_richgo_indices(richgo_indices, new_richgo_idx, korean_closing_time, etf_isin):
    new_richgo_idx.index += korean_closing_time
    new_df = pd.DataFrame({'nav': new_richgo_idx})
    new_df.index.name = 'base_dt'
    new_df.insert(0, 'etf_isin', etf_isin)
    new_df.reset_index(inplace=True)
    new_df.set_index(['etf_isin', 'base_dt'], inplace=True)
    return richgo_indices.append(new_df)

def fundamental_undls2undl_constant_infos(fundamental_undls, undl_constant_infos):
    return undl_constant_infos[undl_constant_infos['fundamental_undl_ticker'].isin(fundamental_undls)]

def fundamental_undls2etf_constant_infos(fundamental_undls, undl_constant_infos, etf_constant_infos):
    undls = undl_constant_infos[undl_constant_infos['fundamental_undl_ticker'].isin(fundamental_undls)]
    return etf_constant_infos[etf_constant_infos['undl_ticker'].isin(undls.index)]

def get_etf2undl2fundamental_undl(conn):
    query = '''select etf_management.etf_constants.etf_ticker,  etf_management.etf_constants.undl_ticker,  etf_management.underlying_constants.fundamental_undl_ticker from etf_management.etf_constants join etf_management.underlying_constants on  etf_management.etf_constants.undl_ticker = etf_management.underlying_constants.ticker'''
    return pd.read_sql(query, con=conn)

def get_sql_table_len(table_name, conn):
    count_df = pd.read_sql('SELECT COUNT(*) FROM ' + table_name, conn)
    return count_df.loc[0][0]

def date_str2datetime(from_date_str, tzinfo=timezone('UTC')):
    yyyy, mm, dd = [int(x) for x in from_date_str.split('-')]
    return datetime(yyyy, mm, dd, 0,0,0, tzinfo=tzinfo)

def str2timedelta(korean_str_represed_time):
    if korean_str_represed_time == '오후 12:00:00':
        return timedelta(hours=23, minutes=59, seconds=59)
    elif korean_str_represed_time == '오전 12:00:00':
        return timedelta(hours=11, minutes=59, seconds=59)
    prefix = korean_str_represed_time[:2]
    hours, minutes, seconds = [int(x) for x in korean_str_represed_time[2:].split(":")]
    assert 0 <= hours < 12
    assert 0 <= minutes < 60
    assert 0 <= seconds < 60
    if prefix == '오전':
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    elif prefix == '오후':
        return timedelta(hours=(hours + 12), minutes=minutes, seconds=seconds)
    else:
        print('Invalid TIME_STR:', korean_str_represed_time)
        raise ValueError

def get_sqlalchemy_conn_str(db_info, db_name):
    HOSTNAME = db_info['host']
    PORT = db_info['port']
    USERNAME = db_info['user']
    PASSWORD = db_info['pwd']
    DATABASE = db_name
    CHARSET1 = 'utf8'
    con_str_fmt = 'mysql+mysqldb://{0}:{1}@{2}:{3}/{4}?charset={5}'
    return con_str_fmt.format(USERNAME, quote_plus(PASSWORD), HOSTNAME, PORT, DATABASE, CHARSET1)

#def get_pd_timestamp_min_midnight():
#    our_datetime = datetime.combine(pd.Timestamp.min.date(), datetime.min.time()) + timedelta(days=1)

def get_pd_timestamp_min():
    return pd.Timestamp.min.tz_localize(tz='UTC')

def timedelta2datetime(timedelta_obj):
    return datetime.min + timedelta_obj

def timedelta2timestamp(timedelta_obj):
    return pd.Timestamp.min + timedelta_obj

def inverse_timedelta2datetime(datetime_obj):
    return datetime_obj - datetime.min

def inverse_timedelta2timestamp(timestamp_obj):
    return timestamp_obj - pd.Timestamp.min

def time2timedelta(time_obj):
    return datetime.combine(date.min, time_obj) - datetime.min

def str2datetime(str_obj):  # '오전 02:00:00
    return timedelta2datetime(str2timedelta(str_obj))

def str2timestamp(str_obj):  # '오전 02:00:00
    return timedelta2timestamp(str2timedelta(str_obj))

def str2pd_timestamp_min_added(str_obj):
    return get_pd_timestamp_min() + str2timedelta(str_obj)

def preprocess_undl_constants_for_db(undl_idx_constant_infos):
    # Column('ticker', String(32), primary_key=True),
    # Column('name', String(128)),
    # Column('currency', String(8)),
    # Column('asset_class', String(64)),
    # Column('class_1', String(8)),
    # Column('class_2', String(8)),
    # Column('leverage_ratio', Integer),
    # Column('country', String(8)),
    # Column('data_vendor', String(32)),
    # Column('trading_end_time', DateTime),
    undl_idx_constant_infos.rename(columns={'CRNCY': 'currency',
                                   'COUNTRY': 'country',
                                   'TRADING_DAY_END_TIME_EOD': 'trading_end_time'}, inplace=True)
    #undl_idx_constant_infos.loc[:, 'trading_end_time'] = \
    #        undl_idx_constant_infos['trading_end_time'].apply(str2pd_timestamp_min_added)
    #undl_idx_constant_infos.drop(labels='if_bl_provide_or_not', axis=1, inplace=True)
    undl_idx_constant_infos['leverage_ratio'] = undl_idx_constant_infos['leverage_ratio'].astype(int)
    return undl_idx_constant_infos.set_index('undl_ticker')

def preprocess_etf_constants_columns_for_db(etf_constant_infos, undl_idx_constant_infos):
    etf_constant_infos.rename(columns={'ETF bloomberg Ticker': 'etf_ticker',
                                        'ACTIVELY_MANAGED': 'active_or_not',
                                        'CURRENCY_HEDGED_INDICATOR': 'currency_hedged_or_not',
                                        'FUND_ASSET_CLASS_FOCUS': 'asset_class',
                                        'ETF_UNDL_INDEX_TICKER': 'undl_ticker',
                                        'ID_ISIN': 'isin',
                                        'FUND_INCEPT_DT': 'etf_inception_dt'}, inplace=True)
    etf_constant_infos =  etf_constant_infos.dropna().set_index('isin')
    #new_idx = ~etf_constant_infos['undl_ticker'].str.startswith('#N/A')
    #etf_constant_infos = etf_constant_infos[new_idx]
    #df = etf_constant_infos.set_index('undl_ticker')
    #df2 = df.index.intersection(undl_idx_constant_infos.index)
    return etf_constant_infos[etf_constant_infos['undl_ticker'].isin(undl_idx_constant_infos.index)]

def make_index_multiple_with_ticker(df, ticker, names):
    try:
        df.index = pd.MultiIndex.from_tuples([(ticker, idx) for idx in df.index])
        df.index.names = names #['undl_ticker', 'base_dt']
    except:
        print('Error!!')
        print(df)
        print('-' * 10)
        print(df.index)
        print('==' * 10)
        print(ticker)
        raise Exception
    return df

#def get_foreigns_in_KRW(historical_df, column_name, FXs, FX_quote):
def get_foreigns_in_KRW(historical_df, FXs, FX_quote, ticker):
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
    return make_index_multiple_with_ticker(historicals, ticker, ['undl_ticker', 'base_dt'])


def preprocess_raw_etf_bdh(new_data, korean_closing_time, etf_isin):
    new_data.index += korean_closing_time
    new_data.index = pd.MultiIndex.from_tuples([(etf_isin, dt) for dt in new_data.index],
                                               names=['isin', 'base_dt'])
    new_data.columns = new_data.columns.get_level_values(1)
    new_data = new_data.rename(columns={'FUND_NET_ASSET_VAL': 'nav',
                                        'PX_LAST': 'price'})
    print()
    return new_data.dropna()