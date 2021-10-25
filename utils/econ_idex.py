import pandas as pd

def get_eco_indicator_ticker2info():
    eco_indicator_ticker2info = {}
    eco_indicator_tickers = ['OEKRKLAC Index', 'NAPMPMI Index',  'NAPMNEWO_NAPMINV', \
                             'RWKELCP Index', 'RWKELCS Index',  \
                             'usyieldspread_10y_2y Index', 'usyieldspread_10y_2y_normalized Index',
                             'ISRATIO Index',
                             'OEOTKLAC Index', 'OEUSKLAC Index']
    for eco_indicator_ticker in eco_indicator_tickers:
        info = {}
        if eco_indicator_ticker == 'OEKRKLAC Index':
            info['name'] = 'Korea OECD Leading Indicators'
            info['period_freq'] = 'm'
            info['release_adjust_amount'] = 2
            info['to_timestamp_freq'] = 'M'
            info['upper_bound'] = 100
            info['lower_bound'] = 100
        elif eco_indicator_ticker == 'NAPMPMI Index':
            info['name'] = 'ISM Manufacturing PMI SA'
            info['period_freq'] = 'm'
            info['release_adjust_amount'] = 1
            info['to_timestamp_freq'] = 'M'
            info['upper_bound'] = 50
            info['lower_bound'] = 50
        elif eco_indicator_ticker == 'NAPMNEWO_NAPMINV':
            info['name'] = 'ISM Manufacturing (New Orders - Inventories)'
            info['period_freq'] = 'm'
            info['release_adjust_amount'] = 1
            info['to_timestamp_freq'] = 'M'
            info['upper_bound'] = 0
            info['lower_bound'] = 0
        elif eco_indicator_ticker == 'RWKELCP Index':
            info['name'] = 'RICHGO Export Leading Indicator(preliminary)'
            info['period_freq'] = 'w'
            info['release_adjust_amount'] = 1
            info['to_timestamp_freq'] = 'W'
            info['upper_bound'] = +0.7
            info['lower_bound'] = -0.7
        elif eco_indicator_ticker == 'RWKELCS Index':
            info['name'] = 'RICHGO Export Leading Indicator(second)'
            info['period_freq'] = 'w'
            info['release_adjust_amount'] = 1
            info['to_timestamp_freq'] = 'W'
            info['upper_bound'] = +0.7
            info['lower_bound'] = -0.7
        elif eco_indicator_ticker == 'usyieldspread_10y_2y Index':
            info['name'] = 'US Treasury 10Y-2Y yield spread'
            info['period_freq'] = 'd'
            info['release_adjust_amount'] = 2
            info['to_timestamp_freq'] = 'D'
            info['upper_bound'] = -1.3481
            info['lower_bound'] = -1.3481
        elif eco_indicator_ticker == 'usyieldspread_10y_2y_normalized Index':
            info['name'] = 'US Treasury 10Y-2Y yield spread(normalized)'
            info['period_freq'] = 'd'
            info['release_adjust_amount'] = 2
            info['to_timestamp_freq'] = 'D'
            info['upper_bound'] = 0
            info['lower_bound'] = 0
        elif eco_indicator_ticker == 'ISRATIO Index':
            info['name'] = 'Total Business_ Inventories to Sales Ratio'
            info['period_freq'] = 'm'
            info['release_adjust_amount'] = 3
            info['to_timestamp_freq'] = 'M'
            info['upper_bound'] = 1.35
            info['lower_bound'] = 1.35
        elif eco_indicator_ticker == 'OEOTKLAC Index':
            info['name'] = 'Total OECD Leading Indicators'
            info['period_freq'] = 'm'
            info['release_adjust_amount'] = 2
            info['to_timestamp_freq'] = 'M'
            info['upper_bound'] = 100
            info['lower_bound'] = 100
        elif eco_indicator_ticker == 'OEUSKLAC Index':
            info['name'] = 'US OECD Leading Indicators'
            info['period_freq'] = 'm'
            info['release_adjust_amount'] = 2
            info['to_timestamp_freq'] = 'M'
            info['upper_bound'] = 100
            info['lower_bound'] = 100
        else:
            print('Error! Invalid eco_indicator_ticker: ', eco_indicator_ticker)
            raise ValueError
        eco_indicator_ticker2info[eco_indicator_ticker] = info

    return eco_indicator_ticker2info

def adjust_econ_idx_date_to_release_date(eco_indicator_info, eco_indicator_ticker, trade_signals):
    ## OECD CLI
    if eco_indicator_ticker == 'OEKRKLAC Index':
        period_freq = eco_indicator_info['period_freq']
        added = eco_indicator_info['release_adjust_amount']
        trade_signals.index = (pd.PeriodIndex(trade_signals.index, freq=period_freq) + added).to_timestamp(
                                to_timestamp_freq)
    ## ISM PMI (ISM Manufacturing)
    elif eco_indicator_ticker == 'NAPMPMI Index':
        trade_signals.index = trade_signals.index + pd.offsets.MonthBegin(1)
    ## ISM PMI (ISM Manufacturing New Orders - ISM Manufacturing Inventories)
    elif eco_indicator_ticker == 'NAPMNEWO_NAPMINV':
        trade_signals.index = trade_signals.index + pd.offsets.MonthBegin(1)

    return trade_signals