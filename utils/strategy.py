
import numpy as np
import pandas as pd

#def strategy_1_old(df, eco_indicator_ticker, threshold):
#    trade_signals = pd.DataFrame()
#    positions = pd.DataFrame()
#    num_of_trade = 0
#    values = []
#    growths = []
#    indices = []
#    for i, (dt, eco_indicator) in enumerate(df.iterrows()):
#        #print()
#        indicator_value = eco_indicator[eco_indicator_ticker, 'px_last']
#        if i == 0:
#            indices.append(dt)
#            if indicator_value > threshold:
#                values.append(1)
#                growths.append(0)
#            else:
#                values.append(0)
#                growths.append(1)
#        else:
#            if indicator_value > threshold and pre_indicator_value <= threshold:
#                indices.append(dt)
#                values.append(1)
#                growths.append(-1)
#                #print('switch to value , ', pre_indicator_value, indicator_value, dt)
#            elif indicator_value <= threshold and pre_indicator_value > threshold:
#                indices.append(dt)
#                values.append(-1)
#                growths.append(1)
#                #print('switch to growth, ', pre_indicator_value, indicator_value, dt)
#        pre_indicator_value = indicator_value
#    values = pd.Series(values, index=indices)
#    growths = pd.Series(growths, index=indices)
#    trade_signals =  pd.DataFrame({'trade_boom': values, 'trade_recession': growths})
#    #print()
#
#    return trade_signals, 1

def strategy_u_bound_l_bound(df, eco_indicator_ticker, upper_bound, lower_bound):
    trade_signals = pd.DataFrame()
    positions = pd.DataFrame()
    num_of_trade = 0
    values = []
    growths = []
    indices = []
    for i, (dt, eco_indicator) in enumerate(df.iterrows()):
        #print()
        indicator_value = eco_indicator[eco_indicator_ticker, 'px_last']
        if indicator_value > upper_bound:
            if len(growths) == 0:
                growths.append(0)
                values.append(1)
                indices.append(dt)
            elif growths[-1] == 1:
                growths.append(-1)
                values.append(1)
                indices.append(dt)
            else:
                pass
        elif indicator_value < lower_bound:
            if len(values) == 0:
                values.append(0)
                growths.append(1)
                indices.append(dt)
            elif values[-1] == 1:
                values.append(-1)
                growths.append(1)
                indices.append(dt)
            else:
                pass
        pre_indicator_value = indicator_value
    values = pd.Series(values, index=indices)
    growths = pd.Series(growths, index=indices)
    trade_signals =  pd.DataFrame({'trade_boom': values, 'trade_recession': growths})
    #print()
    return trade_signals, 1

# def strategy_u_bound_l_bound와 거의 똑같음, input부분만 조금 다름)
def strategy_oecd_cli_1(recent_cli_series, upper_bound=100, lower_bound=100):     #, eco_indicator_ticker, upper_bound, lower_bound):
    trade_signals = pd.DataFrame()
    positions = pd.DataFrame()
    num_of_trade = 0
    values = []
    growths = []
    indices = []
    #for i, (dt, eco_indicator) in enumerate(df.iterrows()):
    for release_date, indicator_value in recent_cli_series.iteritems():
        #print()
        #indicator_value = eco_indicator[eco_indicator_ticker, 'px_last']
        if indicator_value > upper_bound:
            if len(growths) == 0:
                growths.append(0)
                values.append(1)
                indices.append(release_date)
            elif growths[-1] == 1:
                growths.append(-1)
                values.append(1)
                indices.append(release_date)
            else:
                pass
        elif indicator_value < lower_bound:
            if len(values) == 0:
                values.append(0)
                growths.append(1)
                indices.append(release_date)
            elif values[-1] == 1:
                values.append(-1)
                growths.append(1)
                indices.append(release_date)
            else:
                pass
        pre_indicator_value = indicator_value
    values = pd.Series(values, index=indices)
    growths = pd.Series(growths, index=indices)
    trade_signals = pd.DataFrame({'trade_boom': values, 'trade_recession': growths})
    #print()
    return trade_signals


def strategy_2(df, eco_indicator_ticker):
    trades = pd.DataFrame()
    positions = pd.DataFrame()
    df_pct_change = df.diff()[1:]
    num_of_negative = 0
    num_of_positive = 0
    new_position = np.array([0, 0])
    for dt, inddicator_pct_change in df_pct_change.iterrows():
        pct_change = inddicator_pct_change[eco_indicator_ticker, 'px_last']
        if pct_change > 0:
            if num_of_positive == 0:
                num_of_negative = 0
            num_of_positive += 1
        elif pct_change < 0:
            if num_of_negative == 0:
                num_of_positive = 0
            num_of_negative += 1
        else:
            num_of_positive = 0
            num_of_negative = 0

        if num_of_positive >= 3:
            if num_of_positive == 3 and positions['trade_boom'].values[-1] == 0:
                # initial trade
                if trades.index.size == 0:
                    trades = pd.concat([trades,
                                        pd.DataFrame({'trade_boom': 1, 'trade_recession':0},
                                                        index=[dt])],
                                        axis=0)
                    new_position = positions.values[-1] + [1, 0]
                else:
                    trades = pd.concat([trades,
                                        pd.DataFrame({'trade_boom': 1, 'trade_recession':-1},
                                                        index=[dt])],
                                        axis=0)
                    new_position = positions.values[-1] + [1, -1]
        elif num_of_negative >= 3:
            if num_of_negative == 3 and positions['trade_recession'].values[-1] == 0:
                # initial trade
                if trades.index.size == 0:
                    trades = pd.concat([trades,
                                        pd.DataFrame({'trade_boom': 0, 'trade_recession':1},
                                                        index=[dt])],
                                        axis=0)
                    new_position = positions.values[-1] + [0, 1]
                else:
                    trades = pd.concat([trades,
                                        pd.DataFrame({'trade_boom': -1, 'trade_recession':1},
                                                        index=[dt])],
                                        axis=0)
                    new_position = positions.values[-1] + [-1, 1]
        #else:
        #    new_position = np.array([0, 0])
        positions = pd.concat([positions,
                                pd.DataFrame({'trade_boom': new_position[0],
                                              'trade_recession': new_position[1]},
                                            index=[dt])],
                                axis=0)
        #print(dt, pct_change, num_of_positive, num_of_negative, positions.values[-1])
    return trades, positions


def strategy_oecd_cli_2(released_3_clis):     #, eco_indicator_ticker, upper_bound, lower_bound):
    trades = pd.DataFrame(columns = ['trade_boom', 'trade_recession'])
    now_position = None
    for release_date, recent_3_clis in released_3_clis.iterrows():
        if recent_3_clis.is_monotonic_increasing and now_position != "boom":
            # initial trade
            #if trades.index.size == 0:
            if now_position is None:
                trades = pd.concat([trades,
                                    pd.DataFrame({'trade_boom': 1, 'trade_recession': 0},
                                                 index=[release_date])],
                                   axis=0)
            else:
                trades = pd.concat([trades,
                                    pd.DataFrame({'trade_boom': 1, 'trade_recession': -1},
                                                 index=[release_date])],
                                   axis=0)
            now_position = 'boom'
        elif recent_3_clis.is_monotonic_decreasing and now_position != 'recession':
            #if trades.index.size == 0:
            if now_position is None:
                trades = pd.concat([trades,
                                    pd.DataFrame({'trade_boom': 0, 'trade_recession': 1},
                                                 index=[release_date])],
                                   axis=0)
            else:
                trades = pd.concat([trades,
                                    pd.DataFrame({'trade_boom': -1, 'trade_recession': 1},
                                                 index=[release_date])],
                                   axis=0)
            now_position = 'recession'
    return trades