from datetime import datetime

import numpy as np
import pandas as pd
from pytz import timezone, utc
#import torch
#from torch import nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def predict_premordial(underlying_prices, etf_start_dt, fitting_result, initial_ratio):
                       #initial_ratio, korean_closing_time):
    _, _, model = fitting_result
    premordial_undl_prices = underlying_prices[underlying_prices.index < etf_start_dt]
    if premordial_undl_prices.size == 0:
        print(underlying_prices)
        print(etf_start_dt, underlying_prices.index.min())
        raise ValueError
    prediction = model.predict(np.arange(-len(premordial_undl_prices.index), 0).reshape(-1, 1))
    premordial_adjusted_ratio = pd.DataFrame({'adjusted_ratio': prediction.flatten()},
                                             index=premordial_undl_prices.index)
    premordials = pd.concat([premordial_undl_prices, premordial_adjusted_ratio], axis=1)
    #premordial_navs = premordials.apply(lambda x: x['point_in_KRW'] * x['adjusted_ratio'] * initial_ratio / 100, axis=1)
    #premordial_navs.index += korean_closing_time
    return premordials.apply(lambda x: x['point_in_krw'] * x['adjusted_ratio'] * initial_ratio / 100, axis=1)

def get_fit_info(fitting_result, navs_underlying_prices, etf_isin):
    x, y, regr = fitting_result
    prediction = pd.DataFrame({'fitting_result': regr.predict(x).flatten()})
    adjusted_r = navs_underlying_prices['adjusted_ratio'].reset_index()
    fit_etf_undl = pd.concat([adjusted_r, prediction], axis=1).set_index('base_dt')
    fit_etf_undl.index = pd.MultiIndex.from_tuples([(etf_isin, idx) for idx in fit_etf_undl.index],
                                                    names=['etf_isin', 'base_dt'])
    return fit_etf_undl

def update_etf_undl_eq(fitting_result, etf_isin, undl_ticker, multiplier, etf_undl_eq):
    Xs, ys_true, model = fitting_result
    ys_pred = model.predict(Xs)
    mse = mean_squared_error(ys_true, ys_pred)
    rmse = mean_squared_error(ys_true, ys_pred, squared=False)
    R_square = model.score(Xs, ys_true)
    a_tangent = model.coef_[0]
    b_bias = model.intercept_[0]
    if etf_isin in etf_undl_eq.index:
        etf_undl_eq.loc[etf_isin, 'base_dt'] = datetime.now(timezone('UTC'))
        etf_undl_eq.loc[etf_isin, 'undl_ticker'] = undl_ticker
        etf_undl_eq.loc[etf_isin, 'a_tangent'] = a_tangent
        etf_undl_eq.loc[etf_isin, 'b_bias'] = b_bias
        etf_undl_eq.loc[etf_isin, 'r_square'] = R_square
        etf_undl_eq.loc[etf_isin, 'mse'] = mse
        etf_undl_eq.loc[etf_isin, 'rmse'] = rmse
        etf_undl_eq.loc[etf_isin, 'undl_multiplier'] = multiplier
        etf_undl_eq.loc[etf_isin, 'num_of_data'] = int(Xs.size)
        return etf_undl_eq
    else:
        new_df = pd.DataFrame({'base_dt': datetime.now(timezone('UTC')),
                                'undl_ticker': undl_ticker,
                                'fit_method': 'LinearRegressoin',
                                'a_tangent': a_tangent,
                                'b_bias': b_bias,
                                'r_square': R_square,
                                'mse': mse,
                                'rmse': rmse,
                                'undl_multiplier': multiplier,
                                'num_of_data': int(Xs.size)}, index=[etf_isin])
        new_df.index.name = 'etf_isin'
        return etf_undl_eq.append(new_df)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def linear_fit_by_nn(x, y):
    model = nn.Linear(in_features=1, out_features=1, bias=True)
    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1.0)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
    #                                        lr_lambda=lambda epoch: 0.99 ** epoch)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    prediction = model(x)
    loss = criterion(input=prediction, target=y)
    i = 0
    while loss.data.item() > 3:
        i += 1
        prediction = model(x)
        loss = criterion(input=prediction, target=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        #if i % 20 == 0:
        #    print('step {}, loss={:.4}, lr={}'.format(i, loss.data.item(), get_lr(optimizer)))
        if i == 10000:
            break
    j = 0
    for j in range(i, 2500):
        prediction = model(x)
        loss = criterion(input=prediction, target=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    weight = model.weight.data.item(),
    bias = model.bias.data.item(),
    loss_score = loss.data.item()
    return weight, bias, loss_score, max(i, j)

def linear_regression(x, y):
    reg = LinearRegression().fit(x, y)
    return reg.coef_, reg.intercept_, reg.score(x, y), None

def get_torch_x_y(df):
    x = torch.from_numpy(df.index.values).unsqueeze(dim=1).float()
    y = torch.from_numpy(df['adjusted_ratio'].values).unsqueeze(dim=1).float()
    return x, y

def fit_linear(adjusted_ratio_pd):
    df = adjusted_ratio_pd.reset_index()
    x = df.index.values.astype(np.float).reshape(-1, 1)
    y = df['adjusted_ratio'].values.astype(np.float).reshape(-1, 1)
    regr = LinearRegression().fit(x, y)
    #return x, y, model, loss, max(i, j)
    #return x, y, weight, bias, loss_score, max(i, j)
    return x, y, regr

#def append_ax_b(etf_nav_bm_linear_eqs, fitting_result, info):
#    _, _, regr = fitting_result
#    new_row = pd.DataFrame([[regr.coef_[0,0], regr.intercept_[0]]], columns=['A', 'b'], index=[info['ETF_ticker']])
#    return etf_nav_bm_linear_eqs.append(new_row)