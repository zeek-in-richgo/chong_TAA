import os
from os.path import basename, dirname

import matplotlib.pyplot as plt

from utils.data_manipulation import fill_index

def draw_simple(fig, kospi_chong_1, recession_periods_1,
         boom_name, recession_name,
         eco_indicator_name, upper_bound,lower_bound, asset_key, strategy_id='1'):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    fontsize = 25
    fontdict = {'fontsize': fontsize}

    # plt.title('한국 OECD 경기순환지수(Composite Leading Indicator) 가치주 성장주 스위칭 전략')
    #ax1 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(1, 1, 1)
    # ax1.plot('스위칭전략', 'KOSPI200', kospi_chong_1[['my_portfolio', 'KOSPI2 Index']])
    # ax1.plot('my_portfolio', 'KOSPI2 Index', kospi_chong_1[['my_portfolio', 'KOSPI2 Index']])
    #ax1.plot(kospi_chong_1['my_portfolio'], color='purple')
    #ax1.plot(kospi_chong_1['asset_in_boom'], color='blue')
    #ax1.plot(kospi_chong_1['asset_in_recession'], color='red')
    #ax1.plot(kospi_chong_1[['my_portfolio', 'asset_in_boom', 'asset_in_recession']])
    #ax1.plot(kospi_chong_1[['my_portfolio', 'asset_in_boom']])
    ax1.plot(kospi_chong_1[['my_portfolio', 'KOSPI2 Index']])
    print()

    # ax1.axvspan('1998-06-30', '1999-01-29', facecolor='gray', alpha=0.5)
    # span = ('1998-06-30', '1999-01-29')
    for span in recession_periods_1:
        ax1.axvspan(*span, facecolor='gray', alpha=0.5)
    #ax1.legend(['스위칭전략', boom_name, recession_name], fontsize=fontsize, loc='upper left', framealpha=0.4)
    #ax1.legend(['스위칭전략', boom_name], fontsize=fontsize, loc='upper left', framealpha=0.4)
    ax1.legend(['스위칭전략', 'KOSPI 200 지수'], fontsize=fontsize, loc='upper left', framealpha=0.4)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    #ax2 = fig.add_subplot(2, 1, 2)
    ##ax2.plot(kospi_chong_2[['my_portfolio', 'KOSPI2 Index']])
    #ax2.plot(kospi_chong_2[['my_portfolio', 'asset_in_boom', 'asset_in_recession']])
    #for span in recession_periods_2:
    #    ax2.axvspan(*span, facecolor='gray', alpha=0.5)
    #ax2.legend(['스위칭전략', boom_name, recession_name], fontsize=fontsize, loc='upper left', framealpha=0.4)
    #plt.xticks(fontsize=fontsize)
    #plt.yticks(fontsize=fontsize)

    ax1.set_ylabel('1997 = 100', fontdict=fontdict)
    if strategy_id == '1':
        #ax1.set_title(eco_indicator_name + '> {}이면 {}, < {}이면 {}'.format(upper_bound, boom_name,
        #                                                                    lower_bound, recession_name),
        ax1.set_title(eco_indicator_name + '> {}이면 {}, < {}이면 {}'.format(upper_bound, boom_name,
                                                                            lower_bound, recession_name),
                      fontdict=fontdict)
    elif strategy_id == '2':
        ax1.set_title(eco_indicator_name + \
                      '가 3달 연속 증가하면 {}, 3달 연속 감소하면 {}, 그외 포지션유지'.format(boom_name, recession_name),
                      fontdict=fontdict)
    ## ax1.set_xticklabels(fontdict=fontdict)
    ## ax2.set_xticklabels(fontdict=fontdict)

    png_path = './pngs_test/{}/taa_1_{}_strategy{}.png'.format(eco_indicator_name, asset_key, strategy_id)
    os.makedirs(dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    #print('./taa_1_{}_{}.png'.format(asset_key, eco_indicator_name))

def draw_simple_general(fig, kospi_chong_1, recession_periods_1,
         boom_name, recession_name,
         eco_indicator_name, upper_bound,lower_bound, asset_key,
        bm_name='KOSPI200지수', bm_ticker='KOSPI2 Index', strategy_id='1'):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    fontsize = 25
    fontdict = {'fontsize': fontsize}

    # plt.title('한국 OECD 경기순환지수(Composite Leading Indicator) 가치주 성장주 스위칭 전략')
    #ax1 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(1, 1, 1)
    # ax1.plot('스위칭전략', 'KOSPI200', kospi_chong_1[['my_portfolio', 'KOSPI2 Index']])
    # ax1.plot('my_portfolio', 'KOSPI2 Index', kospi_chong_1[['my_portfolio', 'KOSPI2 Index']])
    #ax1.plot(kospi_chong_1['my_portfolio'], color='purple')
    #ax1.plot(kospi_chong_1['asset_in_boom'], color='blue')
    #ax1.plot(kospi_chong_1['asset_in_recession'], color='red')
    #ax1.plot(kospi_chong_1[['my_portfolio', 'asset_in_boom', 'asset_in_recession']])
    #ax1.plot(kospi_chong_1[['my_portfolio', 'asset_in_boom']])
    ax1.plot(kospi_chong_1[['my_portfolio', bm_ticker]])
    print()

    # ax1.axvspan('1998-06-30', '1999-01-29', facecolor='gray', alpha=0.5)
    # span = ('1998-06-30', '1999-01-29')
    for span in recession_periods_1:
        ax1.axvspan(*span, facecolor='gray', alpha=0.5)
    #ax1.legend(['스위칭전략', boom_name, recession_name], fontsize=fontsize, loc='upper left', framealpha=0.4)
    #ax1.legend(['스위칭전략', boom_name], fontsize=fontsize, loc='upper left', framealpha=0.4)
    ax1.legend(['스위칭전략', bm_name], fontsize=fontsize, loc='upper left', framealpha=0.4)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    #ax2 = fig.add_subplot(2, 1, 2)
    ##ax2.plot(kospi_chong_2[['my_portfolio', 'KOSPI2 Index']])
    #ax2.plot(kospi_chong_2[['my_portfolio', 'asset_in_boom', 'asset_in_recession']])
    #for span in recession_periods_2:
    #    ax2.axvspan(*span, facecolor='gray', alpha=0.5)
    #ax2.legend(['스위칭전략', boom_name, recession_name], fontsize=fontsize, loc='upper left', framealpha=0.4)
    #plt.xticks(fontsize=fontsize)
    #plt.yticks(fontsize=fontsize)

    ax1.set_ylabel('1997 = 100', fontdict=fontdict)
    if strategy_id == '1':
        #ax1.set_title(eco_indicator_name + '> {}이면 {}, < {}이면 {}'.format(upper_bound, boom_name,
        #                                                                    lower_bound, recession_name),
        ax1.set_title(eco_indicator_name + '> {}이면 {}, < {}이면 {}'.format(upper_bound, boom_name,
                                                                            lower_bound, recession_name),
                      fontdict=fontdict)
    elif strategy_id == '2':
        ax1.set_title(eco_indicator_name + \
                      '가 3달 연속 증가하면 {}, 3달 연속 감소하면 {}, 그외 포지션유지'.format(boom_name, recession_name),
                      fontdict=fontdict)
    ## ax1.set_xticklabels(fontdict=fontdict)
    ## ax2.set_xticklabels(fontdict=fontdict)

    png_path = './pngs_test/{}/taa_1_{}_strategy{}.png'.format(eco_indicator_name, asset_key, strategy_id)
    os.makedirs(dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    #print('./taa_1_{}_{}.png'.format(asset_key, eco_indicator_name))


def draw_full(fig, kospi_chong_1, kospi_chong_2,
         recession_periods_1, recession_periods_2,
         boom_name, recession_name,
         df, eco_indicator_name, upper_bound, lower_bound, asset_key):
#def draw_full(fig, kospi_chong_1, kospi_chong_2,
#              recession_periods_1, recession_periods_2,
#              boom_name, recession_name,
#              df, eco_indicator_name, threshold, asset_key):

    #fontsize = 25
    #fontdict = {'fontsize': fontsize}

    plt.rcParams['font.family'] = 'Malgun Gothic'
    fontsize = 25
    fontdict = {'fontsize': fontsize}
    fig = plt.figure(figsize=(20, 18))
    # plt.title('한국 OECD 경기순환지수(Composite Leading Indicator) 가치주 성장주 스위칭 전략')
    ax1 = fig.add_subplot(2, 1, 1)
    # ax1.plot('스위칭전략', 'KOSPI200', kospi_chong_1[['my_portfolio', 'KOSPI2 Index']])
    # ax1.plot('my_portfolio', 'KOSPI2 Index', kospi_chong_1[['my_portfolio', 'KOSPI2 Index']])
    #ax1.plot(kospi_chong_1['my_portfolio'], color='purple')
    #ax1.plot(kospi_chong_1['asset_in_boom'], color='blue')
    #ax1.plot(kospi_chong_1['asset_in_recession'], color='red')
    ax1.plot(kospi_chong_1[['my_portfolio', 'asset_in_boom', 'asset_in_recession']])

    # ax1.axvspan('1998-06-30', '1999-01-29', facecolor='gray', alpha=0.5)
    # span = ('1998-06-30', '1999-01-29')
    for span in recession_periods_1:
        ax1.axvspan(*span, facecolor='gray', alpha=0.5)
    ax1.legend(['스위칭전략', boom_name, recession_name], fontsize=fontsize, loc='upper left', framealpha=0.4)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax1_1 = ax1.twinx()
    ax1_1.plot(fill_index(df), color='red')
    ax1_1.plot(kospi_chong_1.index, [upper_bound] * kospi_chong_1.index.size, color='black')
    ax1_1.plot(kospi_chong_1.index, [lower_bound] * kospi_chong_1.index.size, color='black')
    ax1_1.legend([eco_indicator_name, 'upper_bound', 'lower_bound'], fontsize=fontsize, loc='upper right', framealpha=0.4)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax2 = fig.add_subplot(2, 1, 2)
    #ax2.plot(kospi_chong_2[['my_portfolio', 'KOSPI2 Index']])
    ax2.plot(kospi_chong_2[['my_portfolio', 'asset_in_boom', 'asset_in_recession']])
    for span in recession_periods_2:
        ax2.axvspan(*span, facecolor='gray', alpha=0.5)
    ax2.legend(['스위칭전략', boom_name, recession_name], fontsize=fontsize, loc='upper left', framealpha=0.4)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax2_1 = ax2.twinx()
    ax2_1.plot(fill_index(df), color='red')
    ax2_1.legend(eco_indicator_name, fontsize=fontsize, loc='upper right', framealpha=0.4)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax1.set_ylabel('1997 = 100', fontdict=fontdict)
    ax1.set_title(eco_indicator_name + '> {}이면 {}, \n < {}이면 {}'.format(upper_bound, boom_name,
                                                                           lower_bound, recession_name),
                  fontdict=fontdict)
    ax2.set_title(eco_indicator_name + \
                  '가 3달 연속 증가하면 {}, 3달 연속 감소하면 {}, 그외 포지션유지'.format(boom_name, recession_name),
                  fontdict=fontdict)
    # ax1.set_xticklabels(fontdict=fontdict)
    # ax2.set_xticklabels(fontdict=fontdict)

    png_path = './pngs_test_drawed_fully/{}/taa_{}.png'.format(eco_indicator_name, asset_key)
    os.makedirs(dirname(png_path), exist_ok=True)
    plt.savefig(png_path)

    print('./taa_1_{}_{}.png'.format(asset_key, eco_indicator_name))



def draw_correlation_heatmap(correl_mtx):
    # 그림 사이즈 지정
    fig, ax = plt.subplots( figsize=(30,30) )

    # 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
    mask = np.zeros_like(correl_mtx, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # 히트맵을 그린다
    sns.heatmap(correl_mtx,
                cmap = 'RdYlBu_r',
                annot = True,   # 실제 값을 표시한다
                mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
                linewidths=.5,  # 경계면 실선으로 구분하기
                cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
                vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
               )
    #plt.show()
    plt.savefig('./correl_underlying.png', dpi=400, bbox_inches='tight')

def draw_and_save_img(fig, png_path, undl2etf_eq, fitting_result):
    #x, y, model, loss, i = fitting_result
    x, y, regr = fitting_result
    #prediction = model(x)
    prediction_y = regr.predict(x)
    #ax1 = fig.add_subplot(1, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    #ax3 = fig.add_subplot(3, 1, 3)
    #ax1.set_ylabel('NAV & underlying')
    ax2.set_ylabel('NAV / underlying ratio fitting')
    #ax3.set_ylabel('underling_prices_whole_period')
    #ax2.set_xlabel('loss={:.4}, w={:.4}, b={:.4} after {} '.format(loss.data.item(),
    #                                                                model.weight.data.item() ,
    #                                                                model.bias.data.item() ,
    #                                                               i))
    ax2.set_xlabel('RMSE={:.02f}, R-squre={:.02f}, A={:.05f}, b={:.02f}'.format(undl2etf_eq['rmse'],
                                                                                regr.score(x, y),
                                                                                regr.coef_[0,0],
                                                                                regr.intercept_[0]))
    #ax1.plot(navs_underlying_prices.loc[:, ['etf_NAV', 'adjusted_underlying']])
    #ax2.plot(navs_underlying_prices['adjusted_ratio'])
    ax2.scatter(x, y)
    #ax2.plot(x, prediction.data.numpy(), 'b--')
    ax2.plot(x, prediction_y, 'b--')
    #ax3.plot(underlying_prices, label='test')
    #os.makedirs('./imgs.' + suffix, exist_ok=True)
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    #plt.savefig(get_deep_target_img_path(info), dpi=400, bbox_inches='tight')
    plt.clf()

def draw_tracking_error(fig, info, tr, nav_tr, suffix):
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 2, 1)
    ax1.set_ylabel('TRACKING ERR')
    ax2.set_ylabel('NAV TRACKING ERR')

    ax1.plot(tr)
    ax2.plot(nav_tr)
    os.makedirs('./imgs_tr.' + suffix, exist_ok=True)
    plt.savefig(target_img_path(info, './imgs_tr.' + suffix), dpi=400, bbox_inches='tight')
    plt.savefig(get_deep_target_img_path(info), dpi=400, bbox_inches='tight')
    plt.clf()