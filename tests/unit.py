import pytest
import pandas as pd
from pandas.core import series
from finta import TA


ohlc = pd.read_csv('./data/bittrex:btc-usdt.csv', index_col='date',
                   parse_dates=True)


def test_sma():
    '''test TA.ma'''

    ma = TA.SMA(ohlc, 14)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6922.3392206307135


def test_smm():
    '''test TA.SMM'''

    ma = TA.SMM(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6490.0


def test_ema():
    '''test TA.EMA'''

    ma = TA.EMA(ohlc, 50)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 7606.8439195057081


def test_dema():
    '''test TA.DEMA'''

    ma = TA.DEMA(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6323.4192399358026


def test_tema():
    '''test TA.TEMA'''

    ma = TA.TEMA(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6307.4815184378531


def test_trima():
    '''test TA.TRIMA'''

    ma = TA.TRIMA(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 7464.8530730425946


def test_trix():
    '''test TA.TRIX'''

    ma = TA.TRIX(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == -142.64042774606133


def test_vama():
    '''test TA.VAMA'''

    ma = TA.VAMA(ohlc, 20)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6991.5779125774961


def test_er():
    '''test TA.ER'''

    er = TA.ER(ohlc)

    assert isinstance(er, series.Series)
    assert -100 < er.values[-1] < 100


def test_kama():
    '''test TA.KAMA'''

    ma = TA.KAMA(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6742.1178650230386


def test_zlema():
    '''test TA.ZLEMA'''

    ma = TA.ZLEMA(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 5193.0313725800006


def test_wma():
    '''test TA.WMA'''

    ma = TA.WMA(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6474.470030782667


def test_hma():
    '''test TA.WMA'''

    ma = TA.HMA(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6186.937271459321


def test_vwap():
    '''test TA.VWAP'''

    ma = TA.VWAP(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 7976.5174347663306


def test_smma():
    '''test TA.SMMA'''

    ma = TA.SMMA(ohlc)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 8020.2742957005539


def test_macd():
    '''test TA.MACD'''

    macd = TA.MACD(ohlc)

    assert isinstance(macd['MACD'], series.Series)
    assert isinstance(macd['SIGNAL'], series.Series)

    assert macd['MACD'].values[-1] == -419.21923359275115
    assert macd['SIGNAL'].values[-1] == -372.39851312056192


def test_vw_macd():
    '''test TA.VW_MACD'''

    macd = TA.VW_MACD(ohlc)

    assert isinstance(macd['MACD'], series.Series)
    assert isinstance(macd['SIGNAL'], series.Series)

    assert macd['MACD'].values[-1] == -535.21281201397142
    assert macd['SIGNAL'].values[-1] == -511.64584818187575


def test_mom():
    '''test TA.MOM'''

    mom = TA.MOM(ohlc)

    assert isinstance(mom, series.Series)
    assert mom.values[-1] == -1215.5468137099997


def test_roc():
    '''test TA.ROC'''

    roc = TA.ROC(ohlc)

    assert isinstance(roc, series.Series)
    assert roc.values[-1] == -15.98340762838


def test_rsi():
    '''test TA.RSI'''

    rsi = TA.RSI(ohlc)

    assert isinstance(rsi, series.Series)
    assert -100 < rsi.values[-1] < 100


def test_ift_rsi():
    '''test TA.IFT_RSI'''

    rsi = TA.IFT_RSI(ohlc)

    assert isinstance(rsi, series.Series)
    assert rsi.values[-1] == 2.6918116852046792


def test_tr():
    '''test TA.TR'''

    tr = TA.TR(ohlc)

    assert isinstance(tr, series.Series)
    assert tr.values[-1] == 113.39999999999964


def test_atr():
    '''test TA.ATR'''

    tr = TA.ATR(ohlc)

    assert isinstance(tr, series.Series)
    assert tr.values[-1] == 328.56890383071419


def test_sar():
    '''test TA.SAR'''

    sar = TA.SAR(ohlc)

    assert isinstance(sar, series.Series)
    assert sar.values[-1] == 7127.1508782052497


def test_bbands():
    '''test TA.BBANDS'''

    bb = TA.BBANDS(ohlc)

    assert isinstance(bb['UPPER'], series.Series)
    assert isinstance(bb['MIDDLE'], series.Series)
    assert isinstance(bb['LOWER'], series.Series)

    assert bb['UPPER'].values[-1] == 8212.7979228041968
    assert bb['MIDDLE'].values[-1] == 7110.5508235434954
    assert bb['LOWER'].values[-1] == 6008.303724282795


def test_bbwidth():
    '''test TA.BBWIDTH'''

    bb = TA.BBWIDTH(ohlc)

    assert isinstance(bb, series.Series)
    assert 0 < bb.values[-1] < 1


def test_percentb():
    '''test TA.PERCENT_B'''

    bb = TA.PERCENT_B(ohlc)

    assert isinstance(bb, series.Series)
    assert bb.values[-1] == 0.18695874195706308


def test_kc():
    '''test TA.KC'''

    kc = TA.KC(ohlc)

    assert isinstance(kc['UPPER'], series.Series)
    assert isinstance(kc['MIDDLE'], series.Series)
    assert isinstance(kc['LOWER'], series.Series)

    assert kc['UPPER'].values[-1] == 7844.5697540734927
    assert kc['MIDDLE'].values[-1] == 7110.5508235434954
    assert kc['LOWER'].values[-1] == 6376.5318930134981


def test_do():
    '''test TA.DO'''

    do = TA.DO(ohlc)

    assert isinstance(do['UPPER'], series.Series)
    assert isinstance(do['MIDDLE'], series.Series)
    assert isinstance(do['LOWER'], series.Series)

    assert do['UPPER'].values[-1] == 7770.0
    assert do['MIDDLE'].values[-1] == 7010.0005000000001
    assert do['LOWER'].values[-1] == 6250.0010000000002


def test_dmi():
    '''test TA.DMI'''

    dmi = TA.DMI(ohlc)

    assert isinstance(dmi['DI+'], series.Series)
    assert isinstance(dmi['DI-'], series.Series)

    assert dmi['DI+'].values[-1] == 0.32826999511691435
    assert dmi['DI-'].values[-1] == 10.09866984475557


def test_adx():
    '''test TA.ADX'''

    adx = TA.ADX(ohlc)

    assert isinstance(adx, series.Series)
    assert adx.values[-1] == 66.589993072391422


def test_stoch():
    '''test TA.STOCH'''

    st = TA.STOCH(ohlc)

    assert isinstance(st, series.Series)
    assert 0 < st.values[-1] < 100


def test_stochd():
    '''test TA.STOCHD'''

    st = TA.STOCHD(ohlc)

    assert isinstance(st, series.Series)
    assert 0 < st.values[-1] < 100


def test_stochrsi():
    '''test TA.STOCRSI'''

    st = TA.STOCHRSI(ohlc)

    assert isinstance(st, series.Series)
    assert 0 < st.values[-1] < 100


def test_williams():
    '''test TA.WILLIAMS'''

    w = TA.WILLIAMS(ohlc)

    assert isinstance(w, series.Series)
    assert -100 < w.values[-1] < 0


def test_uo():
    '''test TA.UO'''

    uo = TA.UO(ohlc)

    assert isinstance(uo, series.Series)
    assert 0 < uo.values[-1] < 100


def test_ao():
    '''test TA.AO'''

    ao = TA.AO(ohlc)

    assert isinstance(ao, series.Series)
    assert ao.values[-1] == -957.63459032713035


def test_mi():
    '''test TA.MI'''

    mi = TA.MI(ohlc)

    assert isinstance(mi, series.Series)
    assert mi.values[-1] == 23.928086961089647


def test_vortex():
    '''test TA.VORTEX'''

    v = TA.VORTEX(ohlc)

    assert isinstance(v['VIp'], series.Series)
    assert isinstance(v['VIm'], series.Series)

    assert v['VIp'].values[-1] == 37.443158543691659
    assert v['VIm'].values[-1] == -22.605012093615489


def test_kst():
    '''test TA.KST'''

    kst = TA.KST(ohlc)

    assert isinstance(kst['KST'], series.Series)
    assert isinstance(kst['signal'], series.Series)

    assert kst['KST'].values[-1] == -161.7861811191122
    assert kst['signal'].values[-1] == -141.29962282675882
