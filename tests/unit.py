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

    assert isinstance(macd['macd'], series.Series)
    assert isinstance(macd['macd_signal'], series.Series)

    assert macd['MACD'].values[-1] == -419.21923359275115
    assert macd['SIGNAL'].values[-1] == -372.39851312056192
