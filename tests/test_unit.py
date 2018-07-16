import pytest
import os
import pandas as pd
from pandas.core import series
from finta import TA


@pytest.fixture
def rootdir():

    return os.path.dirname(os.path.abspath(__file__))


data_file = os.path.join(rootdir(), 'data/bittrex:btc-usdt.csv')

ohlc = pd.read_csv(data_file, index_col='date',
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


def test_ppo():
    '''test TA.PPO'''

    ppo = TA.PPO(ohlc)

    assert isinstance(ppo['PPO'], series.Series)
    assert isinstance(ppo['SIGNAL'], series.Series)
    assert isinstance(ppo['HISTO'], series.Series)

    assert  macd['PPO'].values[-1] == -5.85551658018139331574047901085578
    assert  macd['SIGNAL'].values[-1] == -5.05947256175217674467603501398116
    assert  macd['HISTO'].values[-1] == -0.79604401842921657106444399687462


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

    assert isinstance(bb['BB_UPPER'], series.Series)
    assert isinstance(bb['BB_MIDDLE'], series.Series)
    assert isinstance(bb['BB_LOWER'], series.Series)

    assert bb['BB_UPPER'].values[-1] == 8212.7979228041968
    assert bb['BB_MIDDLE'].values[-1] == 7110.5508235434954
    assert bb['BB_LOWER'].values[-1] == 6008.303724282795


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

    ma = TA.ZLEMA(ohlc, 20)
    kc = TA.KC(ohlc, MA=ma)

    assert isinstance(kc['KC_UPPER'], series.Series)
    assert isinstance(kc['KC_LOWER'], series.Series)

    assert kc['KC_UPPER'].values[-1] == 6059.9253031099979
    assert kc['KC_LOWER'].values[-1] == 4591.8874420500033


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


def test_tsi():
    '''test TA.TSI'''

    tsi = TA.TSI(ohlc)

    assert isinstance(tsi['TSI'], series.Series)
    assert isinstance(tsi['signal'], series.Series)

    assert tsi['TSI'].values[-1] == -32.128372005959058
    assert tsi['signal'].values[-1] == -26.94173826727873


def test_tp():
    '''test TA.TP'''

    tp = TA.TP(ohlc)

    assert isinstance(tp, series.Series)
    assert tp.values[-1] == 6429.0177287633342


def test_adl():

    adl = TA.ADL(ohlc)

    assert isinstance(adl, series.Series)
    assert adl.values[-1] == 1221072523.7384958


def test_chaikin():
    '''test TA.CHAIKIN'''

    c = TA.CHAIKIN(ohlc)

    assert isinstance(c, series.Series)
    assert c.values[-1] == 650594.74888467789


def test_mfi():
    '''test TA.MFI'''

    mfi = TA.MFI(ohlc)

    assert isinstance(mfi, series.Series)
    assert 0 < mfi.values[-1] < 100


def test_obv():
    '''test TA.OBV'''

    o = TA.OBV(ohlc)

    assert isinstance(o, series.Series)
    assert o.values[-1] == -277433.76499578007


def test_wobv():
    '''test TA.OBV'''

    o = TA.WOBV(ohlc)

    assert isinstance(o, series.Series)
    assert o.values[-1] == -85332065.01331231


def test_vzo():
    '''test TA.VZO'''

    vzo = TA.MFI(ohlc)

    assert isinstance(vzo, series.Series)
    assert -100 < vzo.values[-1] < 100


def test_efi():
    '''test TA.EFI'''

    efi = TA.EFI(ohlc)

    assert isinstance(efi, series.Series)
    assert efi.values[-1] == 6918216.7131493781


def test_cfi():
    '''test TA.CFI'''

    cfi = TA.CFI(ohlc)

    assert isinstance(cfi, series.Series)
    assert cfi.values[-1] == -84856289.556287795


def test_ebbp():
    '''test TA.EBBP'''

    eb = TA.EBBP(ohlc)

    assert isinstance(eb['Bull.'], series.Series)
    assert isinstance(eb['Bear.'], series.Series)
    assert eb['Bull.'].values[-1] == -285.40231904032862


def test_emv():
    '''test TA.EMV'''

    emv = TA.EMV(ohlc)

    assert isinstance(emv, series.Series)
    assert emv.values[-1] == 2407071622.223393


def test_cci():
    '''test TA.CCI'''

    cci = TA.CCI(ohlc)

    assert isinstance(cci, series.Series)
    assert cci.values[-1] == -13.422528354279628


def test_basp():
    '''test TA.BASP'''

    basp = TA.BASP(ohlc)

    assert isinstance(basp['Buy.'], series.Series)
    assert isinstance(basp['Sell.'], series.Series)

    assert basp['Buy.'].values[-1] == 0.066916805574281202
    assert basp['Sell.'].values[-1] == 0.091486900946605054


def test_baspn():
    '''test TA.BASPN'''

    basp = TA.BASPN(ohlc)

    assert isinstance(basp['Buy.'], series.Series)
    assert isinstance(basp['Sell.'], series.Series)

    assert basp['Buy.'].values[-1] == 0.56374213100174275
    assert basp['Sell.'].values[-1] == 0.74103021131003344


def test_cmo():
    '''test TA.CMO'''

    cmo = TA.CMO(ohlc)

    assert isinstance(cmo, series.Series)
    assert -100 < cmo.values[-1] < 100


def test_chandelier():
    '''test TA.CHANDELIER'''

    chan = TA.CHANDELIER(ohlc)

    assert isinstance(chan['Long.'], series.Series)
    assert isinstance(chan['Short.'], series.Series)

    assert chan['Long.'].values[-1] == 6723.8927646477259
    assert chan['Short.'].values[-1] == 5326.4927656377258


def test_qstick():
    '''test TA.QSTICK'''

    q = TA.QSTICK(ohlc)

    assert isinstance(q, series.Series)
    assert q.values[-1] == 0.2466561628571721


def test_tmf():

    with pytest.raises(NotImplementedError):
        tmf = TA.TMF(ohlc)


def test_wto():
    '''test TA.WTO'''

    wto = TA.WTO(ohlc)

    assert isinstance(wto['WT1.'], series.Series)
    assert isinstance(wto['WT2.'], series.Series)

    assert wto['WT1.'].values[-1] == -60.290069910634649
    assert wto['WT2.'].values[-1] == -61.84105024273525


def test_fish():
    '''test TA.FISH'''

    fish = TA.FISH(ohlc)

    assert isinstance(fish, series.Series)
    assert fish.values[-1] == -2.2918315334720125


def test_ichimoku():
    '''test TA.ICHIMOKU'''

    ichi = TA.ICHIMOKU(ohlc)

    assert isinstance(ichi['TENKAN'], series.Series)
    assert isinstance(ichi['KIJUN'], series.Series)
    assert isinstance(ichi['SENKOU'], series.Series)
    assert isinstance(ichi['CHIKOU'], series.Series)

    assert ichi['SENKOU'].values[-1] == 8017.5804297030772


def test_apz():
    '''test TA.APZ'''

    apz = TA.APZ(ohlc)

    assert isinstance(apz['UPPER'], series.Series)
    assert isinstance(apz['LOWER'], series.Series)

    assert apz['UPPER'].values[-1] == 7193.9772579390283


def test_vr():
    '''test TA.VR'''

    with pytest.raises(ValueError):

        vr = TA.VR(ohlc)


def test_sqzmi():
    '''test TA.SQZMI'''

    sqz = TA.SQZMI(ohlc)

    assert isinstance(sqz, series.Series)

    assert not sqz.values[-1]


def test_vpt():
    '''test TA.VPT'''

    vpt = TA.VPT(ohlc)

    assert isinstance(vpt, series.Series)
    assert vpt.values[-1] == 94068.85032709363


def test_fve():
    '''test TA.FVE'''

    fve = TA.FVE(ohlc)

    assert isinstance(fve, series.Series)
    assert -100 < fve.values[-1] < 100
