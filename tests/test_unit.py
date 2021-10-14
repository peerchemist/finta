import pytest
import os
import pandas as pd
from pandas.core import series
from finta import TA


def rootdir():

    return os.path.dirname(os.path.abspath(__file__))


data_file = os.path.join(rootdir(), "data/bittrex_btc-usdt.csv")

ohlc = pd.read_csv(data_file, index_col="date", parse_dates=True)


def test_sma():
    """test TA.ma"""

    ma = TA.SMA(ohlc, 14).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6922.33922063


def test_smm():
    """test TA.SMM"""

    ma = TA.SMM(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6490.0


def test_ssma():
    """test TA.SSMA"""

    ma = TA.SSMA(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6907.53759817


def test_ema():
    """test TA.EMA"""

    ma = TA.EMA(ohlc, 50).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 7606.84391951


def test_dema():
    """test TA.DEMA"""

    ma = TA.DEMA(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6323.41923994


def test_tema():
    """test TA.TEMA"""

    ma = TA.TEMA(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6307.48151844


def test_trima():
    """test TA.TRIMA"""

    ma = TA.TRIMA(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 7464.85307304


def test_trix():
    """test TA.TRIX"""

    ma = TA.TRIX(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == -0.5498364


def test_vama():
    """test TA.VAMA"""

    ma = TA.VAMA(ohlc, 20).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6991.57791258


def test_er():
    """test TA.ER"""

    er = TA.ER(ohlc).round(decimals=8)

    assert isinstance(er, series.Series)
    assert -100 < er.values[-1] < 100


def test_kama():
    """test TA.KAMA"""

    ma = TA.KAMA(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6742.11786502


def test_zlema():
    """test TA.ZLEMA"""

    ma = TA.ZLEMA(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6462.46183365


def test_wma():
    """test TA.WMA"""

    ma = TA.WMA(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6474.47003078


def test_hma():
    """test TA.HMA"""

    ma = TA.HMA(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6186.93727146


def test_evwma():
    """test TA.EVWMA"""

    evwma = TA.EVWMA(ohlc).round(decimals=8)

    assert isinstance(evwma, series.Series)
    assert evwma.values[-1] == 7445.46084062


def test_vwap():
    """test TA.VWAP"""

    ma = TA.VWAP(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 7976.51743477


def test_smma():
    """test TA.SMMA"""

    ma = TA.SMMA(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 8020.2742957


def test_frama():
    """test TA.FRAMA"""

    ma = TA.FRAMA(ohlc).round(decimals=8)

    assert isinstance(ma, series.Series)
    assert ma.values[-1] == 6574.14605454


def test_macd():
    """test TA.MACD"""

    macd = TA.MACD(ohlc).round(decimals=8)

    assert isinstance(macd["MACD"], series.Series)
    assert isinstance(macd["SIGNAL"], series.Series)

    assert macd["MACD"].values[-1] == -419.21923359
    assert macd["SIGNAL"].values[-1] == -372.39851312


def test_ppo():
    """test TA.PPO"""

    ppo = TA.PPO(ohlc).round(decimals=8)

    assert isinstance(ppo["PPO"], series.Series)
    assert isinstance(ppo["SIGNAL"], series.Series)
    assert isinstance(ppo["HISTO"], series.Series)

    assert ppo["PPO"].values[-1] == -5.85551658
    assert ppo["SIGNAL"].values[-1] == -5.05947256
    assert ppo["HISTO"].values[-1] == -0.79604402


def test_vw_macd():
    """test TA.VW_MACD"""

    macd = TA.VW_MACD(ohlc).round(decimals=8)

    assert isinstance(macd["MACD"], series.Series)
    assert isinstance(macd["SIGNAL"], series.Series)

    assert macd["MACD"].values[-1] == -535.21281201
    assert macd["SIGNAL"].values[-1] == -511.64584818


def test_ev_macd():
    """test TA.EV_MACD"""

    macd = TA.EV_MACD(ohlc).round(decimals=8)

    assert isinstance(macd["MACD"], series.Series)
    assert isinstance(macd["SIGNAL"], series.Series)

    assert macd["MACD"].values[-1] == -786.70979566
    assert macd["SIGNAL"].values[-1] == -708.68194345


def test_mom():
    """test TA.MOM"""

    mom = TA.MOM(ohlc).round(decimals=8)

    assert isinstance(mom, series.Series)
    assert mom.values[-1] == -1215.54681371


def test_roc():
    """test TA.ROC"""

    roc = TA.ROC(ohlc).round(decimals=8)

    assert isinstance(roc, series.Series)
    assert roc.values[-1] == -16.0491877

def test_vbm():
    """test TA.VBM"""

    vbm = TA.VBM(ohlc).round(decimals=8)

    assert isinstance(vbm, series.Series)
    assert vbm.values[-1] == -27.57038694

def test_rsi():
    """test TA.RSI"""

    rsi = TA.RSI(ohlc).round(decimals=8)

    assert isinstance(rsi, series.Series)
    assert -100 < rsi.values[-1] < 100


def test_ift_rsi():
    """test TA.IFT_RSI"""

    rsi = TA.IFT_RSI(ohlc).round(decimals=8)

    assert isinstance(rsi, series.Series)
    assert rsi.values[-1] == 0.62803976


def test_dymi():
    """test TA.DYMI"""

    dymi = TA.DYMI(ohlc).round(decimals=8)

    assert isinstance(dymi, series.Series)
    assert dymi.values[-1] == 32.4897564


def test_tr():
    """test TA.TR"""

    tr = TA.TR(ohlc).round(decimals=8)

    assert isinstance(tr, series.Series)
    assert tr.values[-1] == 113.4


def test_atr():
    """test TA.ATR"""

    tr = TA.ATR(ohlc).round(decimals=8)

    assert isinstance(tr, series.Series)
    assert tr.values[-1] == 328.56890383


def test_sar():
    """test TA.SAR"""

    sar = TA.SAR(ohlc).round(decimals=8)

    assert isinstance(sar, series.Series)
    assert sar.values[-1] == 7127.15087821


def test_psar():
    """test TA.PSAR"""

    sar = TA.PSAR(ohlc).round(decimals=8)

    assert isinstance(sar.psar, series.Series)
    assert sar.psar.values[-1] == 7113.5666702


def test_bbands():
    """test TA.BBANDS"""

    bb = TA.BBANDS(ohlc).round(decimals=8)

    assert isinstance(bb["BB_UPPER"], series.Series)
    assert isinstance(bb["BB_MIDDLE"], series.Series)
    assert isinstance(bb["BB_LOWER"], series.Series)

    assert bb["BB_UPPER"].values[-1] == 8212.7979228
    assert bb["BB_MIDDLE"].values[-1] == 7110.55082354
    assert bb["BB_LOWER"].values[-1] == 6008.30372428

def test_mobo():
    """test TA.mobo"""
    
    mbb = TA.MOBO(ohlc).round(decimals=8)

    assert isinstance(mbb["BB_UPPER"], series.Series)
    assert isinstance(mbb["BB_MIDDLE"], series.Series)
    assert isinstance(mbb["BB_LOWER"], series.Series)

    assert mbb["BB_UPPER"].values[-1] == 6919.48336631
    assert mbb["BB_MIDDLE"].values[-1] == 6633.75040888
    assert mbb["BB_LOWER"].values[-1] == 6348.01745146


def test_bbwidth():
    """test TA.BBWIDTH"""

    bb = TA.BBWIDTH(ohlc).round(decimals=8)

    assert isinstance(bb, series.Series)
    assert 0 < bb.values[-1] < 1


def test_percentb():
    """test TA.PERCENT_B"""

    bb = TA.PERCENT_B(ohlc).round(decimals=8)

    assert isinstance(bb, series.Series)
    assert bb.values[-1] == 0.18695874


def test_kc():
    """test TA.KC"""

    ma = TA.ZLEMA(ohlc, 20).round(decimals=8)
    kc = TA.KC(ohlc, MA=ma).round(decimals=8)

    assert isinstance(kc["KC_UPPER"], series.Series)
    assert isinstance(kc["KC_LOWER"], series.Series)

    assert kc["KC_UPPER"].values[-1] == 7014.74943624
    assert kc["KC_LOWER"].values[-1] == 5546.71157518


def test_do():
    """test TA.DO"""

    do = TA.DO(ohlc).round(decimals=8)

    assert isinstance(do["UPPER"], series.Series)
    assert isinstance(do["MIDDLE"], series.Series)
    assert isinstance(do["LOWER"], series.Series)

    assert do["UPPER"].values[-1] == 7770.0
    assert do["MIDDLE"].values[-1] == 7010.0005000000001
    assert do["LOWER"].values[-1] == 6250.0010000000002


def test_dmi():
    """test TA.DMI"""

    dmi = TA.DMI(ohlc).round(decimals=8)

    assert isinstance(dmi["DI+"], series.Series)
    assert isinstance(dmi["DI-"], series.Series)

    assert dmi["DI+"].values[-1] == 7.07135289
    assert dmi["DI-"].values[-1] == 28.62895818


def test_adx():
    """test TA.ADX"""

    adx = TA.ADX(ohlc).round(decimals=8)

    assert isinstance(adx, series.Series)
    assert adx.values[-1] == 46.43950615


def test_stoch():
    """test TA.STOCH"""

    st = TA.STOCH(ohlc).round(decimals=8)

    assert isinstance(st, series.Series)
    assert 0 < st.values[-1] < 100


def test_stochd():
    """test TA.STOCHD"""

    st = TA.STOCHD(ohlc).round(decimals=8)

    assert isinstance(st, series.Series)
    assert 0 < st.values[-1] < 100


def test_stochrsi():
    """test TA.STOCRSI"""

    st = TA.STOCHRSI(ohlc).round(decimals=8)

    assert isinstance(st, series.Series)
    assert 0 < st.values[-1] < 100


def test_williams():
    """test TA.WILLIAMS"""

    w = TA.WILLIAMS(ohlc).round(decimals=8)

    assert isinstance(w, series.Series)
    assert -100 < w.values[-1] < 0


def test_uo():
    """test TA.UO"""

    uo = TA.UO(ohlc).round(decimals=8)

    assert isinstance(uo, series.Series)
    assert 0 < uo.values[-1] < 100


def test_ao():
    """test TA.AO"""

    ao = TA.AO(ohlc).round(decimals=8)

    assert isinstance(ao, series.Series)
    assert ao.values[-1] == -957.63459033


def test_mi():
    """test TA.MI"""

    mi = TA.MI(ohlc).round(decimals=8)

    assert isinstance(mi, series.Series)
    assert mi.values[-1] == 23.92808696


def test_bop():
    """test TA.BOP"""

    bop = TA.BOP(ohlc).round(decimals=8)

    assert isinstance(bop, series.Series)
    assert bop.values[-1] == 0.03045138


def test_vortex():
    """test TA.VORTEX"""

    v = TA.VORTEX(ohlc).round(decimals=8)

    assert isinstance(v["VIp"], series.Series)
    assert isinstance(v["VIm"], series.Series)

    assert v["VIp"].values[-1] == 0.76856105
    assert v["VIm"].values[-1] == 1.27305188


def test_kst():
    """test TA.KST"""

    kst = TA.KST(ohlc).round(decimals=8)

    assert isinstance(kst["KST"], series.Series)
    assert isinstance(kst["signal"], series.Series)

    assert kst["KST"].values[-1] == -157.42229442
    assert kst["signal"].values[-1] == -132.10367593


def test_tsi():
    """test TA.TSI"""

    tsi = TA.TSI(ohlc).round(decimals=8)

    assert isinstance(tsi["TSI"], series.Series)
    assert isinstance(tsi["signal"], series.Series)

    assert tsi["TSI"].values[-1] == -32.12837201
    assert tsi["signal"].values[-1] == -26.94173827


def test_tp():
    """test TA.TP"""

    tp = TA.TP(ohlc).round(decimals=8)

    assert isinstance(tp, series.Series)
    assert tp.values[-1] == 6429.01772876


def test_adl():

    adl = TA.ADL(ohlc).round(decimals=8)

    assert isinstance(adl, series.Series)
    assert adl.values[-1] == 303320.96403697


def test_chaikin():
    """test TA.CHAIKIN"""

    c = TA.CHAIKIN(ohlc).round(decimals=8)

    assert isinstance(c, series.Series)
    assert c.values[-1] == -378.66969549


def test_mfi():
    """test TA.MFI"""

    mfi = TA.MFI(ohlc).round(decimals=8)

    assert isinstance(mfi, series.Series)
    assert 0 < mfi.values[-1] < 100


def test_obv():
    """test TA.OBV"""

    o = TA.OBV(ohlc).round(decimals=8)

    assert isinstance(o, series.Series)
    assert o.values[-1] == -6726.6904375


def test_wobv():
    """test TA.OBV"""

    o = TA.WOBV(ohlc).round(decimals=8)

    assert isinstance(o, series.Series)
    assert o.values[-1] == -85332065.01331231


def test_vzo():
    """test TA.VZO"""

    vzo = TA.VZO(ohlc)

    assert isinstance(vzo, series.Series)
    assert -85 < vzo.values[-1] < 85


def test_pzo():
    """test TA.PZO"""

    pzo = TA.PZO(ohlc)

    assert isinstance(pzo, series.Series)
    assert -85 < pzo.values[-1] < 85


def test_efi():
    """test TA.EFI"""

    efi = TA.EFI(ohlc)

    assert isinstance(efi, series.Series)
    assert efi.values[1] > 0
    assert efi.values[2] > 0

    assert efi.values[-2] < 0
    assert efi.values[-1] < 0


def test_cfi():
    """test TA.CFI"""

    cfi = TA.CFI(ohlc).round(decimals=8)

    assert isinstance(cfi, series.Series)
    assert cfi.values[-1] == -84856289.556287795


def test_ebbp():
    """test TA.EBBP"""

    eb = TA.EBBP(ohlc).round(decimals=8)

    assert isinstance(eb["Bull."], series.Series)
    assert isinstance(eb["Bear."], series.Series)
    assert eb["Bull."].values[-1] == -285.40231904


def test_emv():
    """test TA.EMV"""

    emv = TA.EMV(ohlc).round(decimals=1)

    assert isinstance(emv, series.Series)
    assert emv.values[-1] == -26103140.8
                             
def test_cci():
    """test TA.CCI"""

    cci = TA.CCI(ohlc).round(decimals=8)

    assert isinstance(cci, series.Series)
    assert cci.values[-1] == -91.76341956


def test_basp():
    """test TA.BASP"""

    basp = TA.BASP(ohlc).round(decimals=8)

    assert isinstance(basp["Buy."], series.Series)
    assert isinstance(basp["Sell."], series.Series)

    assert basp["Buy."].values[-1] == 0.06691681
    assert basp["Sell."].values[-1] == 0.0914869


def test_baspn():
    """test TA.BASPN"""

    basp = TA.BASPN(ohlc).round(decimals=8)

    assert isinstance(basp["Buy."], series.Series)
    assert isinstance(basp["Sell."], series.Series)

    assert basp["Buy."].values[-1] == 0.56374213
    assert basp["Sell."].values[-1] == 0.74103021


def test_cmo():
    """test TA.CMO"""

    cmo = TA.CMO(ohlc)

    assert isinstance(cmo, series.Series)
    assert -100 < cmo.values[-1] < 100


def test_chandelier():
    """test TA.CHANDELIER"""

    chan = TA.CHANDELIER(ohlc).round(decimals=8)

    assert isinstance(chan["Long."], series.Series)
    assert isinstance(chan["Short."], series.Series)

    assert chan["Long."].values[-1] == 6801.59276465
    assert chan["Short."].values[-1] == 7091.40723535


def test_qstick():
    """test TA.QSTICK"""

    q = TA.QSTICK(ohlc).round(decimals=8)

    assert isinstance(q, series.Series)
    assert q.values[-1] == 0.24665616


def test_tmf():

    with pytest.raises(NotImplementedError):
        tmf = TA.TMF(ohlc)


def test_wto():
    """test TA.WTO"""

    wto = TA.WTO(ohlc).round(decimals=8)

    assert isinstance(wto["WT1."], series.Series)
    assert isinstance(wto["WT2."], series.Series)

    assert wto["WT1."].values[-1] == -60.29006991
    assert wto["WT2."].values[-1] == -61.84105024


def test_fish():
    """test TA.FISH"""

    fish = TA.FISH(ohlc).round(decimals=8)

    assert isinstance(fish, series.Series)
    assert fish.values[-1] == -2.29183153


def test_ichimoku():
    """test TA.ICHIMOKU"""

    ichi = TA.ICHIMOKU(ohlc, 10, 25).round(decimals=8)

    assert isinstance(ichi["TENKAN"], series.Series)
    assert isinstance(ichi["KIJUN"], series.Series)
    assert isinstance(ichi["SENKOU"], series.Series)
    assert isinstance(ichi["CHIKOU"], series.Series)

    assert ichi["TENKAN"].values[-1] == 6911.5 
    assert ichi["KIJUN"].values[-1] == 6946.5
    assert ichi["SENKOU"].values[-1] == 8243.0 
    assert ichi["CHIKOU"].values[-27] == 6420.45318629


def test_apz():
    """test TA.APZ"""

    apz = TA.APZ(ohlc).round(decimals=8)

    assert isinstance(apz["UPPER"], series.Series)
    assert isinstance(apz["LOWER"], series.Series)

    assert apz["UPPER"].values[-1] == 7193.97725794


def test_sqzmi():
    """test TA.SQZMI"""

    sqz = TA.SQZMI(ohlc)

    assert isinstance(sqz, series.Series)

    assert not sqz.values[-1]


def test_vpt():
    """test TA.VPT"""

    vpt = TA.VPT(ohlc).round(decimals=8)

    assert isinstance(vpt, series.Series)
    assert vpt.values[-1] == 94068.85032709


def test_fve():
    """test TA.FVE"""

    fve = TA.FVE(ohlc)

    assert isinstance(fve, series.Series)
    assert -100 < fve.values[-1] < 100


def test_vfi():
    """test TA.VFI"""

    vfi = TA.VFI(ohlc).round(decimals=8)

    assert isinstance(vfi, series.Series)
    assert vfi.values[-1] == -6.49159549


def test_pivot():
    """test TA.PIVOT"""

    pivot = TA.PIVOT(ohlc).round(decimals=8)

    assert isinstance(pivot["pivot"], series.Series)
    assert isinstance(pivot["s1"], series.Series)
    assert isinstance(pivot["s2"], series.Series)
    assert isinstance(pivot["s3"], series.Series)
    assert isinstance(pivot["r1"], series.Series)
    assert isinstance(pivot["r2"], series.Series)
    assert isinstance(pivot["r3"], series.Series)

    assert pivot["pivot"].values[-1] == 6467.40629761

    assert pivot["s1"].values[-1] == 6364.00470239
    assert pivot["s2"].values[-1] == 6311.00940479
    assert pivot["s3"].values[-1] == 6207.60780957
    assert pivot["s4"].values[-1] == 6104.20621436

    assert pivot["r1"].values[-1] == 6520.40159521
    assert pivot["r2"].values[-1] == 6623.80319043
    assert pivot["r3"].values[-1] == 6676.79848803
    assert pivot["r4"].values[-1] == 6729.79378564


def test_pivot_fib():
    """test TA.PIVOT_FIB"""

    pivot = TA.PIVOT_FIB(ohlc).round(decimals=8)

    assert isinstance(pivot["pivot"], series.Series)
    assert isinstance(pivot["s1"], series.Series)
    assert isinstance(pivot["s2"], series.Series)
    assert isinstance(pivot["s3"], series.Series)
    assert isinstance(pivot["r1"], series.Series)
    assert isinstance(pivot["r2"], series.Series)
    assert isinstance(pivot["r3"], series.Series)

    assert pivot["pivot"].values[-1] == 6467.40629761

    assert pivot["s1"].values[-1] == 6407.66268455
    assert pivot["s2"].values[-1] == 6370.75301784
    assert pivot["s3"].values[-1] == 6311.00940479
    assert pivot["s4"].values[-1] == 6251.26579173

    assert pivot["r1"].values[-1] == 6527.14991066
    assert pivot["r2"].values[-1] == 6564.05957737
    assert pivot["r3"].values[-1] == 6623.80319043
    assert pivot["r4"].values[-1] == 6683.54680348


def test_msd():
    """test TA.MSD"""

    msd = TA.MSD(ohlc).round(decimals=8)

    assert isinstance(msd, series.Series)
    assert msd.values[-1] == 542.25201592


def test_stc():
    """test TA.STC"""

    stc = TA.STC(ohlc).round(decimals=2)

    assert isinstance(stc, series.Series)
    assert 0 <= stc.values[-1] <= 100


def test_evstc():
    """test TA.EVSTC"""

    stc = TA.EVSTC(ohlc).round(decimals=2)

    assert isinstance(stc, series.Series)
    assert 0 <= stc.values[-1] <= 100


def test_williams_fractal():
    """test TA.WILLIAMS_FRACTAL"""

    fractals = TA.WILLIAMS_FRACTAL(ohlc)

    assert isinstance(fractals["BullishFractal"], series.Series)
    assert isinstance(fractals["BearishFractal"], series.Series)
    assert fractals.BearishFractal.values[-3] == 0
    assert fractals.BullishFractal.values[-3] == 0


def test_vc():
    """test TA.VC"""

    vc = TA.VC(ohlc).round(decimals=8)

    assert isinstance(vc["Value Chart Open"], series.Series)
    assert vc.values[-1][0] == 0.50469864
    assert vc.values[-1][-1] == -0.87573258

def test_sma():
    """test TA.WAVEPM"""

    wavepm = TA.WAVEPM(ohlc, 14, 100, "close").round(decimals=8)

    assert isinstance(wavepm, series.Series)
    assert wavepm.values[-1] == 0.83298565