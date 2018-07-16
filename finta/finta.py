
import pandas as pd


class TA:

    @classmethod
    def SMA(cls, ohlc, period=41, column='close'):
        """
        Simple moving average - rolling mean in pandas lingo. Also known as 'MA'.
        The simple moving average (SMA) is the most basic of the moving averages used for trading.
        """

        return pd.Series(ohlc[column].rolling(center=False, window=period,
                         min_periods=period - 1).mean(),
                         name='{0} period SMA'.format(period))

    @classmethod
    def SMM(cls, ohlc, period=9, column='close'):
        """
        Simple moving median, an alternative to moving average. SMA, when used to estimate the underlying trend in a time series,
        is susceptible to rare events such as rapid shocks or other anomalies. A more robust estimate of the trend is the simple moving median over n time periods.
        """

        return pd.Series(ohlc[column].rolling(center=False, window=period,
                         min_periods=period - 1).median(),
                         name='{0} period SMM'.format(period))

    @classmethod
    def EMA(cls, ohlc, period=9, column='close'):  ## EWMA
        """
        Exponential Weighted Moving Average - Like all moving average indicators, they are much better suited for trending markets.
        When the market is in a strong and sustained uptrend, the EMA indicator line will also show an uptrend and vice-versa for a down trend.
        EMAs are commonly used in conjunction with other indicators to confirm significant market moves and to gauge their validity.
        """

        return pd.Series(ohlc[column].ewm(ignore_na=False,
                                          min_periods=period - 1,
                                          span=period).mean(),
                                          name='{0} period EMA'.format(period))

    @classmethod
    def DEMA(cls, ohlc, period=9, column='close'):
        """
        Double Exponential Moving Average - attempts to remove the inherent lag associated to Moving Averages
         by placing more weight on recent values. The name suggests this is achieved by applying a double exponential
        smoothing which is not the case. The name double comes from the fact that the value of an EMA (Exponential Moving Average) is doubled.
        To keep it in line with the actual data and to remove the lag the value 'EMA of EMA' is subtracted from the previously doubled EMA.
        Because EMA(EMA) is used in the calculation, DEMA needs 2 * period -1 samples to start producing values in contrast to the period
        samples needed by a regular EMA
        """

        DEMA = 2 * cls.EMA(ohlc, period) - cls.EMA(ohlc, period).ewm(ignore_na=False,
                                                                     min_periods=period - 1,
                                                                     span=period).mean()

        return pd.Series(DEMA, name='{0} period DEMA'.format(period))

    @classmethod
    def TEMA(cls, ohlc, period=9):
        """
        Triple exponential moving average - attempts to remove the inherent lag associated to Moving Averages by placing more weight on recent values.
        The name suggests this is achieved by applying a triple exponential smoothing which is not the case. The name triple comes from the fact that the
        value of an EMA (Exponential Moving Average) is triple.
        To keep it in line with the actual data and to remove the lag the value 'EMA of EMA' is subtracted 3 times from the previously tripled EMA.
        Finally 'EMA of EMA of EMA' is added.
        Because EMA(EMA(EMA)) is used in the calculation, TEMA needs 3 * period - 2 samples to start producing values in contrast to the period samples
        needed by a regular EMA.
        """

        triple_ema = 3 * cls.EMA(ohlc, period)
        ema_ema_ema = cls.EMA(ohlc, period).ewm(ignore_na=False,
                                                span=period).mean().ewm(ignore_na=False,
                                                                                         span=period).mean()

        TEMA = triple_ema - 3 * cls.EMA(ohlc, period).ewm(ignore_na=False,
                                                          min_periods=period - 1,
                                                          span=period).mean() + ema_ema_ema

        return pd.Series(TEMA, name='{0} period TEMA'.format(period))

    @classmethod
    def TRIMA(cls, ohlc, period=18):  # TMA
        """
        The Triangular Moving Average (TRIMA) [also known as TMA] represents an average of prices,
        but places weight on the middle prices of the time period.
        The calculations double-smooth the data using a window width that is one-half the length of the series.
        source: https://www.thebalance.com/triangular-moving-average-tma-description-and-uses-1031203
        """

        SMA = cls.SMA(ohlc, period).rolling(window=period,
                                            min_periods=period-1).sum()

        return pd.Series(SMA / period, name='{0} period TRIMA'.format(period))

    @classmethod
    def TRIX(cls, ohlc, period=15):
        """
        The Triple Exponential Moving Average Oscillator (TRIX) by Jack Hutson is a momentum indicator that oscillates around zero.
        It displays the percentage rate of change between two triple smoothed exponential moving averages.
        To calculate TRIX we calculate triple smoothed EMA3 of n periods and then substract previous period EMA3 value
        from last EMA3 value and divide the result with yesterdays EMA3 value.
        """

        EMA1 = cls.EMA(ohlc, period)
        EMA2 = EMA1.ewm(span=period).mean()
        EMA3 = EMA2.ewm(span=period).mean()
        TRIX = (EMA3 - EMA3.diff()) / EMA3.diff()

        return pd.Series(TRIX, name='{0} period TRIX'.format(period))

    @classmethod
    def AMA(cls, ohlc, period=None, column='close'):
        """
        This indicator is either quick, or slow, to signal a market entry depending on the efficiency of the move in the market.
        """
        raise NotImplementedError

    @classmethod
    def LWMA(cls, ohlc, period=None, column='close'):
        """
        Linear Weighted Moving Average
        """
        raise NotImplementedError

    @classmethod
    def VAMA(cls, ohlcv, period=8, column='close'):
        """
        Volume Adjusted Moving Average
        """

        vp = ohlcv['volume'] * ohlcv['close']
        volsum = ohlcv['volume'].rolling(window=period).mean()
        volRatio = pd.Series(vp / volsum, name='VAMA')
        cumSum = (volRatio * ohlcv[column]).rolling(window=period).sum()
        cumDiv = volRatio.rolling(window=period).sum()

        return pd.Series(cumSum / cumDiv, name='{0} period VAMA'.format(period))

    @classmethod
    def VIDYA(cls, ohlcv, period=9, smoothing_period=12):
        """ Vidya (variable index dynamic average) indicator is a modification of the traditional Exponential Moving Average (EMA) indicator.
        The main difference between EMA and Vidya is in the way the smoothing factor F is calculated.
        In EMA the smoothing factor is a constant value F=2/(period+1);
        in Vidya the smoothing factor is variable and depends on bar-to-bar price movements."""

        raise NotImplementedError

    @classmethod
    def ER(cls, ohlc, period=10):
        """The Kaufman Efficiency indicator is an oscillator indicator that oscillates between +100 and -100, where zero is the center point.
         +100 is upward forex trending market and -100 is downwards trending markets."""

        change = ohlc['close'].diff(period).abs()
        volatility = ohlc['close'].diff().abs().rolling(window=period).sum()

        return pd.Series(change / volatility, name='{0} period ER'.format(period))

    @classmethod
    def KAMA(cls, ohlc, er=10, ema_fast=2, ema_slow=30, period=20):
        """Developed by Perry Kaufman, Kaufman's Adaptive Moving Average (KAMA) is a moving average designed to account for market noise or volatility.
        Its main advantage is that it takes into consideration not just the direction, but the market volatility as well."""

        er = cls.ER(ohlc, er)
        fast_alpha = 2 / (ema_fast + 1)
        slow_alpha = 2 / (ema_slow + 1)
        sc = pd.Series((er * (fast_alpha - slow_alpha) + slow_alpha) ** 2,
                       name='smoothing_constant')  ## smoothing constant

        sma = pd.Series(ohlc['close'].rolling(period).mean(), name='SMA')  ## first KAMA is SMA
        kama = []
        # Current KAMA = Prior KAMA + smoothing_constant * (Price - Prior KAMA)
        for s, ma, price in zip(sc.iteritems(), sma.shift().iteritems(), ohlc['close'].iteritems()):
            try:
                kama.append(kama[-1] + s[1] * (price[1] - kama[-1]))
            except (IndexError, TypeError):
                if pd.notnull(ma[1]):
                    kama.append(ma[1] + s[1] * (price[1] - ma[1]))
                else:
                    kama.append(None)

        sma['KAMA'] = pd.Series(kama, index=sma.index,
                                name='{0} period KAMA.'.format(period))  ## apply the kama list to existing index
        return sma['KAMA']

    @classmethod
    def ZLEMA(cls, ohlc, period=26):
        """ZLEMA is an abbreviation of Zero Lag Exponential Moving Average. It was developed by John Ehlers and Rick Way.
        ZLEMA is a kind of Exponential moving average but its main idea is to eliminate the lag arising from the very nature of the moving averages
        and other trend following indicators. As it follows price closer, it also provides better price averaging and responds better to price swings."""

        lag = (period - 1) / 2
        return pd.Series((ohlc['close'] + (ohlc['close'].diff(lag))), name='{0} period ZLEMA.'.format(period))

    @classmethod
    def WMA(cls, ohlc, period=9, column='close'):
        """WMA stands for weighted moving average. It helps to smooth the price curve for better trend identification.
        It places even greater importance on recent data than the EMA does."""

        d = (period * (period + 1)) / 2  # denominator
        rev = ohlc[column].iloc[::-1]  ## reverse the series
        wma = []


        def _chunks(series, period):  ## split into chunks of n elements
            for i in enumerate(series):
                c = rev.iloc[i[0]:i[0] + period]
                if len(c) != period:
                    yield None
                else:
                    yield c


        def _wma(chunk, period):  ## calculate wma for each chunk
            w = []
            for price, i in zip(chunk.iloc[::-1].items(), range(period + 1)[1:]):
                w.append(price[1] * i / d)
            return sum(w)


        for i in _chunks(rev, period):
            try:
                wma.append(_wma(i, period))
            except:
                wma.append(None)

        wma.reverse()  ## reverse the wma list to match the Series
        ohlc['WMA'] = pd.Series(wma, index=ohlc.index)  ## apply the wma list to existing index
        return pd.Series(ohlc['WMA'], name='{0} period WMA.'.format(period))

    @classmethod
    def HMA(cls, ohlc, period=16):
        """
        HMA indicator is a common abbreviation of Hull Moving Average.
        The average was developed by Allan Hull and is used mainly to identify the current market trend.
        Unlike SMA (simple moving average) the curve of Hull moving average is considerably smoother.
        Moreover, because its aim is to minimize the lag between HMA and price it does follow the price activity much closer.
        It is used especially for middle-term and long-term trading.
        """

        import math

        wma_a = cls.WMA(ohlc, int(period / 2)) * 2
        wma_b = cls.WMA(ohlc, period)
        deltawma = wma_a - wma_b

        # now calculate WMA of deltawma for sqrt(period)
        p = round(math.sqrt(period))  # period
        d = (p * (p + 1)) / 2
        rev = deltawma.iloc[::-1]  ## reverse the series
        wma = []


        def _chunks(series, period):  ## split into chunks of n elements
            for i in enumerate(series):
                c = rev.iloc[i[0]:i[0] + period]
                if len(c) != period:
                    yield None
                else:
                    yield c


        def _wma(chunk, period):  ## calculate wma for each chunk
            w = []
            for price, i in zip(chunk.iloc[::-1].items(), range(period + 1)[1:]):
                w.append(price[1] * i / d)
            return sum(w)


        for i in _chunks(rev, p):
            try:
                wma.append(_wma(i, p))
            except:
                wma.append(None)

        wma.reverse()
        deltawma['_WMA'] = pd.Series(wma, index=ohlc.index)
        return pd.Series(deltawma['_WMA'], name='{0} period HMA.'.format(period))

    @classmethod
    def VWAP(cls, ohlcv):
        """
        The volume weighted average price (VWAP) is a trading benchmark used especially in pension plans.
        VWAP is calculated by adding up the dollars traded for every transaction (price multiplied by number of shares traded) and then dividing
        by the total shares traded for the day.
        """

        return pd.Series(((ohlcv['volume'] * cls.TP(ohlcv)).cumsum()) / ohlcv['volume'].cumsum(),
                         name='VWAP.')

    @classmethod
    def SMMA(cls, ohlc, period=42, column='close'):
        """The SMMA gives recent prices an equal weighting to historic prices."""

        return pd.Series(ohlc[column].ewm(alpha=1 / period).mean(), name='SMMA')

    @classmethod
    def ALMA(cls, ohlc, period=9, sigma=6, offset=0.85):
        """Arnaud Legoux Moving Average."""

        """dataWindow = _.last(data, period)
        size = _.size(dataWindow)
        m = offset * (size - 1)
        s = size / sigma
        sum = 0
        norm = 0
        for i in [size-1..0] by -1
        coeff = Math.exp(-1 * (i - m) * (i - m) / 2 * s * s)
        sum = sum + dataWindow[i] * coeff
        norm = norm + coeff
        return sum / norm"""

        raise NotImplementedError

    @classmethod
    def MAMA(cls, ohlc, period=16):
        """MESA Adaptive Moving Average"""
        raise NotImplementedError

    @classmethod
    def FRAMA(cls, ohlc, period=16):
        """Fractal Adaptive Moving Average"""
        raise NotImplementedError

    @classmethod
    def MACD(cls, ohlc, period_fast=12, period_slow=26, signal=9):
        """
        MACD, MACD Signal and MACD difference.
        The MACD Line oscillates above and below the zero line, which is also known as the centerline.
        These crossovers signal that the 12-day EMA has crossed the 26-day EMA. The direction, of course, depends on the direction of the moving average cross.
        Positive MACD indicates that the 12-day EMA is above the 26-day EMA. Positive values increase as the shorter EMA diverges further from the longer EMA.
        This means upside momentum is increasing. Negative MACD values indicates that the 12-day EMA is below the 26-day EMA.
        Negative values increase as the shorter EMA diverges further below the longer EMA. This means downside momentum is increasing.

        Signal line crossovers are the most common MACD signals. The signal line is a 9-day EMA of the MACD Line.
        As a moving average of the indicator, it trails the MACD and makes it easier to spot MACD turns.
        A bullish crossover occurs when the MACD turns up and crosses above the signal line.
        A bearish crossover occurs when the MACD turns down and crosses below the signal line.
        """

        EMA_fast = pd.Series(ohlc['close'].ewm(ignore_na=False, min_periods=period_slow - 1, span=period_fast).mean(),
                             name='EMA_fast')
        EMA_slow = pd.Series(ohlc['close'].ewm(ignore_na=False, min_periods=period_slow - 1, span=period_slow).mean(),
                             name='EMA_slow')
        MACD = pd.Series(EMA_fast - EMA_slow, name='MACD')
        MACD_signal = pd.Series(MACD.ewm(ignore_na=False, span=signal).mean(), name='SIGNAL')

        return pd.concat([MACD, MACD_signal], axis=1)
    
    @classmethod
    def PPO(cls, ohlc, period_fast=12, period_slow=26, signal=9):
        """
        PPO, PPO Signal and PPO difference.
        As with MACD, the PPO reflects the convergence and divergence of two moving averages.
        While MACD measures the absolute difference between two moving averages, PPO makes this a relative value by dividing the difference by the slower moving average
        """

        EMA_fast = pd.Series(ohlc['close'].ewm(ignore_na=False, min_periods=period_slow - 1, span=period_fast).mean(),
                             name='EMA_fast')
        EMA_slow = pd.Series(ohlc['close'].ewm(ignore_na=False, min_periods=period_slow - 1, span=period_slow).mean(),
                             name='EMA_slow')
        PPO = pd.Series(((EMA_fast - EMA_slow)/EMA_slow) * 100, name='PPO')
        PPO_signal = pd.Series(PPO.ewm(ignore_na=False, span=signal).mean(), name='SIGNAL')
        PPO_histo = pd.Series(PPO - PPO_signal, name='HISTO')

        return pd.concat([PPO, PPO_signal, PPO_histo], axis=1)

    @classmethod
    def VW_MACD(cls, ohlcv, period_fast=12, period_slow=26, signal=9):
        '''"Volume-Weighted MACD" is an indicator that shows how a volume-weighted moving average can be used to calculate moving average convergence/divergence (MACD).
        This technique was first used by Buff Dormeier, CMT, and has been written about since at least 2002.'''

        vp = ohlcv["volume"] * ohlcv["close"]
        _fast = pd.Series((vp.ewm(ignore_na=False, min_periods=period_fast-1, span=period_fast).mean()) /
                           (ohlcv['volume'].ewm(ignore_na=False, min_periods=period_fast-1, span=period_fast).mean()),
                           name='_fast')

        _slow = pd.Series((vp.ewm(ignore_na=False, min_periods=period_slow-1, span=period_slow).mean()) /
                           (ohlcv['volume'].ewm(ignore_na=False, min_periods=period_slow-1, span=period_slow).mean()),
                           name='_slow')

        MACD = pd.Series(_fast - _slow, name="MACD")
        MACD_signal = pd.Series(MACD.ewm(ignore_na=False, span=signal).mean(), name="SIGNAL")

        return pd.concat([MACD, MACD_signal], axis=1)

    @classmethod
    def MOM(cls, ohlc, period=10):
        """Market momentum is measured by continually taking price differences for a fixed time interval.
        To construct a 10-day momentum line, simply subtract the closing price 10 days ago from the last closing price.
        This positive or negative value is then plotted around a zero line."""

        return pd.Series(ohlc['close'].diff(period), name="MOM".format(period))

    @classmethod
    def ROC(cls, ohlc, period=12):
        """The Rate-of-Change (ROC) indicator, which is also referred to as simply Momentum,
        is a pure momentum oscillator that measures the percent change in price from one period to the next.
        The ROC calculation compares the current price with the price “n” periods ago."""

        return pd.Series((ohlc['close'].diff(period) / ohlc['close'][-period]) * 100, name='ROC')

    @classmethod
    def RSI(cls, ohlc, period=14):
        """Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
        RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30.
        Signals can also be generated by looking for divergences, failure swings and centerline crossovers.
        RSI can also be used to identify the general trend."""

        ## get the price diff
        delta = ohlc['close'].diff()[1:]

        ## positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # EMAs of ups and downs
        _gain = up.ewm(span=period, min_periods=period - 1).mean()
        _loss = down.abs().ewm(span=period, min_periods=period - 1).mean()

        RS = _gain / _loss
        return pd.Series(100 - (100 / (1 + RS)), name='RSI')

    @classmethod
    def IFT_RSI(cls, ohlc, rsi_period=14, wma_period=9):
        """Modified Inverse Fisher Transform applied on RSI.
        Suggested method to use any IFT indicator is to buy when the indicator crosses over –0.5 or crosses over +0.5
        if it has not previously crossed over –0.5 and to sell short when the indicators crosses under +0.5 or crosses under –0.5
        if it has not previously crossed under +0.5."""

        v1 = pd.Series(0.1 * (cls.RSI(ohlc, rsi_period) - 50), name='v1')

        ### v2 = WMA(wma_period) of v1
        d = (wma_period * (wma_period + 1)) / 2  # denominator
        rev = v1.iloc[::-1]  ## reverse the series
        wma = []


        def _chunks(series, period):  ## split into chunks of n elements
            for i in enumerate(series):
                c = rev.iloc[i[0]:i[0] + period]
                if len(c) != period:
                    yield None
                else:
                    yield c


        def _wma(chunk, period):  ## calculate wma for each chunk
            w = []
            for price, i in zip(chunk.iloc[::-1].items(), range(period + 1)[1:]):
                w.append(price[1] * i / d)
            return sum(w)


        for i in _chunks(rev, wma_period):
            try:
                wma.append(_wma(i, wma_period))
            except:
                wma.append(None)

        wma.reverse()  ## reverse the wma list to match the Series
        ###
        v1['v2'] = pd.Series(wma, index=v1.index)
        fish = pd.Series(((2 * v1['v2']) - 1) ** 2 / ((2 * v1['v2']) + 1) ** 2, name='IFT_RSI')
        return fish

    @classmethod
    def SWI(cls, ohlc, period=16):
        """Sine Wave indicator"""
        raise NotImplementedError


    @classmethod
    def TR(cls, ohlc):
        """True Range is the maximum of three price ranges.
        Most recent period's high minus the most recent period's low.
        Absolute value of the most recent period's high minus the previous close.
        Absolute value of the most recent period's low minus the previous close."""

        TR1 = pd.Series(ohlc['high'] - ohlc['low']).abs()  # True Range = High less Low

        TR2 = pd.Series(ohlc['high'] - ohlc['close'].shift(),
                        ).abs()  # True Range = High less Previous Close

        TR3 = pd.Series(ohlc['close'].shift() - ohlc['low'],
                        ).abs()  # True Range = Previous Close less Low

        _TR = pd.concat([TR1, TR2, TR3], axis=1)

        _TR['TR'] = _TR.max(axis=1)

        return pd.Series(_TR['TR'], name="TR")

    @classmethod
    def ATR(cls, ohlc, period=14):
        """Average True Range is moving average of True Range."""

        TR = cls.TR(ohlc)
        return pd.Series(TR.rolling(center=False, window=period, min_periods=period - 1).mean(),
                         name='{0} period ATR'.format(period))

    @classmethod
    def SAR(cls, ohlc, af=0.02, amax=0.2):
        """SAR stands for “stop and reverse,” which is the actual indicator used in the system.
        SAR trails price as the trend extends over time. The indicator is below prices when prices are rising and above prices when prices are falling.
        In this regard, the indicator stops and reverses when the price trend reverses and breaks above or below the indicator."""
        high, low = ohlc.high, ohlc.low

        # Starting values
        sig0, xpt0, af0 = True, high[0], af
        _sar = [low[0] - (high - low).std()]

        for i in range(1, len(ohlc)):
            sig1, xpt1, af1 = sig0, xpt0, af0

            lmin = min(low[i - 1], low[i])
            lmax = max(high[i - 1], high[i])

            if sig1:
                sig0 = low[i] > _sar[-1]
                xpt0 = max(lmax, xpt1)
            else:
                sig0 = high[i] >= _sar[-1]
                xpt0 = min(lmin, xpt1)

            if sig0 == sig1:
                sari = _sar[-1] + (xpt1 - _sar[-1])*af1
                af0 = min(amax, af1 + af)

                if sig0:
                    af0 = af0 if xpt0 > xpt1 else af1
                    sari = min(sari, lmin)
                else:
                    af0 = af0 if xpt0 < xpt1 else af1
                    sari = max(sari, lmax)
            else:
                af0 = af
                sari = xpt0

            _sar.append(sari)

        return pd.Series(_sar, index=ohlc.index)

    @classmethod
    def BBANDS(cls, ohlc, period=20, MA=None, column='close'):
        """
         Developed by John Bollinger, Bollinger Bands® are volatility bands placed above and below a moving average.
         Volatility is based on the standard deviation, which changes as volatility increases and decreases.
         The bands automatically widen when volatility increases and narrow when volatility decreases.

         This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
         Pass desired moving average as <MA> argument. For example BBANDS(MA=TA.KAMA(20)).
         """ 
         
        std = ohlc["close"].rolling(window=period).std()

        if not isinstance(MA, pd.core.series.Series):
            middle_band = pd.Series(cls.SMA(ohlc, period), name='BB_MIDDLE')
        else:
            middle_band = pd.Series(MA, name='BB_MIDDLE')

        upper_bb = pd.Series(middle_band + (2 * std), name='BB_UPPER')
        lower_bb = pd.Series(middle_band - (2 * std), name='BB_LOWER')

        return pd.concat([upper_bb, middle_band, lower_bb], axis=1)

    @classmethod
    def BBWIDTH(cls, ohlc, period=20, MA=None, column='close'):
        """Bandwidth tells how wide the Bollinger Bands are on a normalized basis."""

        BB = TA.BBANDS(ohlc, period, MA, column)

        return pd.Series((BB['BB_UPPER'] - BB['BB_LOWER']) / BB['BB_MIDDLE'],
                         name='{0} period BBWITH'.format(period))

    @classmethod
    def PERCENT_B(cls, ohlc, period=20, MA=None, column='close'):
        """
        %b (pronounced 'percent b') is derived from the formula for Stochastics and shows where price is in relation to the bands.
        %b equals 1 at the upper band and 0 at the lower band.
        """

        BB = TA.BBANDS(ohlc, period, MA, column)
        percent_b = pd.Series((ohlc['close'] - BB['BB_LOWER']) / (BB['BB_UPPER'] - BB['BB_LOWER']), name='%b')

        return percent_b

    @classmethod
    def KC(cls, ohlc, period=20, atr_period=10, MA=None):
        """Keltner Channels [KC] are volatility-based envelopes set above and below an exponential moving average.
        This indicator is similar to Bollinger Bands, which use the standard deviation to set the bands.
        Instead of using the standard deviation, Keltner Channels use the Average True Range (ATR) to set channel distance.
        The channels are typically set two Average True Range values above and below the 20-day EMA.
        The exponential moving average dictates direction and the Average True Range sets channel width.
        Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and channel direction.
        Channels can also be used to identify overbought and oversold levels when the trend is flat."""

        if not isinstance(MA, pd.core.series.Series):
            middle = pd.Series(cls.EMA(ohlc, period), name='KC_MIDDLE')
        else:
            middle = pd.Series(MA, name='KC_MIDDLE')

        up = pd.Series(middle + (2 * cls.ATR(ohlc, atr_period)), name='KC_UPPER')
        down = pd.Series(middle - (2 * cls.ATR(ohlc, atr_period)), name='KC_LOWER')

        return pd.concat([up, down], axis=1)

    @classmethod
    def DO(cls, ohlc, period=20):
        """Donchian Channel, a moving average indicator developed by Richard Donchian.
        It plots the highest high and lowest low over the last period time intervals."""

        upper = pd.Series(ohlc['high'].tail(period).max(), name='UPPER')
        lower = pd.Series(ohlc['low'].tail().min(), name='LOWER')
        middle = pd.Series((upper + lower) / 2, name='MIDDLE')

        return pd.concat([lower, middle, upper], axis=1)

    @classmethod
    def DMI(cls, ohlc, period=14):
        """The directional movement indicator (also known as the directional movement index - DMI) is a valuable tool
         for assessing price direction and strength. This indicator was created in 1978 by J. Welles Wilder, who also created the popular
         relative strength index. DMI tells you when to be long or short.
         It is especially useful for trend trading strategies because it differentiates between strong and weak trends,
         allowing the trader to enter only the strongest trends.
        """

        ohlc['up_move'] = ohlc['high'].diff()
        ohlc['down_move'] = ohlc['low'].diff()

        DMp = []
        DMm = []

        for row in ohlc.itertuples():
            if row.up_move > row.down_move and row.up_move > 0:
                DMp.append(row.up_move)
            else:
                DMp.append(0)

            if row.down_move > row.up_move and row.down_move > 0:
                DMm.append(row.down_move)
            else:
                DMm.append(0)

        ohlc['DMp'] = DMp
        ohlc['DMm'] = DMm

        diplus = pd.Series(
                100 * (ohlc['DMp'] / cls.ATR(ohlc, period * 6)).ewm(span=period, min_periods=period - 1).mean(),
                name='DI+')
        diminus = pd.Series(
                100 * (ohlc['DMm'] / cls.ATR(ohlc, period * 6)).ewm(span=period, min_periods=period - 1).mean(),
                name='DI-')

        return pd.concat([diplus, diminus], axis=1)


    @classmethod
    def ADX(cls, ohlc, period=14):
        """The A.D.X. is 100 * smoothed moving average of absolute value (DMI +/-) divided by (DMI+ + DMI-). ADX does not indicate trend direction or momentum,
        only trend strength. Generally, A.D.X. readings below 20 indicate trend weakness,
        and readings above 40 indicate trend strength. An extremely strong trend is indicated by readings above 50"""

        dmi = cls.DMI(ohlc, period)
        return pd.Series(100 * (abs(dmi['DI+'] - dmi['DI-']) /
                                (dmi['DI+'] + dmi['DI-'])).ewm(alpha=1 / period).mean(),
                                name='{0} period ADX.'.format(period))


    @classmethod
    def PIVOTS(cls, ohlc):
        """Pivots"""
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:pivot_points
        raise NotImplementedError


    @classmethod
    def STOCH(cls, ohlc, period=14):
        """Stochastic oscillator %K
         The stochastic oscillator is a momentum indicator comparing the closing price of a security
         to the range of its prices over a certain period of time.
         The sensitivity of the oscillator to market movements is reducible by adjusting that time
         period or by taking a moving average of the result.
        """

        highest_high = ohlc['high'].rolling(center=False, window=period).max()
        lowest_low = ohlc['low'].rolling(center=False, window=period).min()

        STOCH = pd.Series((highest_high - ohlc['close']) / (highest_high - lowest_low),
                          name='{0} period STOCH %K'.format(period))

        return 100 * STOCH


    @classmethod
    def STOCHD(cls, ohlc, period=3):
        """Stochastic oscillator %D
        STOCH%D is a 3 period simple moving average of %K.
        """

        return pd.Series(cls.STOCH(ohlc).rolling(center=False, window=period, min_periods=period - 1).mean(),
                         name='{0} perood STOCH %D.'.format(period))


    @classmethod
    def STOCHRSI(cls, ohlc, rsi_period=14, stoch_period=14):
        """StochRSI is an oscillator that measures the level of RSI relative to its high-low range over a set time period.
        StochRSI applies the Stochastics formula to RSI values, instead of price values. This makes it an indicator of an indicator.
        The result is an oscillator that fluctuates between 0 and 1."""

        rsi = cls.RSI(ohlc, rsi_period)
        return pd.Series(((rsi - rsi.min()) / (rsi.max() - rsi.min())).rolling(window=stoch_period).mean(),
                         name='{0} period stochastic RSI.'.format(rsi_period))


    @classmethod
    def WILLIAMS(cls, ohlc, period=14):
        """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
         of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
         Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
         of its recent trading range.
         The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
        """

        highest_high = ohlc['high'].rolling(center=False, window=14).max()
        lowest_low = ohlc['low'].rolling(center=False, window=14).min()

        WR = pd.Series((highest_high - ohlc['close']) / (highest_high - lowest_low),
                       name='{0} Williams %R'.format(period))

        return WR * -100


    @classmethod
    def UO(cls, ohlc):
        """Ultimate Oscillator is a momentum oscillator designed to capture momentum across three different time frames.
        The multiple time frame objective seeks to avoid the pitfalls of other oscillators.
        Many momentum oscillators surge at the beginning of a strong advance and then form bearish divergence as the advance continues.
        This is because they are stuck with one time frame. The Ultimate Oscillator attempts to correct this fault by incorporating longer
        time frames into the basic formula."""

        k = []  # current low or past close
        for row, _row in zip(ohlc.itertuples(), ohlc.shift(-1).itertuples()):
            k.append(min(row.low, _row.close))
        bp = pd.Series(ohlc['close'] - k, name='bp')  ## Buying pressure

        Average7 = bp.rolling(window=7).sum() / cls.TR(ohlc).rolling(window=7).sum()
        Average14 = bp.rolling(window=14).sum() / cls.TR(ohlc).rolling(window=14).sum()
        Average28 = bp.rolling(window=28).sum() / cls.TR(ohlc).rolling(window=28).sum()

        return pd.Series((100 * ((4 * Average7) + (2 * Average14) + Average28)) / (4 + 2 + 1))


    @classmethod
    def AO(cls, ohlc, slow_period=34, fast_period=5):
        """
        Awesome Oscillator is an indicator used to measure market momentum. AO calculates the difference of a 34 Period and 5 Period Simple Moving Averages.
        The Simple Moving Averages that are used are not calculated using closing price but rather each bar's midpoints.
        AO is generally used to affirm trends or to anticipate possible reversals. """

        slow = pd.Series(((ohlc['high'] + ohlc['low']) / 2).rolling(window=slow_period).mean(), name='slow_AO')
        fast = pd.Series(((ohlc['high'] + ohlc['low']) / 2).rolling(window=fast_period).mean(), name='fast_AO')

        return pd.Series(fast - slow, name='AO')


    @classmethod
    def MI(cls, ohlc, period=9):
        """Developed by Donald Dorsey, the Mass Index uses the high-low range to identify trend reversals based on range expansions.
        In this sense, the Mass Index is a volatility indicator that does not have a directional bias.
        Instead, the Mass Index identifies range bulges that can foreshadow a reversal of the current trend."""

        _range = pd.Series(ohlc['high'] - ohlc['low'], name='range')
        EMA9 = _range.ewm(span=period, ignore_na=False).mean()
        DEMA9 = EMA9.ewm(span=period, ignore_na=False).mean()
        mass = EMA9 / DEMA9

        return pd.Series(mass.rolling(window=25).sum(), name='Mass Index').tail(period)


    @classmethod
    def VORTEX(cls, ohlc, period=14):
        """The Vortex indicator plots two oscillating lines, one to identify positive trend movement and the other
         to identify negative price movement.
         Indicator construction revolves around the highs and lows of the last two days or periods.
         The distance from the current high to the prior low designates positive trend movement while the
         distance between the current low and the prior high designates negative trend movement.
         Strongly positive or negative trend movements will show a longer length between the two numbers while
         weaker positive or negative trend movement will show a shorter length."""

        VMP = pd.Series(ohlc['high'] - ohlc['low'].shift(-1).abs())
        VMM = pd.Series(ohlc['low'] - ohlc['high'].shift(-1).abs())

        VMPx = VMP.rolling(window=period).sum()
        VMMx = VMM.rolling(window=period).sum()

        VIp = pd.Series(VMPx / cls.TR(ohlc), name='VIp').interpolate(method='index')
        VIm = pd.Series(VMMx / cls.TR(ohlc), name='VIm').interpolate(method='index')

        return pd.concat([VIm, VIp], axis=1)


    @classmethod
    def KST(cls, ohlc, r1=10, r2=15, r3=20, r4=30):
        """Know Sure Thing (KST) is a momentum oscillator based on the smoothed rate-of-change for four different time frames.
        KST measures price momentum for four different price cycles. It can be used just like any momentum oscillator.
        Chartists can look for divergences, overbought/oversold readings, signal line crossovers and centerline crossovers."""

        r1 = cls.ROC(ohlc, r1).rolling(window=10).mean()
        r2 = cls.ROC(ohlc, r2).rolling(window=10).mean()
        r3 = cls.ROC(ohlc, r3).rolling(window=10).mean()
        r4 = cls.ROC(ohlc, r4).rolling(window=15).mean()

        k = pd.Series((r1 * 1) + (r2 * 2) + (r3 * 3) + (r4 * 4), name='KST')
        signal = pd.Series(k.rolling(window=10).mean(), name="signal")

        return pd.concat([k, signal], axis=1)


    @classmethod
    def TSI(cls, ohlc, long=25, short=13, signal=13):
        """True Strength Index (TSI) is a momentum oscillator based on a double smoothing of price changes."""

        ## Double smoother price change
        momentum = pd.Series(ohlc['close'].diff())  ## 1 period momentum
        _EMA25 = pd.Series(momentum.ewm(span=long, min_periods=long - 1).mean(), name='_price change EMA25')
        _DEMA13 = pd.Series(_EMA25.ewm(span=short, min_periods=short - 1).mean(),
                            name='_price change double smoothed DEMA13')

        ## Double smoothed absolute price change
        absmomentum = pd.Series(ohlc['close'].diff().abs())
        _aEMA25 = pd.Series(absmomentum.ewm(span=long, min_periods=long - 1).mean(), name='_abs_price_change EMA25')
        _aDEMA13 = pd.Series(_aEMA25.ewm(span=short, min_periods=short - 1).mean(),
                             name='_abs_price_change double smoothed DEMA13')

        TSI = pd.Series((_DEMA13 / _aDEMA13) * 100, name='TSI')
        signal = pd.Series(TSI.ewm(span=signal, min_periods=signal - 1).mean(), name='signal')

        return pd.concat([TSI, signal], axis=1)


    @classmethod
    def TP(cls, ohlc):
        """Typical Price refers to the arithmetic average of the high, low, and closing prices for a given period."""

        return pd.Series((ohlc['high'] + ohlc['low'] + ohlc['close']) / 3, name='TP')


    @classmethod
    def ADL(cls, ohlcv):
        """The accumulation/distribution line was created by Marc Chaikin to determine the flow of money into or out of a security.
        It should not be confused with the advance/decline line. While their initials might be the same, these are entirely different indicators,
        and their uses are different as well. Whereas the advance/decline line can provide insight into market movements,
        the accumulation/distribution line is of use to traders looking to measure buy/sell pressure on a security or confirm the strength of a trend."""

        MFM = pd.Series(
                (ohlcv['close'] - ohlcv['low']) - (ohlcv['high'] - ohlcv['close']) / (ohlcv['high'] - ohlcv['low']),
                name='MFM')  # Money flow multiplier
        MFV = pd.Series(MFM * ohlcv['volume'], name='MFV')
        return MFV.cumsum()


    @classmethod
    def CHAIKIN(cls, ohlcv):
        """Chaikin Oscillator, named after its creator, Marc Chaikin, the Chaikin oscillator is an oscillator that measures the accumulation/distribution
         line of the moving average convergence divergence (MACD). The Chaikin oscillator is calculated by subtracting a 10-day exponential moving average (EMA)
         of the accumulation/distribution line from a three-day EMA of the accumulation/distribution line, and highlights the momentum implied by the
         accumulation/distribution line."""

        return pd.Series(
                cls.ADL(ohlcv).ewm(span=3, min_periods=2).mean() - cls.ADL(ohlcv).ewm(span=10, min_periods=9).mean())


    @classmethod
    def MFI(cls, ohlc, period=14):
        """The money flow index (MFI) is a momentum indicator that measures
        the inflow and outflow of money into a security over a specific period of time.
        MFI can be understood as RSI adjusted for volume.
        The money flow indicator is one of the more reliable indicators of overbought and oversold conditions, perhaps partly because
        it uses the higher readings of 80 and 20 as compared to the RSI's overbought/oversold readings of 70 and 30"""

        tp = cls.TP(ohlc)
        rmf = pd.Series(tp * ohlc['volume'], name='rmf')  ## Real Money Flow
        _mf = pd.concat([tp, rmf], axis=1)
        _mf['delta'] = _mf['TP'].diff()


        def pos(row):
            if row['delta'] > 0:
                return row['rmf']
            else:
                return 0


        def neg(row):
            if row['delta'] < 0:
                return row['rmf']
            else:
                return 0


        _mf['neg'] = _mf.apply(neg, axis=1)
        _mf['pos'] = _mf.apply(pos, axis=1)

        mfratio = pd.Series(_mf['pos'].rolling(window=period, min_periods=period - 1).sum() /
                            _mf['neg'].rolling(window=period, min_periods=period - 1).sum())

        return pd.Series(100 - (100 / (1 + mfratio)), name='{0} period MFI'.format(period))


    @classmethod
    def OBV(cls, ohlcv):
        """
        On Balance Volume (OBV) measures buying and selling pressure as a cumulative indicator that adds volume on up days and subtracts volume on down days.
        OBV was developed by Joe Granville and introduced in his 1963 book, Granville's New Key to Stock Market Profits.
        It was one of the first indicators to measure positive and negative volume flow.
        Chartists can look for divergences between OBV and price to predict price movements or use OBV to confirm price trends."""

        obv = [0]

        for row, _row in zip(ohlcv.itertuples(), ohlcv.shift(-1).itertuples()):
            if row.close > _row.close:
                obv.append(obv[-1] + row.volume)
            if row.close < _row.close:
                obv.append(obv[-1] - row.volume)
            if row.close == _row.close:
                obv.append(obv[-1])

        ohlcv['OBV'] = obv
        return pd.Series(ohlcv['OBV'], name='On Volume Balance')


    @classmethod
    def WOBV(cls, ohlcv):
        """

        Weighted OBV
        Can also be seen as an OBV indicator that takes the price differences into account.
        In a regular OBV, a high volume bar can make a huge difference,
        even if the price went up only 0.01, and it it goes down 0.01
        instead, that huge volume makes the OBV go down, even though
        hardly anything really happened.
        """

        wobv = pd.Series(ohlcv['volume'] * ohlcv['close'].diff(), name='WOBV')
        return wobv.cumsum()


    @classmethod
    def VZO(cls, ohlc, period=14):
        """VZO uses price, previous price and moving averages to compute its oscillating value.
        It is a leading indicator that calculates buy and sell signals based on oversold / overbought conditions.
        Oscillations between the 5% and 40% levels mark a bullish trend zone, while oscillations between -40% and 5% mark a bearish trend zone.
        Meanwhile, readings above 40% signal an overbought condition, while readings above 60% signal an extremely overbought condition.
        Alternatively, readings below -40% indicate an oversold condition, which becomes extremely oversold below -60%."""

        sign = lambda a: (a > 0) - (a < 0)
        r = ohlc['close'].diff().apply(sign) * ohlc['volume']
        dvma = r.ewm(span=period).mean()
        vma = ohlc['volume'].ewm(span=period).mean()

        return pd.Series(100 * (dvma / vma), name='VZO')


    @classmethod
    def EFI(cls, ohlcv, period=13):
        """Elder's Force Index is an indicator that uses price and volume to assess the power
         behind a move or identify possible turning points."""

        fi = pd.Series((ohlcv['close'] - ohlcv['close'].diff()) * ohlcv['volume'])
        return pd.Series(fi.ewm(ignore_na=False, min_periods=period - 1, span=period).mean(),
                         name='{0} period Force Index'.format(period))


    @classmethod
    def CFI(cls, ohlcv):
        """

        Cummulative Force Index.
        Adopted from  Elder's Force Index.
        """

        fi1 = pd.Series(ohlcv['volume'] * ohlcv['close'].diff())
        cfi = pd.Series(fi1.ewm(ignore_na=False, min_periods=9, span=10).mean(), name='CFI')

        return cfi.cumsum()


    @classmethod
    def EBBP(cls, ohlc):
        """Bull power and bear power by Dr. Alexander Elder show where today’s high and low lie relative to the a 13-day EMA"""

        bull_power = pd.Series(ohlc['high'] - cls.EMA(ohlc, 13), name="Bull.")
        bear_power = pd.Series(ohlc['low'] - cls.EMA(ohlc, 13), name="Bear.")

        return pd.concat([bull_power, bear_power], axis=1)


    @classmethod
    def EMV(cls, ohlcv, period=14):
        """Ease of Movement (EMV) is a volume-based oscillator that fluctuates above and below the zero line.
        As its name implies, it is designed to measure the 'ease' of price movement.
        prices are advancing with relative ease when the oscillator is in positive territory.
        Conversely, prices are declining with relative ease when the oscillator is in negative territory."""

        distance = pd.Series(((ohlcv['high'] + ohlcv['low']) / 2) - (ohlcv['high'].diff() - ohlcv['low'].diff()) / 2)
        box_ratio = pd.Series((ohlcv['volume'] / 1000000) / (ohlcv['high'] - ohlcv['low']))

        _emv = pd.Series(distance / box_ratio)

        return pd.Series(_emv.rolling(window=period).mean(), name='{0} period EMV.'.format(period))


    @classmethod
    def CCI(cls, ohlc, period=20):
        """Commodity Channel Index (CCI) is a versatile indicator that can be used to identify a new trend or warn of extreme conditions.
        CCI measures the current price level relative to an average price level over a given period of time.
        The CCI typically oscillates above and below a zero line. Normal oscillations will occur within the range of +100 and −100.
        Readings above +100 imply an overbought condition, while readings below −100 imply an oversold condition.
        As with other overbought/oversold indicators, this means that there is a large probability that the price will correct to more representative levels."""

        tp = cls.TP(ohlc)
        return pd.Series((tp - tp.rolling(window=period).mean()) / (0.015 * tp.mad()),
                         name='{0} period CCI'.format(period))


    @classmethod
    def COPP(cls, ohlc):
        """The Coppock Curve is a momentum indicator, it signals buying opportunities when the indicator moved from negative territory to positive territory."""

        roc1 = cls.ROC(ohlc, 14)
        roc2 = cls.ROC(ohlc, 11)

        return pd.Series((roc1 + roc2).ewm(span=10, min_periods=9).mean(), name='Coppock Curve')


    @classmethod
    def BASP(cls, ohlc, period=40):
        """BASP indicator serves to identify buying and selling pressure."""

        sp = ohlc['high'] - ohlc['close']
        bp = ohlc['close'] - ohlc['low']
        spavg = sp.ewm(span=period, min_periods=period - 1).mean()
        bpavg = bp.ewm(span=period, min_periods=period - 1).mean()

        nbp = bp / bpavg
        nsp = sp / spavg

        varg = ohlc['volume'].ewm(span=period, min_periods=period - 1).mean()
        nv = ohlc['volume'] / varg

        nbfraw = pd.Series(nbp * nv, name='Buy.')
        nsfraw = pd.Series(nsp * nv, name='Sell.')

        return pd.concat([nbfraw, nsfraw], axis=1)


    @classmethod
    def BASPN(cls, ohlc, period=40):
        """
        Normalized BASP indicator
        """

        sp = ohlc['high'] - ohlc['close']
        bp = ohlc['close'] - ohlc['low']
        spavg = sp.ewm(span=period, min_periods=period - 1).mean()
        bpavg = bp.ewm(span=period, min_periods=period - 1).mean()

        nbp = bp / bpavg
        nsp = sp / spavg

        varg = ohlc['volume'].ewm(span=period, min_periods=period - 1).mean()
        nv = ohlc['volume'] / varg

        nbf = pd.Series((nbp * nv).ewm(span=20).mean(), name='Buy.')
        nsf = pd.Series((nsp * nv).ewm(span=20).mean(), name='Sell.')

        return pd.concat([nbf, nsf], axis=1)


    @classmethod
    def CMO(cls, ohlc, period=9):
        """
        Chande Momentum Oscillator (CMO) - technical momentum indicator invented by the technical analyst Tushar Chande.
        It is created by calculating the difference between the sum of all recent gains and the sum of all recent losses and then
        dividing the result by the sum of all price movement over the period.
        This oscillator is similar to other momentum indicators such as the Relative Strength Index and the Stochastic Oscillator
        because it is range bounded (+100 and -100)."""

        mom = ohlc['close'].diff().abs()
        sma_mom = mom.rolling(window=period).mean()
        mom_len = ohlc['close'].diff(period)

        return pd.Series(100 * (mom_len / (sma_mom * period)))


    @classmethod
    def CHANDELIER(cls, ohlc, period_1=14, period_2=22, k=3):
        """
        Chandelier Exit sets a trailing stop-loss based on the Average True Range (ATR).

        The indicator is designed to keep traders in a trend and prevent an early exit as long as the trend extends.

        Typically, the Chandelier Exit will be above prices during a downtrend and below prices during an uptrend.
        """

        l = pd.Series(ohlc['close'].rolling(window=period_2).max() - cls.ATR(ohlc, 22) * k,
                      name='Long.')
        s = pd.Series(ohlc['close'].rolling(window=period_1).min() - cls.ATR(ohlc, 22) * k,
                      name='Short.')

        return pd.concat([s, l], axis=1)


    @classmethod
    def QSTICK(cls, ohlc, period=14):
        """
        QStick indicator shows the dominance of black (down) or white (up) candlesticks, which are red and green in Chart,
        as represented by the average open to close change for each of past N days."""

        _close = ohlc['close'].tail(period)
        _open = ohlc['open'].tail(period)

        return pd.Series((_close - _open) / period, name='{0} period QSTICK.'.format(period))


    @classmethod
    def TMF(cls, ohlcv, period=21):
        """Indicator by Colin Twiggs which improves upon CMF.
        source: https://user42.tuxfamily.org/chart/manual/Twiggs-Money-Flow.html"""

        ohlcv['ll'] = [ min(l, c) for l, c in zip(ohlcv['low'], ohlcv['close'].shift(-1)) ]
        ohlcv['hh'] = [ max(h, c) for h, c in zip(ohlcv['high'], ohlcv['close'].shift(-1)) ]

        ohlcv['range'] = (2 * ((ohlcv['close'] - ohlcv['ll']) / (ohlcv['hh'] - ohlcv['ll'])) - 1 )
        ohlcv['rangev'] = None

        # TMF Signal Line = EMA(TMF)
        # return TMF
        raise NotImplementedError


    @classmethod
    def WTO(cls, ohlc, channel_lenght=10, average_lenght=21):
        """
        Wave Trend Oscillator
        source: http://www.fxcoaching.com/WaveTrend/
        """

        ap = cls.TP(ohlc)
        esa = ap.ewm(span=channel_lenght).mean()
        d = pd.Series((ap - esa).abs().ewm(span=channel_lenght).mean(), name='d')
        ci = (ap - esa) / (0.015 * d)

        wt1 = pd.Series(ci.ewm(span=average_lenght).mean(), name="WT1.")
        wt2 = pd.Series(wt1.rolling(window=4).mean(), name="WT2.")

        return pd.concat([wt1, wt2], axis=1)


    @classmethod
    def FISH(cls, ohlc, period=10):
        """
        Fisher Transform was presented by John Ehlers. It assumes that price distributions behave like square waves.

        The Fisher Transform uses the mid-point or median price in a series of calculations to produce an oscillator.

        A signal line which is a previous value of itself is also calculated.
        """

        import numpy as np

        med = (ohlc['high'] + ohlc['low']) / 2
        ndaylow = med.rolling(window=period).min()
        ndayhigh = med.rolling(window=period).max()
        raw = (2 * ((med - ndaylow) / (ndayhigh - ndaylow))) - 1
        smooth = raw.ewm(span=5).mean()

        return pd.Series((np.log((1 + smooth) / (1 - smooth))).ewm(span=3).mean(),
                          name='{0} period FISH.'.format(period))


    @classmethod
    def ICHIMOKU(cls, ohlc):
        """
        The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a versatile indicator that defines support and resistance,
        identifies trend direction, gauges momentum and provides trading signals.

        Ichimoku Kinko Hyo translates into “one look equilibrium chart”.
        """

        tenkan_sen = pd.Series((ohlc['high'].rolling(window=9).mean() + ohlc['low'].rolling(window=9).mean()) / 2,
                               name='TENKAN')  ## conversion line
        kijun_sen = pd.Series((ohlc['high'].rolling(window=26).mean() + ohlc['low'].rolling(window=26).mean()) / 2,
                              name='KIJUN')  ## base line

        senkou_span_a = pd.Series(((tenkan_sen / kijun_sen) / 2), name='senkou_span_a')  ## Leading span
        senkou_span_b = pd.Series(
                ((ohlc['high'].rolling(window=52).mean() + ohlc['low'].rolling(window=52).mean()) / 2),
                name='SENKOU')
        chikou_span = pd.Series(ohlc['close'].shift(-26).rolling(window=26).mean(), name='CHIKOU')

        return pd.concat([tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span], axis=1)


    @classmethod
    def APZ(cls, ohlc, period=21, dev_factor=2, MA=None):
        """
        The adaptive price zone (APZ) is a technical indicator developed by Lee Leibfarth.

        The APZ is a volatility based indicator that appears as a set of bands placed over a price chart.

        Especially useful in non-trending, choppy markets,

        the APZ was created to help traders find potential turning points in the markets.
        """

        if not isinstance(MA, pd.Series):
            MA = cls.DEMA(ohlc, period)
        price_range = pd.Series((ohlc['high'] - ohlc['low']).ewm(span=period, min_periods=period - 1).mean())
        volatility_value = pd.Series(price_range.ewm(span=period, min_periods=period - 1).mean(), name='vol_val')

        # upper_band = dev_factor * volatility_value + dema
        upper_band = pd.Series((volatility_value * dev_factor) + MA, name='UPPER')
        lower_band = pd.Series(MA - (volatility_value * dev_factor), name='LOWER')

        return pd.concat([upper_band, lower_band], axis=1)


    @classmethod
    def VR(cls, ohlc, period=14):
        """
        Vector Size Indicator
        :param pd.DataFrame ohlc: 'open, high, low, close' pandas DataFrame
        :return pd.Series: indicator calcs as pandas Series
        """

        import numpy as np

        vector_size = len(ohlc.close)
        high_low_diff = ohlc.high - ohlc.low
        high_close_diff = np.zeros(vector_size)
        high_close_diff[1:] = np.abs(ohlc.high[1:] - ohlc.close[0:vector_size - 1])
        low_close_diff = np.zeros(vector_size)
        low_close_diff[1:] = np.abs(ohlc.low[1:] - ohlc.close[0:vector_size - 1])
        vectors_stacked = np.vstack((high_low_diff, high_close_diff, low_close_diff))

        tr = np.amax(vectors_stacked, axis=0)
        vr = pd.Series(tr / cls.EMA(ohlc.close, period=periods), name="{0} period VR.".format(period))

        return vr


    @classmethod
    def SQZMI(cls, ohlc, period=20, MA=None):
        '''
        Squeeze Momentum Indicator

        The Squeeze indicator attempts to identify periods of consolidation in a market.
        In general the market is either in a period of quiet consolidation or vertical price discovery.
        By identifying these calm periods, we have a better opportunity of getting into trades with the potential for larger moves.
        Once a market enters into a “squeeze”, we watch the overall market momentum to help forecast the market direction and await a release of market energy.

        :param pd.DataFrame ohlc: 'open, high, low, close' pandas DataFrame
        :period: int - number of periods to take into consideration
        :MA pd.Series: override internal calculation which uses SMA with moving average of your choice
        :return pd.Series: indicator calcs as pandas Series

        SQZMI['SQZ'] is bool True/False, if True squeeze is on. If false, squeeeze has fired.
        SQZMI['DIR'] is bool True/False, if True momentum is up. If false, momentum is down.
        '''

        if not isinstance(MA, pd.core.series.Series):
            ma = pd.Series(cls.SMA(ohlc, period))
        else:
            ma = None

        bb = cls.BBANDS(ohlc, period=period, MA=ma)
        kc = cls.KC(ohlc, period=period)
        comb = pd.concat([bb, kc], axis=1)

        def sqz_on(row):
            if row['BB_LOWER'] > row['KC_LOWER'] and row['BB_UPPER'] < row['KC_UPPER']:
                return True
            else:
                return False

        comb['SQZ'] = comb.apply(sqz_on, axis=1)

        return pd.Series(comb['SQZ'], name='{0} period SQZMI'.format(period))


    @classmethod
    def VPT(cls, ohlc):
        '''
        Volume Price Trend
        The Volume Price Trend uses the difference of price and previous price with volume and feedback to arrive at its final form.
        If there appears to be a bullish divergence of price and the VPT (upward slope of the VPT and downward slope of the price) a buy opportunity exists.
        Conversely, a bearish divergence (downward slope of the VPT and upward slope of the price) implies a sell opportunity. 
        '''

        hilow = ((ohlc['high'] - ohlc['low']) * 100)
        openclose = ((ohlc['close'] - ohlc['open']) *100)
        vol = (ohlc['volume'] / hilow)
        spreadvol = (openclose * vol).cumsum()

        vpt = spreadvol + spreadvol

        return pd.Series(vpt, name="VPT")


    @classmethod
    def FVE(cls, ohlc, period=22, factor=0.3):
        '''
        FVE is a money flow indicator, but it has two important innovations: first, the F VE takes into account both intra and
        interday price action, and second, minimal price changes are taken into account by introducing a price threshold.
        '''

        hl2 = (ohlc['high'] + ohlc['low']) / 2
        tp = TA.TP(ohlc)
        smav = ohlc['volume'].rolling(window=period, min_periods=period-1).mean()
        mf = pd.Series((ohlc['close'] - hl2 + tp.diff()), name='mf')

        _mf = pd.concat([ohlc['close'], ohlc['volume'], mf], axis=1)

        def vol_shift(row):

            if row['mf'] > factor * row['close'] / 100:
                return row['volume']
            elif row['mf'] < -factor * row['close'] / 100:
                return -row['volume']
            else:
                return 0

        _mf['vol_shift'] = _mf.apply(vol_shift, axis=1)
        _sum = _mf['vol_shift'].rolling(window=period, min_periods=period-1).sum()

        return pd.Series((_sum / smav) / period * 100)


if __name__ == '__main__':
    print([k for k in TA.__dict__.keys() if k[0] not in '_'])
