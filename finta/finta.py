import pandas as pd

class TA:

    @classmethod
    def SMA(cls, ohlc, period=41, column="close"):
        '''Simple moving average - rolling mean in pandas lingo. Also known as "MA".
        The simple moving average (SMA) is the most basic of the moving averages used for trading. 
        The simple moving average formula is calculated by taking the average closing price of a stock over the last n periods.'''

        return pd.Series(ohlc[column].rolling(center=False, window=period, min_periods=period-1).mean(), name="SMA")
        
    @classmethod
    def SMM(cls, ohlc, period=9, column="close"):
        '''Simple moving median, an alternative to moving average. SMA, when used to estimate the underlying trend in a time series, 
        is susceptible to rare events such as rapid shocks or other anomalies. 
        A more robust estimate of the trend is the simple moving median over n time periods'''

        return pd.Series(ohlc[column].rolling(center=False, window=period, min_periods=period-1).median(), name="SMM")

    @classmethod
    def EMA(cls, ohlc, period=9, column="close"): ## EWMA
        '''Exponential Weighted Moving Average - Like all moving average indicators, they are much better suited for trending markets. 
        When the market is in a strong and sustained uptrend, the EMA indicator line will also show an uptrend and vice-versa for a down trend.
        EMAs are commonly used in conjunction with other indicators to confirm significant market moves and to gauge their validity.'''

        return pd.Series(ohlc[column].ewm(ignore_na=False, min_periods=period-1, span=period).mean(), name="EMA")
    
    @classmethod
    def DEMA(cls, ohlc, period=9, column="close"):
        '''Double Exponential Moving Average - attempts to remove the inherent lag associated to Moving Averages
         by placing more weight on recent values. The name suggests this is achieved by applying a double exponential
        smoothing which is not the case. The name double comes from the fact that the value of an EMA (Exponential Moving Average) is doubled.
        To keep it in line with the actual data and to remove the lag the value "EMA of EMA" is subtracted from the previously doubled EMA.
        Because EMA(EMA) is used in the calculation, DEMA needs 2 * period -1 samples to start producing values in contrast to the period 
        samples needed by a regular EMA'''
        
        DEMA = 2 * cls.EMA(ohlc, period) - cls.EMA(ohlc, period).ewm(ignore_na=False, min_periods=period-1, span=period).mean()

        return pd.Series(DEMA, name="DEMA")
    
    @classmethod
    def TEMA(cls, ohlc, period=9):
        '''Triple exponential moving average - attempts to remove the inherent lag associated to Moving Averages by placing more weight on recent values. 
        The name suggests this is achieved by applying a triple exponential smoothing which is not the case. The name triple comes from the fact that the 
        value of an EMA (Exponential Moving Average) is triple. 
        To keep it in line with the actual data and to remove the lag the value "EMA of EMA" is subtracted 3 times from the previously tripled EMA. 
        Finally "EMA of EMA of EMA" is added.
        Because EMA(EMA(EMA)) is used in the calculation, TEMA needs 3 * period - 2 samples to start producing values in contrast to the period samples
        needed by a regular EMA.'''

        triple_ema = 3 * cls.EMA(ohlc, period)
        ema_ema_ema = cls.EMA(ohlc, period).ewm(ignore_na=False, span=period).mean().ewm(ignore_na=False, span=period).mean()
        TEMA = triple_ema - 3 * cls.EMA(ohlc, period).ewm(ignore_na=False, min_periods=period-1, span=period).mean() + ema_ema_ema

        return pd.Series(TEMA, name="TEMA")
    
    @classmethod
    def TRIMA(cls, ohlc, period=18): ## TMA
        '''The Triangular Moving Average (TRIMA) [also known as TMA] represents an average of prices, but places weight on the middle prices of the time period. 
        The calculations double-smooth the data using a window width that is one-half the length of the series.
        The TRIMA is simply the SMA of the SMA'''
        
        SMA = cls.SMA(ohlc, period).rolling(center=False, window=period, min_periods=period-1).mean()
        return pd.Series(SMA, name="TRIMA")
    
    @classmethod
    def TRIX(cls, ohlc, period=15):
        '''The Triple Exponential Moving Average Oscillator (TRIX) by Jack Hutson is a momentum indicator that oscillates around zero. 
        It displays the percentage rate of change between two triple smoothed exponential moving averages.
        To calculate TRIX we calculate triple smoothed EMA3 of n periods and then substract previous period EMA3 value 
        from last EMA3 value and divide the result with yesterdays EMA3 value.'''

        EMA1 = cls.EMA(ohlc, period)
        EMA2 = EMA1.ewm(span=period).mean()
        EMA3 = EMA2.ewm(span=period).mean()
        TRIX = (EMA3 - EMA3.diff()) / EMA3.diff()
        
        return pd.Series(TRIX, name="TRIX")

    @classmethod
    def AMA(cls, ohlc, period=None, column="close"):
        '''This indicator is either quick, or slow, to signal a market entry depending on the efficiency of the move in the market.'''
        raise NotImplementedError
    
    @classmethod
    def LWMA(cls, ohlc, period=None, column="close"):
        '''Linear Weighter Moving Average'''
        raise NotImplementedError
    
    @classmethod
    def VAMA(cls, ohlcv, period=8, column="close"):
        '''Volume Adjusted Moving Average'''

        vp = ohlcv["volume"] * ohlcv["close"]
        volsum = ohlcv["volume"].rolling(window=period).mean()
        volRatio = pd.Series(vp / volsum, name="VAMA")
        cumSum = (volRatio * ohlcv[column]).rolling(window=period).sum()
        cumDiv = volRatio.rolling(window=period).sum()
        
        return pd.Series(cumSum/cumDiv, name="VAMA")

    @classmethod
    def VIDYA(cls, ohlcv, period=9, smoothing_period=12):
        """ Vidya (variable index dynamic average) indicator is a modification of the traditional Exponential Moving Average (EMA) indicator. 
        The main difference between EMA and Vidya is in the way the smoothing factor F is calculated. 
        In EMA the smoothing factor is a constant value F=2/(period+1); 
        in Vidya the smoothing factor is variable and depends on bar-to-bar price movements."""
        
        raise NotImplemetedError  

    @classmethod
    def KAMA(cls, ohlc, er=10, ema_fast=2, ema_slow=30, period=20):
        """Developed by Perry Kaufman, Kaufman's Adaptive Moving Average (KAMA) is a moving average designed to account for market noise or volatility. 
        KAMA will closely follow prices when the price swings are relatively small and the noise is low. 
        KAMA will adjust when the price swings widen and follow prices from a greater distance."""

        er = cls.ER(ohlc, er)
        sc = (er * ((2/ema_fast+1) - (2/ema_slow+1)) + (2/ema_slow+1)) **2 ## smoothing constant

        # Current KAMA = Prior KAMA + SC x (Price - Prior KAMA)

        raise NotImplementedError
    
    @classmethod
    def ZLEMA(cls, ohlc, period=26):
        """ZLEMA is an abbreviation of Zero Lag Exponential Moving Average. It was developed by John Ehlers and Rick Way. 
        ZLEMA is a kind of Exponential moving average but its main idea is to eliminate the lag arising from the very nature of the moving averages 
        and other trend following indicators. As it follows price closer, it also provides better price averaging and responds better to price swings."""

        lag = (period-1)/2
        return pd.Series((ohlc["close"] + (ohlc["close"].diff(lag))), name="ZLEMA")

    @classmethod
    def WMA(cls, ohlc, period=42, column="close"):
        """WMA stands for weighted moving average. It helps to smooth the price curve for better trend identification. 
        It places even greater importance on recent data than the EMA does."""

        d = (period * (period + 1)) / 2

        raise NotImplementedError

    @classmethod
    def HMA(cls, ohlc, period=16):
        """HMA indicator is a common abbreviation of Hull Moving Average. The average was developed by Allan Hull and is used mainly to identify the 
        current market trend. Unlike SMA (simple moving average) the curve of Hull moving average is considerably smoother. 
        Moreover, because its aim is to minimize the lag between HMA and price it does follow the price activity much closer. 
        It is used especially for middle-term and long-term trading."""

        """
        Calculate WMA (weighted moving average) for half of the period (8-day WMA in this case) and multiple the result by 2.
        Calculate WMA of the full period (16-day WMA) and subtract if from the first result (2 * WMA8).
        Calculate the square root of the full time period, i. e. √16 = 4.
        Calculate 4-day WMA from the result you got in step 2.
        """

        """
        wmaA     = closes.apply(talib.MA,   timeperiod = HMAPeriodsb / 2, matype = MAType.WMA).dropna() * 2.0  
        wmaB     = closes.apply(talib.MA,   timeperiod = HMAPeriodsb, matype = MAType.WMA).dropna()  
        wmaDiffs = wmaA - wmaB  
        hma      = wmaDiffs.apply(talib.MA, timeperiod = math.sqrt(HMAPeriodsb), matype = MAType.WMA)  """

    @classmethod
    def VWAP(cls, ohlcv):
        '''The volume weighted average price (VWAP) is a trading benchmark used especially in pension plans. 
        VWAP is calculated by adding up the dollars traded for every transaction (price multiplied by number of shares traded) and then dividing 
        by the total shares traded for the day.'''
        
        return pd.Series(((ohlcv["volume"] * cls.TP(ohlcv)).cumsum()) / ohlcv["volume"].cumsum(), name="VWAP")
    
    @classmethod
    def SMMA(cls, ohlc, period=42, column="close"):
        """The SMMA gives recent prices an equal weighting to historic prices."""
        
        return pd.Series(ohlc[column].ewm(alpha=1/period).mean(), name="SMMA")
    
    @classmethod
    def ER(cls, ohlc, period=10):
        """The Kaufman Efficiency indicator is an oscillator indicator that oscillates between +100 and -100, where zero is the center point.
         +100 is upward forex trending market and -100 is downwards trending markets."""
        
        change = ohlc["close"].diff(period).abs()
        volatility = ohlc["close"].diff().abs().rolling(window=period).sum()

        return pd.Series(change/volatility, name="ER")

    @classmethod
    def MACD(cls, ohlc, period_fast=12, period_slow=26, signal=9):
        '''MACD, MACD Signal and MACD difference.
        The MACD Line oscillates above and below the zero line, which is also known as the centerline.
        These crossovers signal that the 12-day EMA has crossed the 26-day EMA. The direction, of course, depends on the direction of the moving average cross. 
        Positive MACD indicates that the 12-day EMA is above the 26-day EMA. Positive values increase as the shorter EMA diverges further from the longer EMA. 
        This means upside momentum is increasing. Negative MACD values indicates that the 12-day EMA is below the 26-day EMA. 
        Negative values increase as the shorter EMA diverges further below the longer EMA. This means downside momentum is increasing.
        
        Signal line crossovers are the most common MACD signals. The signal line is a 9-day EMA of the MACD Line.  
        As a moving average of the indicator, it trails the MACD and makes it easier to spot MACD turns.  
        A bullish crossover occurs when the MACD turns up and crosses above the signal line.  
        A bearish crossover occurs when the MACD turns down and crosses below the signal line.
        '''
        
        EMA_fast = pd.Series(ohlc["close"].ewm(ignore_na=False, min_periods=period_slow-1, span=period_fast).mean(), name="EMA_fast")
        EMA_slow = pd.Series(ohlc["close"].ewm(ignore_na=False, min_periods=period_slow-1, span=period_slow).mean(), name="EMA_slow")
        MACD = pd.Series(EMA_fast - EMA_slow, name="macd")
        MACD_signal = pd.Series(MACD.ewm(ignore_na=False, span=signal).mean(), name="macd_signal")
        
        return pd.concat([MACD, MACD_signal], axis=1)
    
    @classmethod
    def MOM(cls, ohlc, period=10):
        """Market momentum is measured by continually taking price differences for a fixed time interval. 
        To construct a 10-day momentum line, simply subtract the closing price 10 days ago from the last closing price. 
        This positive or negative value is then plotted around a zero line."""

        return ohlc["close"].diff(period)

    @classmethod
    def ROC(cls, ohlc, period=12):
        """The Rate-of-Change (ROC) indicator, which is also referred to as simply Momentum, 
        is a pure momentum oscillator that measures the percent change in price from one period to the next. 
        The ROC calculation compares the current price with the price “n” periods ago."""

        return pd.Series((ohlc["close"].diff(period) / ohlc["close"][-period]) * 100, name="ROC")

    @classmethod
    def RSI(cls, ohlc, period=14):
        """Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. 
        RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30. 
        Signals can also be generated by looking for divergences, failure swings and centerline crossovers. RSI can also be used to identify the general trend."""

        gain = [0]
        loss = [0]
        for row, _row in zip(ohlc.itertuples(), ohlc.shift(-1).itertuples()):
            if row.close - _row.close > 0:
                gain.append(row.close - _row.close)
                loss.append(0)
            if row.close - _row.close < 0:
                gain.append(0)
                loss.append(abs(row.close - _row.close))
            if row.close - _row.close == 0:
                gain.append(0)
                loss.append(0)
        
        ohlc["gain"] = gain
        ohlc["loss"] = loss

        avg_gain = ohlc["gain"].rolling(window=period).mean()
        avg_loss = ohlc["loss"].rolling(window=period).mean()

        RS = avg_gain / avg_loss
        return pd.Series(100 - (100 / (1 + RS)), name="RSI")
    
    @classmethod
    def TR(cls, ohlc, period=14):
        """True Range is the maximum of three price ranges.    
        Most recent period's high minus the most recent period's low.
        Absolute value of the most recent period's high minus the previous close.
        Absolute value of the most recent period's low minus the previous close."""
        
        TR1 = pd.Series(ohlc["high"].tail(period) - ohlc["low"].tail(period), name="high_low")
        TR2 = pd.Series(ohlc["high"].tail(period) - ohlc["close"].shift(-1).abs().tail(period), name="high_previous_close")
        TR3 = pd.Series(ohlc["close"].shift(-1).tail(period) - ohlc["low"].abs().tail(period), name="previous_close_low")
        TR = pd.concat([TR1, TR2, TR3], axis=1)
        
        l = []
        for row in TR.itertuples():
            l.append(max(row.high_low, row.high_previous_close, row.previous_close_low))
        
        TR["TA"] = l
        return pd.Series(TR["TA"], name="True Range")
    
    @classmethod
    def ATR(cls, ohlc, period=14):
        """Average True Range is moving average of True Range."""
        
        TR = cls.TR(ohlc, period*2)
        return pd.Series(TR.rolling(center=False, window=period, min_periods=period-1).mean(), name="ATR").tail(period)

    @classmethod
    def SAR(cls, ohlc):
        """SAR stands for “stop and reverse,” which is the actual indicator used in the system. 
        SAR trails price as the trend extends over time. The indicator is below prices when prices are rising and above prices when prices are falling.
        In this regard, the indicator stops and reverses when the price trend reverses and breaks above or below the indicator."""
        raise NotImplementedError

    @classmethod
    def BBANDS(cls, ohlc, period=20, column='close'):
        """Developed by John Bollinger, Bollinger Bands® are volatility bands placed above and below a moving average.
         Volatility is based on the standard deviation, which changes as volatility increases and decreases. 
         The bands automatically widen when volatility increases and narrow when volatility decreases. 
         This dynamic nature of Bollinger Bands also means they can be used on different securities with the standard settings. 
         For signals, Bollinger Bands can be used to identify M-Tops and W-Bottoms or to determine the strength of the trend.
         
         %b (pronounced "percent b") is derived from the formula for Stochastics and shows where price is in relation to the bands. 
         %b equals 1 at the upper band and 0 at the lower band. Writing upperBB for the upper Bollinger Band, 
         lowerBB for the lower Bollinger Band, and last for the last (price) valu%b (pronounced "percent b") is derived from the 
         formula for Stochastics and shows where price is in relation to the bands. %b equals 1 at the upper band and 0 at the 
         lower band. Writing upperBB for the upper Bollinger Band, lowerBB for the lower Bollinger Band, and last for the last (price) value
         
         Bandwidth tells how wide the Bollinger Bands are on a normalized basis. 
         Writing the same symbols as before, and middleBB for the moving average, or middle Bollinger Band:
         """ 
        
        std = ohlc["close"].tail(period).std()
        SMA = pd.Series(cls.SMA(ohlc, period), name="middle_bband")
        upper_bb = pd.Series(SMA + (2 * std), name="upper_bband")
        lower_bb = pd.Series(SMA - (2 * std), name="lower_bband")
        
        percent_b = pd.Series((ohlc["close"] - lower_bb) / (upper_bb - lower_bb), name="%b")
        b_bandwith = pd.Series((upper_bb - lower_bb) / SMA, name="b_bandwith")
        
        return pd.concat([upper_bb, SMA, lower_bb, b_bandwith, percent_b], axis=1)

    @classmethod
    def KC(cls, ohlc, period=20):
        """Keltner Channels [KC] are volatility-based envelopes set above and below an exponential moving average. 
        This indicator is similar to Bollinger Bands, which use the standard deviation to set the bands. 
        Instead of using the standard deviation, Keltner Channels use the Average True Range (ATR) to set channel distance. 
        The channels are typically set two Average True Range values above and below the 20-day EMA. 
        The exponential moving average dictates direction and the Average True Range sets channel width. 
        Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and channel direction. 
        Channels can also be used to identify overbought and oversold levels when the trend is flat."""

        middle = pd.Series(cls.SMA(ohlc, 20), name="middle_kchannel")
        up = pd.Series(middle + (2 * cls.ATR(ohlc, 10)), name="upper_kchannel")
        down = pd.Series(middle - (2 * cls.ATR(ohlc, 10)), name="lower_kchannel")

        return pd.concat([up, middle, down], axis=1)

    @classmethod
    def DO(cls, ohlc, period=20):
        """Donchian Channel, a moving average indicator developed by Richard Donchian. It plots the highest high and lowest low over the last period time intervals."""

        upper = pd.Series(ohlc["high"].max(), name="upper_dchannel")
        lower = pd.Series(ohlc["low"].min(), name="lower_dchannel")
        middle = pd.Series((upper / lower) / 2, name="middle_dchannel")

        return pd.concat([lower, middle, upper], axis=1)

    @classmethod
    def DMI(cls, ohlc, period=14):
        """The directional movement indicator (also known as the directional movement index - DMI) is a valuable tool
         for assessing price direction and strength. This indicator was created in 1978 by J. Welles Wilder, who also created the popular
         relative strength index. DMI tells you when to be long or short. 
         It is especially useful for trend trading strategies because it differentiates between strong and weak trends, 
         allowing the trader to enter only the strongest trends.
        """
        
        ohlc["up_move"] = ohlc["high"].diff()
        ohlc["down_move"] = ohlc["low"].diff()

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
        
        ohlc["DMp"] = DMp
        ohlc["DMm"] = DMm
        
        diplus = pd.Series(100 * (ohlc["DMp"] / cls.ATR(ohlc, period * 6)).ewm(span=period, min_periods=period - 1).mean(), name="positive_DMI")
        diminus = pd.Series(100 * (ohlc["DMm"] / cls.ATR(ohlc, period * 6)).ewm(span=period, min_periods=period - 1).mean(), name="negative_DMI")
        
        return pd.concat([diplus, diminus], axis=1)
    
    @classmethod
    def ADX(cls, ohlc, period=14):
        """The A.D.X. is 100 * smoothed moving average of absolute value (DMI +/-) divided by (DMI+ + DMI-). ADX does not indicate trend direction or momentum, 
        only trend strength. Generally, A.D.X. readings below 20 indicate trend weakness, 
        and readings above 40 indicate trend strength. An extremely strong trend is indicated by readings above 50"""

        dmi = cls.DMI(ohlc, period)
        return pd.Series(100 * (abs(dmi["positive_DMI"] - dmi["negative_DMI"]) / 
                            (dmi["positive_DMI"] + dmi["negative_DMI"])).ewm(alpha=1/period).mean(), name="ADX")

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
        
        highest_high = ohlc["high"].rolling(center=False, window=14).max()
        lowest_low = ohlc["low"].rolling(center=False, window=14).min()

        STOCH = pd.Series( (highest_high - ohlc["close"]) / (highest_high - lowest_low), name='{0} period STOCH %K'.format(period))

        return 100 * STOCH

    @classmethod
    def STOCHD(cls, ohlc, period=3):
        """Stochastic oscillator %D
        STOCH%D is a 3 period simple moving average of %K.
        """
        
        return pd.Series(cls.STOCHK(ohlc).rolling(center=False, window=period, min_periods=period-1).mean(), name="STOCH %D")

    @classmethod
    def STOCHRSI(cls, ohlc, rsi_period=14, stoch_period=14):
        """StochRSI is an oscillator that measures the level of RSI relative to its high-low range over a set time period. 
        StochRSI applies the Stochastics formula to RSI values, instead of price values. This makes it an indicator of an indicator. 
        The result is an oscillator that fluctuates between 0 and 1."""
        
        rsi = cls.RSI(ohlc, rsi_period)
        return pd.Series(((rsi - rsi.min()) / (rsi.max() - rsi.min())).rolling(window=stoch_period).mean(),
                                                                         name="Stochastic RSI")
        
    @classmethod
    def WILLIAMS(cls, ohlc, period=14):
        """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
         of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams. 
         Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between, 
         of its recent trading range.
         The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
        """

        highest_high = ohlc["high"].rolling(center=False, window=14).max()
        lowest_low = ohlc["low"].rolling(center=False, window=14).min()

        WR = pd.Series( (highest_high - ohlc["close"]) / (highest_high - lowest_low), name="{0} Williams %R".format(period))
        
        return WR * -100
    
    @classmethod
    def UO(cls, ohlc):
        """Ultimate Oscillator is a momentum oscillator designed to capture momentum across three different time frames.
        The multiple time frame objective seeks to avoid the pitfalls of other oscillators. 
        Many momentum oscillators surge at the beginning of a strong advance and then form bearish divergence as the advance continues. 
        This is because they are stuck with one time frame. The Ultimate Oscillator attempts to correct this fault by incorporating longer 
        time frames into the basic formula."""

        k = [] # current low or past close
        for row, _row in zip(ohlc.itertuples(), ohlc.shift(-1).itertuples()):
            k.append(min(row.low, _row.close))
        bp = pd.Series(ohlc["close"] - k, name="bp") ## Buying pressure

        Average7 = bp.rolling(window=7).sum() / cls.TR(ohlc, 7).sum()
        Average14 = bp.rolling(window=14).sum() / cls.TR(ohlc, 14).sum()
        Average28 = bp.rolling(window=28).sum() / cls.TR(ohlc, 28).sum()

        return pd.Series((100 * ((4 * Average7) + (2 * Average14) + Average28)) / (4+2+1))
    
    @classmethod
    def AO(cls, ohlc, slow_period=34, fast_period=5):
        '''Awesome Oscillator is an indicator used to measure market momentum. AO calculates the difference of a 34 Period and 5 Period Simple Moving Averages.
        The Simple Moving Averages that are used are not calculated using closing price but rather each bar's midpoints. 
        AO is generally used to affirm trends or to anticipate possible reversals. '''
        
        slow = pd.Series( ((ohlc["high"] + ohlc["low"]) / 2).rolling(window=slow_period).mean(), name="slow_AO")
        fast = pd.Series( ((ohlc["high"] + ohlc["low"]) / 2).rolling(window=fast_period).mean(), name="fast_AO")

        return pd.Series(fast - slow, name="AO")

    @classmethod
    def MI(cls, ohlc, period=9):
        """Developed by Donald Dorsey, the Mass Index uses the high-low range to identify trend reversals based on range expansions. 
        In this sense, the Mass Index is a volatility indicator that does not have a directional bias. 
        Instead, the Mass Index identifies range bulges that can foreshadow a reversal of the current trend."""
        
        _range = pd.Series(ohlc['high'] - ohlc['low'], name="range")
        EMA9 = _range.ewm(span=period, ignore_na=False).mean()
        DEMA9 = EMA9.ewm(span=period, ignore_na=False).mean()
        mass = EMA9 / DEMA9
        
        return pd.Series(mass.rolling(window=25).sum(), name='Mass Index').tail(period)

    @classmethod
    def VORTEX(cls, ohlc, period=14):
        """The Vortex indicator plots two oscillating lines, one to identify positive trend movement and the other
         to identify negative price movement. Crosses between the lines trigger buy and sell signals that are designed 
         to capture the most dynamic trending action, higher or lower. 
         There’s no neutral setting for the indicator, which will always generate a bullish or bearish bias. 
         Indicator construction revolves around the highs and lows of the last two days or periods. 
         The distance from the current high to the prior low designates positive trend movement while the
         distance between the current low and the prior high designates negative trend movement. 
         Strongly positive or negative trend movements will show a longer length between the two numbers while 
         weaker positive or negative trend movement will show a shorter length.
         Readings are usually captured over 14 periods (though the technician can choose any length), 
         and then adjusted using technical indicator creator J. Welles Wilder’s True Range.
         Results are posted as continuous lines beneath price bars, while crossovers are compared to other 
         trend-following indicators to produce valid trading signals. 
         Traders can use the vortex indicator as a standalone signal generator, 
         but keep in mind it is vulnerable to significant whipsaws and false signals in congested or mixed markets."""
        
        VMP = pd.Series(ohlc["high"] - ohlc["low"].shift(-1).abs())
        VMM = pd.Series(ohlc["low"] - ohlc["high"].shift(-1).abs())
        
        VMPx = VMP.rolling(window=period).sum().tail(period)
        VMMx = VMM.rolling(window=period).sum().tail(period)
        
        VIp = pd.Series(VMPx / cls.TR(ohlc, period), name="VIp").interpolate(method="index")
        VIm = pd.Series(VMMx / cls.TR(ohlc, period), name="VIm").interpolate(method="index")
        
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

        k = pd.Series((r1 * 1) + (r2 * 2) + (r3 * 3) + (r4 * 4), name="KST")
        signal = k.rolling(window=10).mean()

        return pd.concat([k, signal], axis=1)

    @classmethod
    def TSI(cls, ohlc, long=25, short=13, signal=13):
        """True Strength Index (TSI) is a momentum oscillator based on a double smoothing of price changes."""
        
        ## Double smoother price change
        momentum = pd.Series(ohlc["close"].diff()) ## 1 period momentum
        _EMA25 = pd.Series(momentum.ewm(span=long, min_periods=long-1).mean(), name="_price change EMA25")
        _DEMA13 = pd.Series(_EMA25.ewm(span=short, min_periods=short-1).mean(), name="_price change double smoothed DEMA13")

        ## Double smoothed absolute price change
        absmomentum = pd.Series(ohlc["close"].diff().abs())
        _aEMA25 = pd.Series(absmomentum.ewm(span=long, min_periods=long-1).mean(), name="_abs_price_change EMA25")
        _aDEMA13 = pd.Series(_aEMA25.ewm(span=short, min_periods=short-1).mean(), name="_abs_price_change double smoothed DEMA13")
        
        TSI = pd.Series((_DEMA13 / _aDEMA13) * 100, name="True Strenght Index")
        signal = pd.Series(TSI.ewm(span=signal, min_periods=signal-1).mean(), name="TSI signal")
        
        return pd.concat([TSI, signal], axis=1)
    
    @classmethod
    def TP(cls, ohlc):
        """Typical Price refers to the arithmetic average of the high, low, and closing prices for a given period."""

        return pd.Series((ohlc['high'] + ohlc['low'] + ohlc['close']) / 3, name="TP")

    @classmethod
    def ADL(cls, ohlcv):
        """The accumulation/distribution line was created by Marc Chaikin to determine the flow of money into or out of a security. 
        It should not be confused with the advance/decline line. While their initials might be the same, these are entirely different indicators,
        and their uses are different as well. Whereas the advance/decline line can provide insight into market movements, 
        the accumulation/distribution line is of use to traders looking to measure buy/sell pressure on a security or confirm the strength of a trend."""
         
        MFM = pd.Series( (ohlcv["close"] - ohlcv["low"]) - (ohlcv["high"] - ohlcv["close"]) / (ohlcv["high"] - ohlcv["low"]), name="MFM")# Money flow multiplier
        MFV = pd.Series(MFM * ohlcv["volume"], name="MFV")
        return MFV.cumsum()

    @classmethod
    def CHAIKIN(cls, ohlcv):
        """Chaikin Oscillator, named after its creator, Marc Chaikin, the Chaikin oscillator is an oscillator that measures the accumulation/distribution
         line of the moving average convergence divergence (MACD). The Chaikin oscillator is calculated by subtracting a 10-day exponential moving average (EMA) 
         of the accumulation/distribution line from a three-day EMA of the accumulation/distribution line, and highlights the momentum implied by the 
         accumulation/distribution line."""
        
        return pd.Series(cls.ADL(ohlcv).ewm(span=3, min_periods=2).mean() - cls.ADL(ohlcv).ewm(span=10, min_periods=9).mean())

    @classmethod
    def MFI(cls, ohlcv, period=14):
        """The money flow index (MFI) is a momentum indicator that measures 
        the inflow and outflow of money into a security over a specific period of time.
        MFI can be understood as RSI adjusted for volume.
        The money flow indicator is one of the more reliable indicators of overbought and oversold conditions, perhaps partly because
        it uses the higher readings of 80 and 20 as compared to the RSI's overbought/oversold readings of 70 and 30"""

        tp = cls.TP(ohlcv)
        rmf = pd.Series(tp * ohlcv["volume"], name="rmf") ## Real Money Flow
        _mf = pd.concat([tp, rmf], axis=1)

        mfp = [] # Positive money flow is calculated by adding the money flow of all the days where the typical price is higher than the previous day's typical price.
        mfn = [] # Negative money flow is calculated by adding the money flow of all the days where the typical price is lower than the previous day's typical price.

        for row, _row in zip(_mf.itertuples(), _mf.shift(-1).itertuples()):
            if row.tp > _row.tp:
                mfp.append(row.rmf)
                mfn.append(0)
            else:
                mfn.append(row.rmf)
                mfp.append(0)
        
        _mf["mfp"] = mfp
        _mf["mfn"] = mfn
        _mf["{0} period positive Money Flow".format(period)] = _mf["mfp"].rolling(window=period).sum()
        _mf["{0} period negative Money Flow".format(period)] = _mf["mfn"].rolling(window=period).sum()
        
        mfratio = pd.Series(_mf["{0} period positive Money Flow".format(period)].tail(period) / 
                            _mf["{0} period negative Money Flow".format(period)].tail(period), name="Money Flow Ratio")  
        
        return pd.Series(100 - (100 / (1 + mfratio)), name="Money Flow Index")

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
        
        ohlcv["OBV"] = obv
        return pd.Series(ohlcv["OBV"], name="On Volume Balance")

    @classmethod
    def EFI(cls, ohlcv, period=13):
        """Elder's Force Index is an indicator that uses price and volume to assess the power
         behind a move or identify possible turning points."""

        fi = pd.Series((ohlcv['close'] - ohlcv['close'].diff()) * ohlcv["volume"])
        return pd.Series(fi.ewm(ignore_na=False, min_periods=period-1, span=period).mean(), name="{0} period Force Index")

    @classmethod
    def EMV(cls, ohlcv, period=14):
        """Ease of Movement (EMV) is a volume-based oscillator that fluctuates above and below the zero line. 
        As its name implies, it is designed to measure the "ease" of price movement.
        prices are advancing with relative ease when the oscillator is in positive territory. 
        Conversely, prices are declining with relative ease when the oscillator is in negative territory."""

        distance = pd.Series(((ohlcv['high'] + ohlcv['low']) / 2) - (ohlcv['high'].diff() - ohlcv['low'].diff()) / 2) 
        box_ratio = pd.Series((ohlcv["volume"] / 1000000) / (ohlcv["high"] - ohlcv["low"]))
        
        _emv = pd.Series(distance / box_ratio)
        
        return pd.Series(_emv.rolling(window=period).mean(), name="EMV")

    @classmethod
    def CCI(cls, ohlc, period=20):
        """Commodity Channel Index (CCI) is a versatile indicator that can be used to identify a new trend or warn of extreme conditions.
        CCI measures the current price level relative to an average price level over a given period of time. 
        The CCI typically oscillates above and below a zero line. Normal oscillations will occur within the range of +100 and −100. 
        Readings above +100 imply an overbought condition, while readings below −100 imply an oversold condition.
        As with other overbought/oversold indicators, this means that there is a large probability that the price will correct to more representative levels."""

        tp = cls.TP(ohlc)
        return pd.Series((tp - tp.rolling(window=period).mean()) / (0.015 * tp.mad()), name="{0} period CCI".format(period))

    @classmethod
    def COPP(cls, ohlc):
        """The Coppock Curve is a momentum indicator, it signals buying opportunities when the indicator moved from negative territory to positive territory."""

        roc1 = cls.ROC(ohlc, 14)
        roc2 = cls.ROC(ohlc, 11)
        
        return pd.Series((roc1 + roc2).ewm(span=10, min_periods=9).mean(), name="Coppock Curve")

    @classmethod
    def BASP(cls, ohlc, period=40):
        """BASP serves to identify buying and selling pressure."""

        sp = ohlc["high"] - ohlc["close"]
        bp = ohlc["close"] - ohlc["low"]
        spavg = sp.ewm(span=period, min_periods=period-1).mean()
        bpavg = bp.ewm(span=period, min_periods=period-1).mean()
        
        nbp = bp/bpavg
        nsp = sp/spavg

        varg = ohlc["volume"].ewm(span=period, min_periods=period-1).mean()
        nv = ohlc["volume"] / varg
        
        nbfraw = pd.Series(nbp * nv, name="Buying pressure")
        nsfraw = pd.Series(nsp * nv, name="Selling pressure")
        
        basp_raw = pd.concat([nbfraw, nsfraw], axis=1)

        nbf = pd.Series((nbp * nv).ewm(span=20).mean(), name="Buying pressure normalized.")
        nsf = pd.Series((nsp * nv).ewm(span=20).mean(), name="Selling pressure normalized.")

        return pd.concat([basp_raw, nbf, nsf], axis=1)

    @classmethod
    def CMO(cls, ohlc, period=9):
        """Chande Momentum Oscillator (CMO) - technical momentum indicator invented by the technical analyst Tushar Chande. It is created by calculating the difference between the sum of 
        all recent gains and the sum of all recent losses and then dividing the result by the sum of all price movement over the period. 
        This oscillator is similar to other momentum indicators such as the Relative Strength Index and the Stochastic Oscillator 
        because it is range bounded (+100 and -100)."""

        raise NotImplementedError

    @classmethod
    def CHANDELIER(cls, ohlc, period_1=14, period_2=22, k=3):
        """Chandelier Exit sets a trailing stop-loss based on the Average True Range (ATR). 
        The indicator is designed to keep traders in a trend and prevent an early exit as long as the trend extends. 
        Typically, the Chandelier Exit will be above prices during a downtrend and below prices during an uptrend."""

        l = pd.Series(ohlc["close"].rolling(window=period_2).max() - cls.ATR(ohlc, 22) * k, name="Chandelier exit - long.")
        s = pd.Series(ohlc["close"].rolling(window=period_1).min() - cls.ATR(ohlc, 22) * k, name="Chandelier exit - short.")

        return pd.concat([s, l], axis=1)

    @classmethod
    def WTO(cls, ohlc, channel_lenght=10, average_lenght=21):
        """Wave Trend Oscillator"""

        ap = cls.TP(ohlc)
        esa = ap.ewm(span=channel_lenght).mean()
        d = pd.Series((ap - esa).abs().ewm(span=cl).mean(), name="d")
        ci = (ap - esa) / (0.015 * d)
        wt1 = ci.ewm(span=average_lenght).mean()
        wt2 = wt1.rolling(window=4).mean()

        return pd.concat([wt1, wt2], axis=1)

    @classmethod
    def FTIE(cls, ohlc, period=10):
        """Copyright by HPotter v1.0 01/07/2014
        Market prices do not have a Gaussian probability density function as many traders think. Their probability curve is not bell-shaped.
        But trader can create a nearly Gaussian PDF for prices by normalizing them or creating a normalized indicator such as the 
        relative strength index and applying the Fisher transform. Such a transformed output creates the peak swings as relatively rare events.
        Fisher transform formula is: y = 0.5 * ln ((1+x)/(1-x)) The sharp turning points of these peak swings clearly and unambiguously 
        identify price reversals in a timely manner. """

        """Length = input(10, minval=1)
        xHL2 = hl2
        xMaxH = highest(xHL2, Length)
        xMinL = lowest(xHL2,Length)
        nValue1 = 0.33 * 2 * ((xHL2 - xMinL) / (xMaxH - xMinL) - 0.5) + 0.67 * nz(nValue1[1])
        nValue2 = iff(nValue1 > .99,  .999,
                    iff(nValue1 < -.99, -.999, nValue1))
        nFish = 0.5 * log((1 + nValue2) / (1 - nValue2)) + 0.5 * nz(nFish[1])
        pos =	iff(nFish > nz(nFish[1]), 1,
                iff(nFish < nz(nFish[1]), -1, nz(pos[1], 0))) """
        
        #xHL2 = (ohlc["high"] + ohlc["low"]) / 2
        #xMaxH = hl2.rolling(window=period).max()
        #xMinL = hl2.rolling(window=period).min()
        #nValue1 = 0.33 * 2 * ( (xHL2 - xMinL) / (xMaxH - xMinL) - 0.5) + 0.67 * nValue1[1]
        
        raise NotImplementedError
    
    @classmethod
    def ICHIMOKU(cls, ohlc):
        """The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a versatile indicator that defines support and resistance,
        identifies trend direction, gauges momentum and provides trading signals. 
        Ichimoku Kinko Hyo translates into “one look equilibrium chart”."""

        tenkan_sen = pd.Series((ohlc["high"].rolling(window=9).mean() + ohlc["low"].rolling(window=9).mean()) / 2, name="tenkan_sen") ## conversion line
        kijun_sen = pd.Series((ohlc["high"].rolling(window=26).mean() + ohlc["low"].rolling(window=26).mean()) / 2, name="kijun_sen") ## base line
        
        senkou_span_a = pd.Series(((tenkan_sen / kijun_sen) / 2), name="senkou_span_a") ## Leading span
        senkou_span_b = pd.Series(((ohlc["high"].rolling(window=52).mean() + ohlc["low"].rolling(window=52).mean()) / 2), name="senkou_span_b")
        chikou_span = pd.Series(ohlc["close"].shift(-26).rolling(window=26).mean(), name="chikou_span")

        return pd.concat([tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span], axis=1)

