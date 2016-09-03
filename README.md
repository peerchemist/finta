# FinTA (Financial Technical Analysis)

Common financial technical indicators implemented in Pandas.

**This is work in progress, bugs are expected and results of indicators
might not be correct.**

> Dependencies:

-   python (3+)
-   pandas (0.18+)

TA class is very well documented and there should be no trouble
exploring it and using with your data. Each class method expects proper
`ohlc` data as input.

> How to:

`git clone https://github.com/peerchemist/finta && cd finta`

`sudo python setup.py install` ## to install globally

or 

`pip install --user .` ## to install locally (as user) which is preffered

`from finta.finta import TA`

finta expectes properly formated `ohlc` dataframe, with column names in `lowercase` ["open", "high", "low", close"] and ["volume"] for some indicators.

To prepare dataframe into ohlc format you can do something as following:

`df.columns = ["date", 'price', 'volume']` ## standardize column names of your source

`df.index = df["date"]` ## set index on the date column, which is requirement to sort it by time periods

`ohlc = df["price"].resample(24h).ohlc()` ## ## select only price column, resample by time period and return daily ohlc (you can choose different time period)

Resulting `ohlc` Series is ready to use with this library.

> Examples:

`TA.SMA(ohlc, 42)` ## will return Pandas Series object with Simple
moving average for 42 periods

`TA.AO(ohlc)` ## will return Pandas Series object with "Awesome oscillator" values

`TA.OBV(ohlc)` ## expects ["volume"] column as input

`TA.BBANDS(ohlc)` ## will return Series with Bollinger Bands columns [upper_bb, SMA, lower_bb, b_bandwith, percent_b]

------------------------------------------------------------------------

I welcome pull requests with new indicators or fixes for existing ones.
Please submit only indicators that belong in public domain and are
royalty free.

------------------------------------------------------------------------

Some of the code is based from
[pandas\_talib](https://github.com/femtotrader/pandas_talib) project but
it is so radically changed that fork would not be a valid option. I urge
the authors to merge with my code manually and continue developing on
it.
