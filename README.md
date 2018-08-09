# FinTA (Financial Technical Analysis)

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![PyPI](https://img.shields.io/pypi/v/finta.svg?style=flat-square)](https://pypi.python.org/pypi/finta/)
[![](https://img.shields.io/badge/python-3.4+-blue.svg)](https://www.python.org/download/releases/3.4.0/)
[![Build Status](https://travis-ci.org/peerchemist/finta.svg?branch=master)](https://travis-ci.org/peerchemist/finta)

Common financial technical indicators implemented in Pandas.

**This is work in progress, bugs are expected and results of indicators
might not be correct.**

> Supported indicators:

```
['SMA', 'SMM', 'EMA', 'DEMA', 'TEMA', 'TRIMA', 'TRIX', 'AMA', 'LWMA', 'VAMA', 'VIDYA', 'ER', 'KAMA', 'ZLEMA', 'WMA', 'HMA', 'VWAP', 'SMMA', 'ALMA', 'MAMA', 'FRAMA', 'MACD', 'PPO', 'VW_MACD', 'MOM', 'ROC', 'RSI', 'IFT_RSI', 'SWI', 'TR', 'ATR', 'SAR', 'BBANDS', 'BBWIDTH', 'PERCENT_B', 'KC', 'DO', 'DMI', 'ADX', 'PIVOTS', 'STOCH', 'STOCHD', 'STOCHRSI', 'WILLIAMS', 'UO', 'AO', 'MI', 'VORTEX', 'KST', 'TSI', 'TP', 'ADL', 'CHAIKIN', 'MFI', 'OBV', 'WOBV', 'VZO', 'EFI', 'CFI', 'EBBP', 'EMV', 'CCI', 'COPP', 'BASP', 'BASPN', 'CMO', 'CHANDELIER', 'QSTICK', 'TMF', 'WTO', 'FISH', 'ICHIMOKU', 'APZ', 'VR', 'SQZMI', 'VPT', 'FVE', 'VFI']
```

> Dependencies:

-   python (3.4+)
-   pandas (0.21.1+)

TA class is very well documented and there should be no trouble
exploring it and using with your data. Each class method expects proper
`ohlc` data as input.

## Install:

`pip install finta`

or latest development version:

`pip install git+git://github.com/peerchemist/finta.git`

### Import

`from finta import TA`

> Prepare data to use with Finta:

finta expects properly formated `ohlc` DataFrame, with column names in `lowercase`:
 ["open", "high", "low", close"] and ["volume"] for indicators that expect `ohlcv` input.

To prepare the DataFrame into `ohlc` format you can do something as following:

`df.columns = ["date", 'close', 'volume']`  # standardize column names of your source

`df.set_index('date', inplace=True)`  # set index on the date column, which is requirement to sort it by time periods

`ohlc = df["close"].resample("24h").ohlc()`  # select only price column, resample by time period and return daily ohlc (you can choose different time period)

`ohlc()` method applied on the Series above will automatically format the dataframe in format expected by the library so resulting `ohlc` Series is ready to use.

____________________________________________________________________________

> Examples:

`TA.SMA(ohlc, 42)` ## will return Pandas Series object with Simple
moving average for 42 periods

`TA.AO(ohlc)` ## will return Pandas Series object with "Awesome oscillator" values

`TA.OBV(ohlc)` ## expects ["volume"] column as input

`TA.BBANDS(ohlc)` ## will return Series with Bollinger Bands columns [BB_UPPER, BB_LOWER, BB_MIDDLE]

`TA.BBANDS(ohlc, TA.KAMA(ohlc, 20))` ## will return Series with calculated BBANDS values but will use KAMA instead of MA for calculation, other types of Moving Averages are allowed as well.

------------------------------------------------------------------------

I welcome pull requests with new indicators or fixes for existing ones.
Please submit only indicators that belong in public domain and are
royalty free.

## Contributing

1. Fork it (https://github.com/peerchemist/finta/fork)
2. Study how it's implemented
3. Create your feature branch (`git checkout -b my-new-feature`)
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin my-new-feature`)
6. Create a new Pull Request

------------------------------------------------------------------------

## Donate

Support the development by donating in cryptocurrency:

XBT: 3PTyUNfn4uoSZGQ48tGMnqorca1DW9Xs4M

XPC: PFdR14r9JM2EQSDh9nRZQ6EW5yzHjNJz3E
