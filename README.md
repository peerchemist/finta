# FinTA (Financial Technical Analysis)

[![Financial Contributors on Open Collective](https://opencollective.com/finta/all/badge.svg?label=financial+contributors)](https://opencollective.com/finta) [![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![PyPI](https://img.shields.io/pypi/v/finta.svg?style=flat-square)](https://pypi.python.org/pypi/finta/)
[![](https://img.shields.io/badge/python-3.4+-blue.svg)](https://www.python.org/download/releases/3.4.0/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Build Status](https://travis-ci.org/peerchemist/finta.svg?branch=master)](https://travis-ci.org/peerchemist/finta)

Common financial technical indicators implemented in Pandas.

*This is work in progress, bugs are expected and results of some indicators
may not be accurate.*

## Supported indicators:

Finta supports 75 trading indicators:

```
* Simple Moving Average 'SMA'
* Simple Moving Median 'SMM'
* Smoothed Simple Moving Average 'SSMA'
* Exponential Moving Average 'EMA'
* Double Exponential Moving Average 'DEMA'
* Triple Exponential Moving Average 'TEMA'
* Triangular Moving Average 'TRIMA'
* Triple Exponential Moving Average Oscillator 'TRIX'
* Volume Adjusted Moving Average 'VAMA'
* Kaufman Efficiency Indicator 'ER'
* Kaufman's Adaptive Moving Average 'KAMA'
* Zero Lag Exponential Moving Average 'ZLEMA'
* Weighted Moving Average 'WMA'
* Hull Moving Average 'HMA'
* Elastic Volume Moving Average 'EVWMA'
* Volume Weighted Average Price 'VWAP'
* Smoothed Moving Average 'SMMA'
* Moving Average Convergence Divergence 'MACD'
* Percentage Price Oscillator 'PPO'
* Volume-Weighted MACD 'VW_MACD'
* Elastic-Volume weighted MACD 'EV_MACD'
* Market Momentum 'MOM'
* Rate-of-Change 'ROC'
* Relative Strenght Index 'RSI'
* Inverse Fisher Transform RSI 'IFT_RSI'
* True Range 'TR'
* Average True Range 'ATR'
* Stop-and-Reverse 'SAR'
* Bollinger Bands 'BBANDS'
* Bollinger Bands Width 'BBWIDTH'
* Percent B 'PERCENT_B'
* Keltner Channels 'KC'
* Donchian Channel 'DO'
* Directional Movement Indicator 'DMI'
* Average Directional Index 'ADX'
* Pivot Points 'PIVOT'
* Fibonacci Pivot Points 'PIVOT_FIB'
* Stochastic Oscillator %K 'STOCH'
* Stochastic oscillator %D 'STOCHD'
* Stochastic RSI 'STOCHRSI'
* Williams %R 'WILLIAMS'
* Ultimate Oscillator 'UO'
* Awesome Oscillator 'AO'
* Mass Index 'MI'
* Vortex Indicator 'VORTEX'
* Know Sure Thing 'KST'
* True Strength Index 'TSI'
* Typical Price 'TP'
* Accumulation-Distribution Line 'ADL'
* Chaikin Oscillator 'CHAIKIN'
* Money Flow Index 'MFI'
* On Balance Volume 'OBV'
* Weighter OBV 'WOBV'
* Volume Zone Oscillator 'VZO'
* Price Zone Oscillator 'PZO'
* Elder's Force Index 'EFI'
* Cummulative Force Index 'CFI'
* Bull power and Bear Power 'EBBP'
* Ease of Movement 'EMV'
* Commodity Channel Index 'CCI'
* Coppock Curve 'COPP'
* Buy and Sell Pressure 'BASP'
* Normalized BASP 'BASPN'
* Chande Momentum Oscillator 'CMO'
* Chandelier Exit 'CHANDELIER'
* Qstick 'QSTICK'
* Twiggs Money Index 'TMF'
* Wave Trend Oscillator 'WTO'
* Fisher Transform 'FISH'
* Ichimoku Cloud 'ICHIMOKU'
* Adaptive Price Zone 'APZ'
* Vector Size Indicator 'VR'
* Squeeze Momentum Indicator 'SQZMI'
* Volume Price Trend 'VPT'
* Finite Volume Element 'FVE'
* Volume Flow Indicator 'VFI'
* Moving Standard deviation 'MSD'
```

## Dependencies:

-   python (3.4+)
-   pandas (0.21.1+)

TA class is very well documented and there should be no trouble
exploring it and using with your data. Each class method expects proper `ohlc` DataFrame as input.

## Install:

`pip install finta`

or latest development version:

`pip install git+git://github.com/peerchemist/finta.git`

## Import

`from finta import TA`

Prepare data to use with finta:

finta expects properly formated `ohlc` DataFrame, with column names in `lowercase`:
["open", "high", "low", close"] and ["volume"] for indicators that expect `ohlcv` input.

To prepare the DataFrame into `ohlc` format you can do something as following:

### standardize column names of your source
`df.columns = ["date", 'close', 'volume']`

### set index on the date column, which is requirement to sort it by time periods
`df.set_index('date', inplace=True)`

### select only price column, resample by time period and return daily ohlc (you can choose different time period)
`ohlc = df["close"].resample("24h").ohlc()`


`ohlc()` method applied on the Series above will automatically format the dataframe in format expected by the library so resulting `ohlc` Series is ready to use.
____________________________________________________________________________

## Examples:

### will return Pandas Series object with the Simple moving average for 42 periods
`TA.SMA(ohlc, 42)`

### will return Pandas Series object with "Awesome oscillator" values
`TA.AO(ohlc)`

### expects ["volume"] column as input
`TA.OBV(ohlc)`

### will return Series with Bollinger Bands columns [BB_UPPER, BB_LOWER]
`TA.BBANDS(ohlc)`

### will return Series with calculated BBANDS values but will use KAMA instead of MA for calculation, other types of Moving Averages are allowed as well.
`TA.BBANDS(ohlc, TA.KAMA(ohlc, 20))`

------------------------------------------------------------------------

I welcome pull requests with new indicators or fixes for existing ones.
Please submit only indicators that belong in public domain and are
royalty free.

## Contributing

1. Fork it (https://github.com/peerchemist/finta/fork)
2. Study how it's implemented.
3. Create your feature branch (`git checkout -b my-new-feature`).
4. Run [black](https://github.com/ambv/black) code formatter on the finta.py to ensure uniform code style.
5. Commit your changes (`git commit -am 'Add some feature'`).
6. Push to the branch (`git push origin my-new-feature`).
7. Create a new Pull Request.

------------------------------------------------------------------------

## Donate

Buy me a beer 🍺:

Bitcoin: 39PdX8jhXvUpzpkDibwMAVHVs6ZtoHCjnm

Peercoin: PRn448Km1ZJ2BhdPQfiSS3q4Af2vkjwwvH

## Contributors

### Code Contributors

This project exists thanks to all the people who contribute. [[Contribute](CONTRIBUTING.md)].
<a href="https://github.com/peerchemist/finta/graphs/contributors"><img src="https://opencollective.com/finta/contributors.svg?width=890&button=false" /></a>

### Financial Contributors

Become a financial contributor and help us sustain our community. [[Contribute](https://opencollective.com/finta/contribute)]

#### Individuals

<a href="https://opencollective.com/finta"><img src="https://opencollective.com/finta/individuals.svg?width=890"></a>

#### Organizations

Support this project with your organization. Your logo will show up here with a link to your website. [[Contribute](https://opencollective.com/finta/contribute)]

<a href="https://opencollective.com/finta/organization/0/website"><img src="https://opencollective.com/finta/organization/0/avatar.svg"></a>
<a href="https://opencollective.com/finta/organization/1/website"><img src="https://opencollective.com/finta/organization/1/avatar.svg"></a>
<a href="https://opencollective.com/finta/organization/2/website"><img src="https://opencollective.com/finta/organization/2/avatar.svg"></a>
<a href="https://opencollective.com/finta/organization/3/website"><img src="https://opencollective.com/finta/organization/3/avatar.svg"></a>
<a href="https://opencollective.com/finta/organization/4/website"><img src="https://opencollective.com/finta/organization/4/avatar.svg"></a>
<a href="https://opencollective.com/finta/organization/5/website"><img src="https://opencollective.com/finta/organization/5/avatar.svg"></a>
<a href="https://opencollective.com/finta/organization/6/website"><img src="https://opencollective.com/finta/organization/6/avatar.svg"></a>
<a href="https://opencollective.com/finta/organization/7/website"><img src="https://opencollective.com/finta/organization/7/avatar.svg"></a>
<a href="https://opencollective.com/finta/organization/8/website"><img src="https://opencollective.com/finta/organization/8/avatar.svg"></a>
<a href="https://opencollective.com/finta/organization/9/website"><img src="https://opencollective.com/finta/organization/9/avatar.svg"></a>
