# pandas-ta
Common financial technical indicators implemented in Pandas.

> Dependencies:

* python (3.0 +)
* pandas (0.18 +)

**This is first release, in the future this will be made into python(3) module and will be instalable via pip.**

TA class is very well documented and there should be no trouble exploring it and using with your data.
Each class method expects proper `ohlc` data as input.

> How to:

`from pandas-ta import TA`

`TA.SMA(ohlc, 42)` will return Pandas Series object with Simple moving average for 42 periods


I welcome pull requests with new indicators or fixes for existing ones. Please submit only indicators that belong in public domain and are royalty free.

_______________________________________________________________

Some of the code is based from [pandas_talib](https://github.com/femtotrader/pandas_talib) project but it is so radically changed that fork would not be a valid option.
I invite the authors to merge with my code manually and continue developing on it.