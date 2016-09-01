# pandas-ta
Common financial technical indicators implemented in Pandas

**This is first release, in the future this will be made into python module and will be instalable via pip.**

TA class is very well documented and there should be no trouble exploring it and using with your data.
Each class method expects proper `ohlc` data as input.

> How to:

`from pandas-ta import TA`

`TA.SMA(ohlc, 42)` will return Pandas Series object with Simple moving average for 42 periods

