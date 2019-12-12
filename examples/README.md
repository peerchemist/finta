# Examples

I recommend using [ipython](https://ipython.org/) console while playing with finta.

## Fetching the data

For this example I'll use 5y of AMZN tick data retrieved from [NASDAQ site](https://www.nasdaq.com/market-activity/stocks/amzn/historical) directly.

## Loading and preparing the data

Open the ipython console.

> from finta import TA

> import pandas as pd

Load the .csv:

> ohlc = pd.read_csv("HistoricalQuotes.csv", index_col="Date", parse_dates=True)

Now we need to make this ohlc comply to standards.

Column names:

> ohlc.columns

We need lowercase column names:

> ohlc.columns = ['close', 'volume', 'open', 'high', 'low']

As you can see some of the values in the DataFrame have a "$" prefix. Let's see if we can remove that.
You may notice that values have "$" prefix, we must remove that before continuing.
This small function bellow will do that for us.

```
def split(dollar: str) -> float:
    return float(dollar.split("$")[1])
```

Now apply it to each column:

> ohlc["close"] = ohlc["close"].apply(split)

> ohlc["low"] = ohlc["low"].apply(split)

> ohlc["high"] = ohlc["high"].apply(split)

> ohlc["open"] = ohlc["open"].apply(split)

## TA

Jump right into it to see how easy it is.

> TA.RSI(ohlc).tail(10)

```
Date
2014-12-26    55.099394
2014-12-24    43.666451
2014-12-23    50.085415
2014-12-22    50.594291
2014-12-19    38.730709
2014-12-18    35.584319
2014-12-17    38.632773
2014-12-16    32.701255
2014-12-15    55.449033
2014-12-12    57.338081
Name: RSI, dtype: float64
```

Those are daily candles with standard RSI-14.
How about weekly candles and EMA-5?

Resample the ohlc:

> from finta.utils import resample_calendar

finta.utils has a nice utility: "resample_calendar" which will make nice weekly candles in a jiffy.

> weekly_ohlc = resample_calendar(ohlc, "7d")

> TA.EMA(weekly_ohlc, 5).tail(10)

```
2019-10-04    1756.299843
2019-10-11    1766.693228
2019-10-18    1771.388819
2019-10-25    1773.145879
2019-11-01    1778.163920
2019-11-08    1770.309280
2019-11-15    1758.442853
2019-11-22    1778.465235
2019-11-29    1765.803490
2019-12-06    1760.108994
Freq: W-FRI, Name: 5 period EMA, dtype: float64
```

That's it, you now know the basics of finta.