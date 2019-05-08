import pandas as pd


def to_dataframe(ticks: list) -> pd.DataFrame:
    """Convert list to Series compatible with the library."""

    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index("time", inplace=True)

    return df


def resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample DataFrame by <interval>."""

    d = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

    return df.resample(interval).agg(d)
