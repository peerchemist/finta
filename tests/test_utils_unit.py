import pytest
import os
from pandas import DataFrame, Series
from finta import TA
from finta.utils import to_dataframe, resample, trending_down, trending_up
import numpy
import json


def rootdir():

    return os.path.dirname(os.path.abspath(__file__))


data_file = os.path.join(rootdir(), "data/poloniex_xrp-btc.json")

with open(data_file, "r") as outfile:
    data = json.load(outfile)


def test_to_dataframe():

    assert isinstance(to_dataframe(data), DataFrame)


def test_resample():

    df = to_dataframe(data)
    assert isinstance(resample(df, "2d"), DataFrame)
    assert list(resample(df, "2d").index.values[-2:]) == [
        numpy.datetime64("2019-05-05T00:00:00.000000000"),
        numpy.datetime64("2019-05-07T00:00:00.000000000"),
    ]


def test_resample_calendar():

    df = to_dataframe(data)
    assert isinstance(resample(df, "W-Mon"), DataFrame)
    assert list(resample(df, "W-Mon").index.values[-2:]) == [
        numpy.datetime64("2019-05-06T00:00:00.000000000"),
        numpy.datetime64("2019-05-13T00:00:00.000000000"),
    ]


def test_trending_up():

    df = to_dataframe(data)
    ma = TA.HMA(df)
    assert isinstance(trending_up(ma, 10), Series)

    assert not trending_up(ma, 10).values[-1]


def test_trending_down():

    df = to_dataframe(data)
    ma = TA.HMA(df)
    assert isinstance(trending_down(ma, 10), Series)

    assert trending_down(ma, 10).values[-1]
