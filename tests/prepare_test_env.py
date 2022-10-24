"""Prepares test environment. 
 - Adds the root folder to paths so package will be importable.
 - Defines some constants to speed up tests."""

import sys, os
import pandas as pd

from pandas import DataFrame

from typing import Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TEST_RESULTS_CSV: str = "./tests/datasets/test-results.csv"
TEST_SMALL_RESULTS_CSV: str = "./tests/datasets/test-small-results.csv"

READ_TEST_CSV: Callable[[str], DataFrame] = lambda path=TEST_RESULTS_CSV: pd.read_csv(
    path, index_col="time", parse_dates=["time"]
)
