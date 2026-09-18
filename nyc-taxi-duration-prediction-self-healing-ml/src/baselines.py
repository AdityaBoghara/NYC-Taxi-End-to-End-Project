"""Training-only route medians with sample-count fallbacks."""
import numpy as np
import pandas as pd


class RouteMedian:
    """Route + weekday + four-hour period -> route -> pickup -> global median."""

    def __init__(self, min_count=30):
        self.min_count = min_count

    def fit(self, X, y):
        frame = X.copy()
        frame["period"] = frame["hour"] // 4
        frame["target"] = np.asarray(y)
        if len(frame) == 0 or not np.isfinite(frame.target).all():
            raise ValueError("Baseline needs nonempty finite training targets.")
        self.global_median = float(frame.target.median())
        self.levels = []
        for keys in [
            ["PULocationID", "DOLocationID", "dayofweek", "period"],
            ["PULocationID", "DOLocationID"], ["PULocationID"],
        ]:
            grouped = frame.groupby(keys).target.agg(["median", "count"])
            self.levels.append((keys, grouped.loc[grouped["count"] >= self.min_count, "median"]))
        return self

    def predict(self, X):
        frame = X.copy()
        frame["period"] = frame["hour"] // 4
        result = np.full(len(frame), np.nan)
        for keys, table in self.levels:
            index = (pd.Index(frame[keys[0]]) if len(keys) == 1
                     else pd.MultiIndex.from_frame(frame[keys]))
            values = table.reindex(index).to_numpy()
            result = np.where(np.isnan(result), values, result)
        return np.where(np.isnan(result), self.global_median, result)
