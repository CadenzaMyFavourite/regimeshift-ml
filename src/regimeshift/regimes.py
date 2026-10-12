"""Training-fitted clusters of daily observable market summaries."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import SUMMARY_FEATURES


@dataclass
class RegimeModel:
    preprocessing: object
    clustering: KMeans
    fit_dates: list

    @classmethod
    def fit(cls, summaries, seed=42):
        preprocessing = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
        )
        transformed = preprocessing.fit_transform(
            summaries.select(SUMMARY_FEATURES).to_numpy().astype(np.float64)
        )
        if len(np.unique(transformed, axis=0)) < 3:
            raise ValueError("Training data needs at least three distinct daily summaries")
        model = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(transformed)
        return cls(preprocessing, model, summaries["date_id"].to_list())

    def assign(self, previous_summary):
        values = np.array([[previous_summary.get(name, np.nan) for name in SUMMARY_FEATURES]],
                          dtype=float)
        return int(self.clustering.predict(self.preprocessing.transform(values))[0])

    def describe(self):
        centers = self.preprocessing[-1].inverse_transform(self.clustering.cluster_centers_)
        return [{"regime": r, **dict(zip(SUMMARY_FEATURES, center.tolist()))}
                for r, center in enumerate(centers)]
