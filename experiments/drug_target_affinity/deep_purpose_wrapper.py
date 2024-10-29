from numbers import Integral, Real

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from imblearn.base import BaseSampler
import DeepPurpose.DTI
import DeepPurpose.utils
from bipartite_learn.base import BaseBipartiteEstimator
from bipartite_learn.utils import _X_is_multipartite

from .data_process import preprocess_dti_for_deeppurpose


class DeepPurposeWrapper(BaseBipartiteEstimator, RegressorMixin):
    _parameter_constraints = {
        "config": [dict],
        "under_sampler": [None, BaseSampler],
        "binarizer": [
            None,
            Interval(Real, None, None, closed="both"),
            callable,
        ],
    }

    def __init__(
        self,
        config,
        *,
        under_sampler=None,
        binarizer=None,
    ):
        self.config = config
        self.under_sampler = under_sampler
        self.binarizer = binarizer

    def fit(self, X, y):
        self._validate_params()
        # self._validate_data(X, y)  # TODO: currently only works on floats

        # FIXME
        if self.under_sampler is not None:
            y_col = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y
            y_col = y_col.reshape(-1, 1)
            self.y_bin_ = (
                self.binarizer(y_col)
                if callable(self.binarizer)
                else y_col < self.binarizer
            )
            self.under_sampler_ = clone(self.under_sampler)
            self.under_sampler_.fit_resample(y_col, self.y_bin_)
            idx = self.under_sampler_.sample_indices_
            print(
                f"[{type(self).__name__}] Resulting {len(idx)} samples from"
                f" originally {y.size}."
            )
        else:
            idx = None

        self.data_fit_ = train_data = preprocess_dti_for_deeppurpose(
            X, y,
            drug_encoding=self.config['drug_encoding'],
            target_encoding=self.config['target_encoding'],
            convert_to_log=False,
            subset=idx,
        )

        self.estimator_ = DeepPurpose.DTI.model_initialize(**self.config)
        self.estimator_.train(train_data)

    def predict(self, X):
        if not _X_is_multipartite(X):
            raise ValueError

        y = np.full((X[0].shape[0], X[1].shape[0]), -1.0)

        data = preprocess_dti_for_deeppurpose(
            X, y,
            drug_encoding=self.config['drug_encoding'],
            target_encoding=self.config['target_encoding'],
            convert_to_log=False,
        )

        return np.array(self.estimator_.predict(data))

