from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from sklearn.preprocessing import OneHotEncoder

from lightwood.api import dtype
from lightwood.api.types import ModelAnalysis, StatisticalAnalysis, TimeseriesSettings
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
from lightwood.helpers.general import evaluate_accuracy
from lightwood.ensemble import BaseEnsemble
from lightwood.encoder.text.pretrained import PretrainedLangEncoder

from lightwood.analysis.acc_stats import AccStats
from lightwood.analysis.nc.norm import Normalizer
from lightwood.analysis.nc.nc import BoostedAbsErrorErrFunc
from lightwood.analysis.nc.util import clean_df, set_conf_range
from lightwood.analysis.nc.icp import IcpRegressor, IcpClassifier
from lightwood.analysis.nc.nc import RegressorNc, ClassifierNc, MarginErrFunc
from lightwood.analysis.nc.wrappers import ConformalClassifierAdapter, ConformalRegressorAdapter, t_softmax


def shap_model_analyzer(
        predictor: BaseEnsemble,
        data: List[EncodedDs],
        train_data: List[EncodedDs],
        stats_info: StatisticalAnalysis,
        target: str,
        ts_cfg: TimeseriesSettings,
        dtype_dict: Dict[str, str],
        disable_column_importance: bool,
        fixed_significance: float,
        positive_domain: bool,
        confidence_normalizer: bool,
        accuracy_functions
):
    runtime_analyzer = {}

    encoded_train_data = ConcatedEncodedDs(train_data)
    encoded_data = ConcatedEncodedDs(data)

    data_type = dtype_dict[target]
    data_subtype = data_type

    is_numerical = data_type in [dtype.integer, dtype.float] or data_type in [dtype.array]
    is_classification = data_type in (dtype.categorical, dtype.binary)
    is_multi_ts = ts_cfg.is_timeseries and ts_cfg.nr_predictions > 1

    data = encoded_data.data_frame
    runtime_analyzer = {}
    predictions = {}
    input_cols = list([col for col in data.columns if col != target])
    normal_predictions = predictor(encoded_data) if not is_classification else predictor(
        encoded_data, predict_proba=True)
    normal_predictions = normal_predictions.set_index(data.index)

    result_df = pd.DataFrame(index=data.index, columns=['confidence', 'lower', 'upper'], dtype=float)

    acc_stats = AccStats(dtype_dict=dtype_dict, target=target, buckets=stats_info.buckets)
    acc_stats.fit(data, normal_predictions, conf=result_df)
    bucket_accuracy, accuracy_histogram, cm, accuracy_samples = acc_stats.get_accuracy_stats(
        is_classification=is_classification, is_numerical=is_numerical)
    runtime_analyzer['bucket_accuracy'] = bucket_accuracy

    model_analysis = ModelAnalysis(
        accuracies={},
        accuracy_histogram=normal_predictions,
        accuracy_samples=accuracy_samples,
        train_sample_size=len(encoded_train_data),
        test_sample_size=len(encoded_data),
        confusion_matrix=cm,
        column_importances={},
        histograms=stats_info.histograms,
        dtypes=dtype_dict
    )

    return model_analysis, runtime_analyzer
