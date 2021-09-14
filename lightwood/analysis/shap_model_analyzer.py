from typing import Dict, List

import pandas as pd

from lightwood.api import dtype
from lightwood.api.types import ModelAnalysis, StatisticalAnalysis
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
from lightwood.ensemble import BaseEnsemble
from lightwood.analysis.acc_stats import AccStats


def shap_model_analyzer(
        predictor: BaseEnsemble,
        data: List[EncodedDs],
        train_data: List[EncodedDs],
        stats_info: StatisticalAnalysis,
        target: str,
        dtype_dict: Dict[str, str],
):
    encoded_train_data = ConcatedEncodedDs(train_data)
    encoded_data = ConcatedEncodedDs(data)

    data_type = dtype_dict[target]

    is_numerical = data_type in [dtype.integer, dtype.float] or data_type in [dtype.array]
    is_classification = data_type in (dtype.categorical, dtype.binary)

    data = encoded_data.data_frame
    normal_predictions = predictor(encoded_data) if not is_classification else predictor(
        encoded_data, predict_proba=True)
    normal_predictions = normal_predictions.set_index(data.index)

    result_df = pd.DataFrame(index=data.index, columns=['confidence', 'lower', 'upper'], dtype=float)

    acc_stats = AccStats(dtype_dict=dtype_dict, target=target, buckets=stats_info.buckets)
    acc_stats.fit(data, normal_predictions, conf=result_df)
    bucket_accuracy, accuracy_histogram, cm, accuracy_samples = acc_stats.get_accuracy_stats(
        is_classification=is_classification, is_numerical=is_numerical)

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
    return model_analysis, {}
