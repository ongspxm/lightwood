from copy import deepcopy

import torch
import numpy as np
import pandas as pd

# restore_icp_state, clear_icp_state
from lightwood.analysis.nc.util import get_numerical_conf_range, get_categorical_conf, get_anomalies
from lightwood.helpers.ts import get_inferred_timestamps, add_tn_conf_bounds
from lightwood.api.dtype import dtype
from lightwood.api.types import TimeseriesSettings
from lightwood.ensemble import BestOf
import shap


def shap_explain(data: pd.DataFrame,
            encoded_data: torch.Tensor,
            predictions: pd.DataFrame,
            ensemble: BestOf,
            timeseries_settings: TimeseriesSettings,
            analysis: dict,
            target_name: str,
            target_dtype: str,
            positive_domain: bool,
            fixed_confidence: float,
            anomaly_detection: bool,

            # forces specific confidence level in ICP
            anomaly_error_rate: float,

            # ignores anomaly detection for N steps after an
            # initial anomaly triggers the cooldown period;
            # implicitly assumes series are regularly spaced
            anomaly_cooldown: int,

            ts_analysis: dict = None
            ):

    data = data.reset_index(drop=True)

    explainer = shap.Explainer(ensemble.models[ensemble.best_index].model)
    shap_values = explainer(encoded_data)
    shap.plots.waterfall(shap_values[0])

    insights = pd.DataFrame()
    if target_name in data.columns:
        insights['truth'] = data[target_name]
    else:
        insights['truth'] = [None] * len(predictions['prediction'])
    insights['prediction'] = predictions['prediction']

    # Make sure the target and real values are of an appropriate type
    if target_dtype in (dtype.integer):
        insights['prediction'] = insights['prediction'].astype(int)
    elif target_dtype in (dtype.float):
        insights['prediction'] = insights['prediction'].astype(float)
    elif target_dtype in (dtype.short_text, dtype.rich_text, dtype.binary, dtype.categorical):
        insights['prediction'] = insights['prediction'].astype(str)

    return insights
