from types import SimpleNamespace
from typing import Dict, Tuple

import shap
import pandas as pd

from lightwood.analysis.base import BaseAnalysisBlock


class shap_analyzer_block(BaseAnalysisBlock):
    def __init__(self):
        super().__init__(deps=None)

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        return info

    def explain(self, insights: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        ns = SimpleNamespace(**kwargs)
        explainer = shap.Explainer(ns.predictor.mixers[ns.predictor.best_index].model)
        shap_values = explainer.shap_values(ns.encoded_val_data).toarray()
        shap_df = pd.DataFrame(shap_values,
                               columns=[f"feature_{i}_impact" for i in range(shap_values.shape[1])])

        # Merge SHAP values into final dataframe
        insights = pd.concat([insights, shap_df], axis=1)

        # Add model base response
        insights['base_response'] = insights.apply(
            lambda x: x['prediction'] - sum([x[f'feature_{i}_impact'] for i in range(36)]), axis=1)

        return insights

