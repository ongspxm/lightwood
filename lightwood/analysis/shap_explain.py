import shap
import torch
import pandas as pd

from lightwood.api.dtype import dtype
from lightwood.ensemble import BestOf


def shap_explain(data: pd.DataFrame,
            encoded_data: torch.Tensor,
            predictions: pd.DataFrame,
            ensemble: BestOf,
            target_name: str,
            target_dtype: str,
            ):

    data = data.reset_index(drop=True)
    explainer = shap.Explainer(ensemble.models[ensemble.best_index].model)
    shap_values = explainer.shap_values(encoded_data).toarray()
    shap_df = pd.DataFrame(shap_values,
                           columns=[f"feature_{i}_impact" for i in range(shap_values.shape[1])])

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

    # Merge SHAP values into final dataframe
    insights = pd.concat([insights, shap_df], axis=1)

    # Add model base response
    insights['base_response'] = insights.apply(lambda x: x['prediction'] - sum([x[f'feature_{i}_impact'] for i in range(36)]), axis=1)

    return insights
