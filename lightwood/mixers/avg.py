import numpy as np
from lightwood.mixers import BaseMixer
from lightwood.constants.lightwood import COLUMN_DATA_TYPES


class AvgEnsemble(BaseMixer):
    def __init__(self, predictors, stop_training_after_seconds=None):
        super().__init__()
        self.supported_types = [COLUMN_DATA_TYPES.NUMERIC]  # TODO: shouldn't need this, do type check in Native
        self.predictors = predictors
        self.stop_training_after_seconds = stop_training_after_seconds

    def _fit(self, train_ds, test_ds):
        """ We don't need to fit anything """
        pass

    def _predict(self, when_data_source, include_extra_data=False):
        """ Predict the mean value of other trained mixers """
        output = {}
        all_preds = {}
        mixers = [p._mixer for p in self.predictors]
        for mixer in mixers:
            preds = mixer.predict(when_data_source)
            for target_col in when_data_source.output_features:
                if target_col['type'] in self.supported_types:
                    output[target_col['name']] = {}
                    if target_col['name'] not in all_preds.keys():
                        all_preds[target_col['name']] = [preds[target_col['name']]]
                    else:
                        all_preds[target_col['name']].append(preds[target_col['name']])

        if len(mixers) > 1:
            for col in all_preds.keys():
                output[col]['predictions'] = np.mean([p['predictions'] for p in all_preds['Sunspots']], axis=0).tolist()
        else:
            output = all_preds
        return output

