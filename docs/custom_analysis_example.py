import numpy as np
import pandas as pd

from lightwood.api.high_level import ProblemDefinition, predictor_from_code, json_ai_from_problem, code_from_json_ai

np.random.seed(42)


if __name__ == '__main__':
    # Load data and define the task
    df = pd.read_csv("https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv")

    train_df = df.sample(frac=0.8)
    test_df = df[~df.index.isin(train_df.index)]

    target = 'rental_price'
    pdef = ProblemDefinition.from_dict({'target': target,  # column to predict
                                        'time_aim': 30,    # time budget to build a predictor
                                        'nfolds': 10})

    # name and generate predictor code
    p_name = 'home_rentals_shap'
    json_ai = json_ai_from_problem(train_df, problem_definition=pdef)

    # custom model analysis by overriding JsonAI analyzer with custom modules
    json_ai.imports = [
        "from lightwood.analysis.shap_model_analyzer import shap_model_analyzer",
        "from lightwood.analysis.shap_explain import shap_explain"
    ]

    json_ai.analyzer = {
        'module': 'shap_model_analyzer',
        'args': {
            'stats_info': '$statistical_analysis',
            'predictor': '$ensemble',
            'data': 'test_data',
            'train_data': 'train_data',
            'target': '$target',
            'dtype_dict': '$dtype_dict',
        }
    }

    json_ai.explainer = {
        'module': 'shap_explain',
        'args': {
            'data': 'data',
            'encoded_data': 'encoded_data',
            'predictions': 'df',
            'ensemble': 'self.ensemble',
            'target_name': '$target',
            'target_dtype': '$dtype_dict[self.target]',
        }
    }

    # force LightGBM for fast SHAP support
    json_ai.outputs[target].models = [json_ai.outputs[target].models[1]]

    predictor_class_code = code_from_json_ai(json_ai)

    # instantiate and train predictor
    predictor = predictor_from_code(predictor_class_code)
    predictor.learn(train_df)

    # save predictor and its code
    predictor.save(f'./{p_name}.pkl')
    with open(f'./{p_name}.py', 'wb') as fp:
        fp.write(predictor_class_code.encode('utf-8'))

    train_preds = predictor.predict(train_df)
    test_preds = predictor.predict(test_df)
