# TODO: type hint the returns
# TODO: df_std_dev is not clear in behavior; this would imply all std. of each column but that is not true, it should be renamed df_std_target_dev  # noqa

from typing import Dict, List, Optional, Union
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from dataclasses import dataclass
from lightwood.helpers.log import log
from dataclasses_json import dataclass_json
from dataclasses_json.core import _asdict, Json
import json


# See: https://www.python.org/dev/peps/pep-0589/ for how this works
# Not very intuitive but very powerful abstraction, might be useful in other places (@TODO)
class Module(TypedDict):
    """
    Modules are the blocks of code that end up being called from the JSON AI, representing either object instantiations or function calls.

    :param module: Name of the module (function or class name)
    :param args: Argument to pass to the function or constructor
    """ # noqa
    module: str
    args: Dict[str, str]


@dataclass
class Feature:
    """
    Within a dataframe, each column is considered its own "feature" (unless ignored etc.). \
        The following expects each feature to have descriptions of the following:

    :param encoder: the methodology for encoding a feature (a Lightwood Encoder)
    :param data_dtype: The type of information within this column (ex.: numerical, categorical, etc.)
    :param dependency: Any custom attributes for this feature that may require non-standard processing. This highly\
    depends on the encoder (ex: Pretrained text may be fine-tuned on the target; time-series requires prior time-steps).
    """

    encoder: Module
    data_dtype: str = None
    dependency: List[str] = None

    @staticmethod
    def from_dict(obj: Dict):
        """
        Create ``Feature`` objects from the a dictionary representation.

        :param obj: A dictionary representation of a column feature's attributes. Must include keys *encoder*, \
            *data_dtype*, and *dependency*.

        :Example:

        >>> my_dict = {"feature_A": {"encoder": MyEncoder, "data_dtype": "categorical", "dependency": None}}
        >>> print(Feature.from_dict(my_dict["feature_A"]))
        >>> Feature(encoder=None, data_dtype='categorical', dependency=None)

        :returns: A Feature object with loaded information.
        """
        encoder = obj["encoder"]
        data_dtype = obj.get("data_dtype", None)
        dependency = obj.get("dependency", None)

        feature = Feature(encoder=encoder, data_dtype=data_dtype, dependency=dependency)

        return feature

    @staticmethod
    def from_json(data: str):
        """
        Create ``Feature`` objects from JSON representation. This method calls on :ref: `from_dict` after loading the \
            json config.

        :param data: A JSON representation of the feature.

        :returns: Loaded information into the Feature representation.
        """
        return Feature.from_dict(json.loads(data))

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        """
        Converts a Feature to a dictionary representation.

        :returns: A python dictionary with strings indicating the three key elements and their respective values of \
            the Feature class.
        """
        as_dict = _asdict(self, encode_json=encode_json)
        for k in list(as_dict.keys()):
            if as_dict[k] is None:
                del as_dict[k]
        return as_dict

    def to_json(self) -> Dict[str, Json]:
        """
        Converts a Feature into a JSON object. Calls ``to_dict`` under the hood.

        :returns: Json config syntax for the three key elements and their respective values of the Feature class.
        """
        return json.dumps(self.to_dict(), indent=4)


@dataclass_json
@dataclass
class Output:
    """
    A representation for the output feature. This is specifically used on the target column of your dataset. \
    Four attributes are expected as seen below.

    Note, currently supervised tasks are supported, hence categorical, numerical, and time-series are the expected \
    outputs types. Complex features such as text generation are not currently available by default.

    :param data_dtype: The type of information within the target column (ex.: numerical, categorical, etc.).
    :param encoder: the methodology for encoding the target feature (a Lightwood Encoder). There can only be one \
    encoder for the output target.
    :param mixers: The list of ML algorithms that are trained for the target distribution.
    :param ensemble: For a panel of ML algorithms, the approach of selecting the best mixer, and the metrics used in \
    that evaluation.
    """

    data_dtype: str
    encoder: str = None
    mixers: List[str] = None
    ensemble: str = None


@dataclass_json
@dataclass
class TypeInformation:
    """
    For a dataset, provides information on columns types, how they're used, and any other potential identifiers.

    TypeInformation is generated within ``data.infer_types``, where small samples of each column are evaluated in a custom framework to understand what kind of data type the model is. The user may override data types, but it is recommended to do so within a JSON-AI config file.

    :param dtypes: For each column's name, the associated data type inferred.
    :param additional_info: Any possible sub-categories or additional descriptive information.
    :param identifiers: Columns within the dataset highly suspected of being identifiers or IDs. These do not contain informatic value, therefore will be ignored in subsequent training/analysis procedures unless manually indicated.
    """ # noqa

    dtypes: Dict[str, str]
    additional_info: Dict[str, object]
    identifiers: Dict[str, str]

    def __init__(self):
        self.dtypes = dict()
        self.additional_info = dict()
        self.identifiers = dict()


@dataclass_json
@dataclass
class StatisticalAnalysis:
    """
    The Statistical Analysis data class allows users to consider key descriptors of their data using simple \
        techniques such as histograms, mean and standard deviation, word count, missing values, and any detected bias\
             in the information.

    :param nr_rows: Number of rows (samples) in the dataset
    :param df_std_dev: The standard deviation of the target of the dataset
    :param train_observed_classes:
    :param target_class_distribution:
    :param histograms:
    :param buckets:
    :param missing:
    :param distinct:
    :param bias:
    :param avg_words_per_sentence:
    :param positive_domain:
    """

    nr_rows: int
    df_std_dev: Optional[float]
    train_observed_classes: object  # Union[None, List[str]]
    target_class_distribution: object  # Dict[str, float]
    histograms: object  # Dict[str, Dict[str, List[object]]]
    buckets: object  # Dict[str, Dict[str, List[object]]]
    missing: object
    distinct: object
    bias: object
    avg_words_per_sentence: object
    positive_domain: bool


@dataclass_json
@dataclass
class DataAnalysis:
    """
    Data Analysis wraps :class: `.StatisticalAnalysis` and :class: `.TypeInformation` together. Further details can be seen in their respective documentation references.
    """ # noqa

    statistical_analysis: StatisticalAnalysis
    type_information: TypeInformation


@dataclass
class TimeseriesSettings:
    """
    For time-series specific problems, more specific treatment of the data is necessary. The following attributes \
        enable time-series tasks to be carried out properly.

    :param is_timeseries: Whether the input data should be treated as time series; if true, this flag is checked in \
        subsequent internal steps to ensure processing is appropriate for time-series data.
    :param order_by: A list of columns by which the data should be ordered.
    :param group_by: Optional list of columns by which the data should be grouped. Each different combination of values\
         for these columns will yield a different series.
    :param window: The temporal horizon (number of rows) that a model intakes to "look back" into when making a\
         prediction, after the rows are ordered by order_by columns and split into groups if applicable.
    :param nr_predictions: The number of points in the future that predictions should be made for, defaults to 1. Once \
        trained, the model will be able to predict up to this many points into the future.
    :param historical_columns: The temporal dynamics of these columns will be used as additional context to train the \
        time series predictor. Note that a non-historical column shall still be used to forecast, but without \
            considering their change through time.
    :param target_type: Automatically inferred dtype of the target (e.g. `dtype.integer`, `dtype.float`).
    :param use_previous_target: Use the previous values of the target column to generate predictions. Defaults to True.
    """

    is_timeseries: bool
    order_by: List[str] = None
    window: int = None
    group_by: List[str] = None
    use_previous_target: bool = True
    nr_predictions: int = None
    historical_columns: List[str] = None
    target_type: str = (
        ""  # @TODO: is the current setter (outside of initialization) a sane option?
        # @TODO: George: No, I don't think it is, we need to pass this some other way
    )
    allow_incomplete_history: bool = False

    @staticmethod
    def from_dict(obj: Dict):
        """
        Creates a TimeseriesSettings object from python dictionary specifications.

        :param: obj: A python dictionary with the necessary representation for time-series. The only mandatory columns are ``order_by`` and ``window``.

        :returns: A populated ``TimeseriesSettings`` object.
        """ # noqa
        if len(obj) > 0:
            for mandatory_setting in ["order_by", "window"]:
                if mandatory_setting not in obj:
                    err = f"Missing mandatory timeseries setting: {mandatory_setting}"
                    log.error(err)
                    raise Exception(err)

            timeseries_settings = TimeseriesSettings(
                is_timeseries=True,
                order_by=obj["order_by"],
                window=obj["window"],
                use_previous_target=obj.get("use_previous_target", True),
                historical_columns=[],
                nr_predictions=obj.get("nr_predictions", 1),
                allow_incomplete_history=obj.get('allow_incomplete_history', False)
            )
            for setting in obj:
                timeseries_settings.__setattr__(setting, obj[setting])

        else:
            timeseries_settings = TimeseriesSettings(is_timeseries=False)

        return timeseries_settings

    @staticmethod
    def from_json(data: str):
        """
        Creates a TimeseriesSettings object from JSON specifications via python dictionary.

        :param: data: JSON-config file with necessary Time-series specifications

        :returns: A populated ``TimeseriesSettings`` object.
        """
        return TimeseriesSettings.from_dict(json.loads(data))

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        """
        Creates a dictionary from ``TimeseriesSettings`` object

        :returns: A python dictionary containing the ``TimeSeriesSettings`` specifications.
        """
        return _asdict(self, encode_json=encode_json)

    def to_json(self) -> Dict[str, Json]:
        """
        Creates JSON config from TimeseriesSettings object
        :returns: The JSON config syntax containing the ``TimeSeriesSettings`` specifications.
        """
        return json.dumps(self.to_dict())


@dataclass
class ProblemDefinition:
    """
    The ``ProblemDefinition`` object indicates details on how the models that predict the target are prepared. \
        The only required specification from a user is the ``target``, which indicates the column within the input \
        data that the user is trying to predict. Within the ``ProblemDefinition``, the user can specify aspects \
        about how long the feature-engineering preparation may take, and nuances about training the models.

    :param target: The name of the target column; this is the column that will be used as the goal of the prediction.
    :param pct_invalid: Number of data points maximally tolerated as invalid/missing/unknown. \
        If the data cleaning process exceeds this number, no subsequent steps will be taken.
    :param unbias_target: all classes are automatically weighted inverse to how often they occur
    :param seconds_per_mixer: Number of seconds maximum to spend PER mixer trained in the list of possible mixers.
    :param seconds_per_encoder: Number of seconds maximum to spend when training an encoder that requires data to \
    learn a representation.
    :param time_aim: Time budget (in seconds) to train all needed components for the predictive tasks, including \
        encoders and models.
    :param target_weights: indicates to the accuracy functions how much to weight every target class.
    :param positive_domain: For numerical taks, force predictor output to be positive (integer or float).
    :param timeseries_settings: TimeseriesSettings object for time-series tasks, refer to its documentation for \
         available settings.
    :param anomaly_detection: Whether to conduct unsupervised anomaly detection; currently supported only for time-\
        series.
    :param ignore_features: The names of the columns the user wishes to ignore in the ML pipeline. Any column name \
        found in this list will be automatically removed from subsequent steps in the ML pipeline.
    :param fit_on_all: Whether to fit the model on the held-out validation data. Validation data is strictly \
        used to evaluate how well a model is doing and is NEVER trained. However, in cases where users anticipate new \
            incoming data over time, the user may train the model further using the entire dataset.
    :param strict_mode: crash if an `unstable` block (mixer, encoder, etc.) fails to run.
    :param seed_nr: custom seed to use when generating a predictor from this problem definition.
    """

    target: str
    pct_invalid: float
    unbias_target: bool
    seconds_per_mixer: Union[int, None]
    seconds_per_encoder: Union[int, None]
    time_aim: Union[int, None]
    target_weights: Union[List[float], None]
    positive_domain: bool
    timeseries_settings: TimeseriesSettings
    anomaly_detection: bool
    ignore_features: List[str]
    fit_on_all: bool
    strict_mode: bool
    seed_nr: int

    @staticmethod
    def from_dict(obj: Dict):
        """
        Creates a ProblemDefinition object from a python dictionary with necessary specifications.

        :param obj: A python dictionary with the necessary features for the ``ProblemDefinition`` class.
        Only requires ``target`` to be specified.

        :returns: A populated ``ProblemDefinition`` object.
        """
        target = obj['target']
        pct_invalid = obj.get('pct_invalid', 2)
        unbias_target = obj.get('unbias_target', True)
        seconds_per_mixer = obj.get('seconds_per_mixer', None)
        seconds_per_encoder = obj.get('seconds_per_encoder', None)
        time_aim = obj.get('time_aim', None)
        target_weights = obj.get('target_weights', None)
        positive_domain = obj.get('positive_domain', False)
        timeseries_settings = TimeseriesSettings.from_dict(obj.get('timeseries_settings', {}))
        anomaly_detection = obj.get('anomaly_detection', True)
        ignore_features = obj.get('ignore_features', [])
        fit_on_all = obj.get('fit_on_all', True)
        strict_mode = obj.get('strict_mode', True)
        seed_nr = obj.get('seed_nr', 420)
        problem_definition = ProblemDefinition(
            target=target,
            pct_invalid=pct_invalid,
            unbias_target=unbias_target,
            seconds_per_mixer=seconds_per_mixer,
            seconds_per_encoder=seconds_per_encoder,
            time_aim=time_aim,
            target_weights=target_weights,
            positive_domain=positive_domain,
            timeseries_settings=timeseries_settings,
            anomaly_detection=anomaly_detection,
            ignore_features=ignore_features,
            fit_on_all=fit_on_all,
            strict_mode=strict_mode,
            seed_nr=seed_nr
        )

        return problem_definition

    @staticmethod
    def from_json(data: str):
        """
        Creates a ProblemDefinition Object from JSON config file.

        :param data:

        :returns: A populated ProblemDefinition object.
        """
        return ProblemDefinition.from_dict(json.loads(data))

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        """
        Creates a python dictionary from the ProblemDefinition object

        :returns: A python dictionary
        """
        return _asdict(self, encode_json=encode_json)

    def to_json(self) -> Dict[str, Json]:
        """
        Creates a JSON config from the ProblemDefinition object

        :returns: TODO
        """
        return json.dumps(self.to_dict())


@dataclass
class JsonAI:
    """
    The JsonAI Class allows users to construct flexible JSON config to specify their ML pipeline. JSON-AI follows a \
    recipe of how to pre-process data, construct features, and train on the target column. To do so, the following \
    specifications are required internally.

    :param features: The corresponding``Feature`` object for each of the column names of the dataset
    :param outputs: The column name of the target and its ``Output`` object
    :param problem_definition: The ``ProblemDefinition`` criteria.
    :param identifiers: A dictionary of column names and respective data types that are likely identifiers/IDs within the data. Through the default cleaning process, these are ignored.
    :param cleaner: The Cleaner object represents the pre-processing step on a dataframe. The user can specify custom subroutines, if they choose, on how to handle preprocessing. Alternatively, "None" suggests Lightwood's default approach in ``data.cleaner``.
    :param splitter: The Splitter object is the method in which the input data is split into training/validation/testing data.
    :param analyzer: The Analyzer object is used to evaluate how well a model performed on the predictive task.
    :param explainer: The Explainer object deploys explainability tools of interest on a model to indicate how well a model generalizes its predictions.
    :param analysis_blocks: The blocks that get used in both analysis and inference inside the analyzer and explainer blocks.
    :param timeseries_transformer: Procedure used to transform any timeseries task dataframe into the format that lightwood expects for the rest of the pipeline.  
    :param timeseries_analyzer: Procedure that extracts key insights from any timeseries in the data (e.g. measurement frequency, target distribution, etc).
    :param accuracy_functions: A list of performance metrics used to evaluate the best mixers.
    """ # noqa

    features: Dict[str, Feature]
    outputs: Dict[str, Output]
    problem_definition: ProblemDefinition
    identifiers: Dict[str, str]
    cleaner: Optional[Module] = None
    splitter: Optional[Module] = None
    analyzer: Optional[Module] = None
    explainer: Optional[Module] = None
    analysis_blocks: Optional[List[Module]] = None
    timeseries_transformer: Optional[Module] = None
    timeseries_analyzer: Optional[Module] = None
    accuracy_functions: Optional[List[str]] = None

    @staticmethod
    def from_dict(obj: Dict):
        """
        Creates a JSON-AI object from dictionary specifications of the JSON-config.
        """
        features = {k: Feature.from_dict(v) for k, v in obj["features"].items()}
        outputs = {k: Output.from_dict(v) for k, v in obj["outputs"].items()}
        problem_definition = ProblemDefinition.from_dict(obj["problem_definition"])
        identifiers = obj["identifiers"]
        cleaner = obj.get("cleaner", None)
        splitter = obj.get("splitter", None)
        analyzer = obj.get("analyzer", None)
        explainer = obj.get("explainer", None)
        analysis_blocks = obj.get("analysis_blocks", None)
        timeseries_transformer = obj.get("timeseries_transformer", None)
        timeseries_analyzer = obj.get("timeseries_analyzer", None)
        accuracy_functions = obj.get("accuracy_functions", None)

        json_ai = JsonAI(
            features=features,
            outputs=outputs,
            problem_definition=problem_definition,
            identifiers=identifiers,
            cleaner=cleaner,
            splitter=splitter,
            analyzer=analyzer,
            explainer=explainer,
            analysis_blocks=analysis_blocks,
            timeseries_transformer=timeseries_transformer,
            timeseries_analyzer=timeseries_analyzer,
            accuracy_functions=accuracy_functions,
        )

        return json_ai

    @staticmethod
    def from_json(data: str):
        """ Creates a JSON-AI object from JSON config"""
        return JsonAI.from_dict(json.loads(data))

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        """
        Creates a python dictionary with necessary modules within the ML pipeline specified from the JSON-AI object.

        :returns: A python dictionary that has the necessary components of the ML pipeline for a given dataset.
        """
        as_dict = _asdict(self, encode_json=encode_json)
        for k in list(as_dict.keys()):
            if k == "features":
                feature_dict = {}
                for name in self.features:
                    feature_dict[name] = self.features[name].to_dict()
                as_dict[k] = feature_dict
            if as_dict[k] is None:
                del as_dict[k]
        return as_dict

    def to_json(self) -> Dict[str, Json]:
        """
        Creates JSON config to represent the necessary modules within the ML pipeline specified from the JSON-AI object.

        :returns: A JSON config that has the necessary components of the ML pipeline for a given dataset.
        """
        return json.dumps(self.to_dict(), indent=4)


@dataclass_json
@dataclass
class ModelAnalysis:
    """
    The ``ModelAnalysis`` class stores useful information to describe a model and understand its predictive performance on a validation dataset.
    For each trained ML algorithm, we store:

    :param accuracies: Dictionary with obtained values for each accuracy function (specified in JsonAI)
    :param accuracy_histogram: Dictionary with histograms of reported accuracy by target value.
    :param accuracy_samples: Dictionary with sampled pairs of observed target values and respective predictions.
    :param train_sample_size: Size of the training set (data that parameters are updated on)
    :param test_sample_size: Size of the testing set (explicitly held out)
    :param column_importances: Dictionary with the importance of each column for the model, as estimated by an approach that closely follows a leave-one-covariate-out strategy.
    :param confusion_matrix: A confusion matrix for the validation dataset.
    :param histograms: Histogram for each dataset feature.
    :param dtypes: Inferred data types for each dataset feature.

    """ # noqa

    accuracies: Dict[str, float]
    accuracy_histogram: Dict[str, list]
    accuracy_samples: Dict[str, list]
    train_sample_size: int
    test_sample_size: int
    column_importances: Dict[str, float]
    confusion_matrix: object
    histograms: object
    dtypes: object


@dataclass
class PredictionArguments:
    """
    This class contains all possible arguments that can be passed to a Lightwood predictor at inference time.
    On each predict call, all arguments included in a parameter dictionary will update the respective fields
    in the `PredictionArguments` instance that the predictor will have.
    
    :param predict_proba: triggers (where supported) predictions in raw probability output form. I.e. for classifiers,
    instead of returning only the predicted class, the output additionally includes the assigned probability for
    each class.   
    :param all_mixers: forces an ensemble to return predictions emitted by all its internal mixers. 
    :param fixed_confidence: For analyzer module, specifies a fixed `alpha` confidence for the model calibration so \
        that predictions, in average, are correct `alpha` percent of the time.
    :param anomaly_error_rate: Error rate for unsupervised anomaly detection. Bounded between 0.01 and 0.99 \
        (respectively implies wider and tighter bounds, all other parameters being equal).
    :param anomaly_cooldown: Sets the minimum amount of timesteps between consecutive firings of the the anomaly \
        detector.
    """  # noqa

    predict_proba: bool = False
    all_mixers: bool = False
    fixed_confidence: Union[int, float, None] = None
    anomaly_error_rate: Union[float, None] = None
    anomaly_cooldown: int = 1

    @staticmethod
    def from_dict(obj: Dict):
        """
        Creates a ``PredictionArguments`` object from a python dictionary with necessary specifications.

        :param obj: A python dictionary with the necessary features for the ``PredictionArguments`` class.

        :returns: A populated ``PredictionArguments`` object.
        """

        # maybe this should be stateful instead, and save the latest used value for each field?
        predict_proba = obj.get('predict_proba', PredictionArguments.predict_proba)
        all_mixers = obj.get('all_mixers', PredictionArguments.all_mixers)
        fixed_confidence = obj.get('fixed_confidence', PredictionArguments.fixed_confidence)
        anomaly_error_rate = obj.get('anomaly_error_rate', PredictionArguments.anomaly_error_rate)
        anomaly_cooldown = obj.get('anomaly_cooldown', PredictionArguments.anomaly_cooldown)

        pred_args = PredictionArguments(
            predict_proba=predict_proba,
            all_mixers=all_mixers,
            fixed_confidence=fixed_confidence,
            anomaly_error_rate=anomaly_error_rate,
            anomaly_cooldown=anomaly_cooldown,
        )

        return pred_args

    def to_dict(self, encode_json=False) -> Dict[str, Json]:
        """
        Creates a python dictionary from the ``PredictionArguments`` object

        :returns: A python dictionary
        """
        return _asdict(self, encode_json=encode_json)
