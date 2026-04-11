#
# This code is auto-generated using the sklearn_wrapper_template.py_template template.
# Do not modify the auto-generated code(except automatic reformatting by precommit hooks).
#
import inspect
import os
from typing import Iterable, Optional, Union, List, Any, Dict, Set, Tuple
from uuid import uuid4

import cloudpickle as cp
import numpy as np
import pandas as pd
from numpy import typing as npt


import numpy
from sklearn.experimental import enable_iterative_imputer
import sklearn
import sklearn.impute
from sklearn.utils.metaestimators import available_if

from snowflake.ml.modeling.framework.base import BaseTransformer, _process_cols
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions, modeling_error_messages
from snowflake.ml._internal.env_utils import SNOWML_SPROC_ENV
from snowflake.ml._internal.utils import identifier
from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark._internal.type_utils import convert_sp_to_sf_type
from snowflake.ml.modeling._internal.model_trainer_builder import ModelTrainerBuilder
from snowflake.ml.modeling._internal.transformer_protocols import (
    BatchInferenceKwargsTypedDict,
    ScoreKwargsTypedDict
)
from snowflake.ml.model._signatures import utils as model_signature_utils
from snowflake.ml.model.model_signature import (
    BaseFeatureSpec,
    DataType,
    FeatureSpec,
    ModelSignature,
    _infer_signature,
    _rename_signature_with_snowflake_identifiers,
)

from snowflake.ml.modeling._internal.model_transformer_builder import ModelTransformerBuilder

from snowflake.ml.modeling._internal.estimator_utils import (
    gather_dependencies,
    original_estimator_has_callable,
    transform_snowml_obj_to_sklearn_obj,
    validate_sklearn_args,
)

_PROJECT = "ModelDevelopment"
# Derive subproject from module name by removing "sklearn"
# and converting module name from underscore to CamelCase
# e.g. sklearn.linear_model -> LinearModel.
_SUBPROJECT = "".join([s.capitalize() for s in "sklearn.impute".replace("sklearn.", "").split("_")])

DATAFRAME_TYPE = Union[DataFrame, pd.DataFrame]

class IterativeImputer(BaseTransformer):
    r"""Multivariate imputer that estimates each feature from all the others
    For more details on this class, see [sklearn.impute.IterativeImputer]
    (https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)

    Parameters
    ----------

    input_cols: Optional[Union[str, List[str]]]
        A string or list of strings representing column names that contain features.
        If this parameter is not specified, all columns in the input DataFrame except
        the columns specified by label_cols, sample_weight_col, and passthrough_cols
        parameters are considered input columns. Input columns can also be set after
        initialization with the `set_input_cols` method.
    
    label_cols: Optional[Union[str, List[str]]]
        A string or list of strings representing column names that contain labels.
        Label columns must be specified with this parameter during initialization
        or with the `set_label_cols` method before fitting.

    output_cols: Optional[Union[str, List[str]]]
        A string or list of strings representing column names that will store the
        output of predict and transform operations. The length of output_cols must
        match the expected number of output columns from the specific predictor or
        transformer class used.
        If you omit this parameter, output column names are derived by adding an
        OUTPUT_ prefix to the label column names for supervised estimators, or
        OUTPUT_<IDX>for unsupervised estimators. These inferred output column names
        work for predictors, but output_cols must be set explicitly for transformers.
        In general, explicitly specifying output column names is clearer, especially
        if you don’t specify the input column names.
        To transform in place, pass the same names for input_cols and output_cols.
        be set explicitly for transformers. Output columns can also be set after
        initialization with the `set_output_cols` method.

    sample_weight_col: Optional[str]
        A string representing the column name containing the sample weights.
        This argument is only required when working with weighted datasets. Sample
        weight column can also be set after initialization with the
        `set_sample_weight_col` method.

    passthrough_cols: Optional[Union[str, List[str]]]
        A string or a list of strings indicating column names to be excluded from any
        operations (such as train, transform, or inference). These specified column(s)
        will remain untouched throughout the process. This option is helpful in scenarios
        requiring automatic input_cols inference, but need to avoid using specific
        columns, like index columns, during training or inference. Passthrough columns
        can also be set after initialization with the `set_passthrough_cols` method.

    drop_input_cols: Optional[bool], default=False
        If set, the response of predict(), transform() methods will not contain input columns.

    estimator: estimator object, default=BayesianRidge()
        The estimator to use at each step of the round-robin imputation.
        If `sample_posterior=True`, the estimator must support
        `return_std` in its `predict` method.

    missing_values: int or np.nan, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    sample_posterior: bool, default=False
        Whether to sample from the (Gaussian) predictive posterior of the
        fitted estimator for each imputation. Estimator must support
        `return_std` in its `predict` method if set to `True`. Set to
        `True` if using `IterativeImputer` for multiple imputations.

    max_iter: int, default=10
        Maximum number of imputation rounds to perform before returning the
        imputations computed during the final round. A round is a single
        imputation of each feature with missing values. The stopping criterion
        is met once `max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol`,
        where `X_t` is `X` at iteration `t`. Note that early stopping is only
        applied if `sample_posterior=False`.

    tol: float, default=1e-3
        Tolerance of the stopping condition.

    n_nearest_features: int, default=None
        Number of other features to use to estimate the missing values of
        each feature column. Nearness between features is measured using
        the absolute correlation coefficient between each feature pair (after
        initial imputation). To ensure coverage of features throughout the
        imputation process, the neighbor features are not necessarily nearest,
        but are drawn with probability proportional to correlation for each
        imputed target feature. Can provide significant speed-up when the
        number of features is huge. If `None`, all features will be used.

    initial_strategy: {'mean', 'median', 'most_frequent', 'constant'},             default='mean'
        Which strategy to use to initialize the missing values. Same as the
        `strategy` parameter in :class:`~sklearn.impute.SimpleImputer`.

    fill_value: str or numerical value, default=None
        When `strategy="constant"`, `fill_value` is used to replace all
        occurrences of missing_values. For string or object data types,
        `fill_value` must be a string.
        If `None`, `fill_value` will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    imputation_order: {'ascending', 'descending', 'roman', 'arabic',             'random'}, default='ascending'
        The order in which the features will be imputed. Possible values:

        - `'ascending'`: From features with fewest missing values to most.
        - `'descending'`: From features with most missing values to fewest.
        - `'roman'`: Left to right.
        - `'arabic'`: Right to left.
        - `'random'`: A random order for each round.

    skip_complete: bool, default=False
        If `True` then features with missing values during :meth:`transform`
        which did not have any missing values during :meth:`fit` will be
        imputed with the initial imputation method only. Set to `True` if you
        have many features with no missing values at both :meth:`fit` and
        :meth:`transform` time to save compute.

    min_value: float or array-like of shape (n_features,), default=-np.inf
        Minimum possible imputed value. Broadcast to shape `(n_features,)` if
        scalar. If array-like, expects shape `(n_features,)`, one min value for
        each feature. The default is `-np.inf`.

    max_value: float or array-like of shape (n_features,), default=np.inf
        Maximum possible imputed value. Broadcast to shape `(n_features,)` if
        scalar. If array-like, expects shape `(n_features,)`, one max value for
        each feature. The default is `np.inf`.

    verbose: int, default=0
        Verbosity flag, controls the debug messages that are issued
        as functions are evaluated. The higher, the more verbose. Can be 0, 1,
        or 2.

    random_state: int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use. Randomizes
        selection of estimator features if `n_nearest_features` is not `None`,
        the `imputation_order` if `random`, and the sampling from posterior if
        `sample_posterior=True`. Use an integer for determinism.
        See :term:`the Glossary <random_state>`.

    add_indicator: bool, default=False
        If `True`, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    keep_empty_features: bool, default=False
        If True, features that consist exclusively of missing values when
        `fit` is called are returned in results when `transform` is called.
        The imputed value is always `0` except when
        `initial_strategy="constant"` in which case `fill_value` will be
        used instead.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        *,
        estimator=None,
        missing_values=np.nan,
        sample_posterior=False,
        max_iter=10,
        tol=0.001,
        n_nearest_features=None,
        initial_strategy="mean",
        fill_value=None,
        imputation_order="ascending",
        skip_complete=False,
        min_value=-np.inf,
        max_value=np.inf,
        verbose=0,
        random_state=None,
        add_indicator=False,
        keep_empty_features=False,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        label_cols: Optional[Union[str, Iterable[str]]] = None,
        passthrough_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
        sample_weight_col: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)
        self.set_label_cols(label_cols)
        self.set_passthrough_cols(passthrough_cols)
        self.set_drop_input_cols(drop_input_cols)
        self.set_sample_weight_col(sample_weight_col)
        self._use_external_memory_version = False
        self._batch_size = -1        
        deps: Set[str] = set([f'numpy=={np.__version__}', f'scikit-learn=={sklearn.__version__}', f'cloudpickle=={cp.__version__}'])
        deps = deps | gather_dependencies(estimator)
        self._deps = list(deps)
        estimator = transform_snowml_obj_to_sklearn_obj(estimator)
        init_args = {'estimator':(estimator, None, False),
            'missing_values':(missing_values, np.nan, False),
            'sample_posterior':(sample_posterior, False, False),
            'max_iter':(max_iter, 10, False),
            'tol':(tol, 0.001, False),
            'n_nearest_features':(n_nearest_features, None, False),
            'initial_strategy':(initial_strategy, "mean", False),
            'fill_value':(fill_value, None, False),
            'imputation_order':(imputation_order, "ascending", False),
            'skip_complete':(skip_complete, False, False),
            'min_value':(min_value, -np.inf, False),
            'max_value':(max_value, np.inf, False),
            'verbose':(verbose, 0, False),
            'random_state':(random_state, None, False),
            'add_indicator':(add_indicator, False, False),
            'keep_empty_features':(keep_empty_features, False, False),}
        cleaned_up_init_args = validate_sklearn_args(
            args=init_args,
            klass=sklearn.impute.IterativeImputer
        )
        self._sklearn_object: Any = sklearn.impute.IterativeImputer(
            **cleaned_up_init_args,
        )
        self._model_signature_dict: Optional[Dict[str, ModelSignature]] = None
        # If user used snowpark dataframe during fit, here it stores the snowpark input_cols, otherwise the processed input_cols
        self._snowpark_cols: Optional[List[str]] = self.input_cols
        self._autogenerated = True
        self._class_name=IterativeImputer.__class__.__name__
        self._subproject = _SUBPROJECT


    def _get_rand_id(self) -> str:
        """
        Generate random id to be used in sproc and stage names.

        Returns:
            Random id string usable in sproc, table, and stage names.
        """
        return str(uuid4()).replace("-", "_").upper()

    def set_input_cols(self, input_cols: Optional[Union[str, Iterable[str]]]) -> "IterativeImputer":
        """
        Input columns setter.

        Args:
            input_cols: A single input column or multiple input columns.

        Returns:
            self
        """
        self.input_cols = _process_cols(input_cols)
        self._snowpark_cols = self.input_cols
        return self

    def _get_active_columns(self) -> List[str]:
        """"Get the list of columns that are relevant to the transformer."""
        selected_cols = (
            self.input_cols +
            self.label_cols +
            ([self.sample_weight_col] if self.sample_weight_col is not None else [])
        )
        return selected_cols

    def _fit(self, dataset: Union[DataFrame, pd.DataFrame]) -> "IterativeImputer":
        """Fit the imputer on `X` and return self
        For more details on this function, see [sklearn.impute.IterativeImputer.fit]
        (https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer.fit)


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.

        Returns:
            self
        """
        self._infer_input_output_cols(dataset)
        if isinstance(dataset, DataFrame):
            session = dataset._session
            assert session is not None  # keep mypy happy
            # Specify input columns so column pruning will be enforced
            selected_cols = self._get_active_columns()
            if len(selected_cols) > 0:
                dataset = dataset.select(selected_cols)

            self._snowpark_cols = dataset.select(self.input_cols).columns

            # If we are already in a stored procedure, no need to kick off another one.
            if SNOWML_SPROC_ENV in os.environ:
                statement_params = telemetry.get_function_usage_statement_params(
                    project=_PROJECT,
                    subproject=_SUBPROJECT,
                    function_name=telemetry.get_statement_params_full_func_name(
                        inspect.currentframe(), IterativeImputer.__class__.__name__
                    ),
                    api_calls=[Session.call],
                    custom_tags={"autogen": True} if self._autogenerated else None,
                )
                pd_df: pd.DataFrame = dataset.to_pandas(statement_params=statement_params)
                pd_df.columns = dataset.columns
                dataset = pd_df

        model_trainer = ModelTrainerBuilder.build(
            estimator=self._sklearn_object,
            dataset=dataset,
            input_cols=self.input_cols,
            label_cols=self.label_cols,
            sample_weight_col=self.sample_weight_col,
            autogenerated=self._autogenerated,
            subproject=_SUBPROJECT,
            use_external_memory_version=self._use_external_memory_version,
            batch_size=self._batch_size,
        )
        self._sklearn_object = model_trainer.train()
        self._is_fitted = True
        self._generate_model_signatures(dataset)
        return self

    def _batch_inference_validate_snowpark(
        self,
        dataset: DataFrame,
        inference_method: str,
    ) -> None:
        """Util method to run validate that batch inference can be run on a snowpark dataframe.

        Args:
            dataset: snowpark dataframe
            inference_method: the inference method such as predict, score...
            
        Raises:
            SnowflakeMLException: If the estimator is not fitted, raise error
            SnowflakeMLException: If the session is None, raise error

        """
        if not self._is_fitted:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.METHOD_NOT_ALLOWED,
                original_exception=RuntimeError(
                    f"Estimator {self.__class__.__name__} not fitted before calling {inference_method} method."
                ),
            )

        session = dataset._session
        if session is None:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(
                    "Session must not specified for snowpark dataset."
                ),
            )


    @available_if(original_estimator_has_callable("predict"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    def predict(self, dataset: Union[DataFrame, pd.DataFrame]) -> Union[DataFrame, pd.DataFrame]:
        """Method not supported for this class.


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.

        Returns:
            Transformed dataset.
        """
        super()._check_dataset_type(dataset)
        inference_method = "predict"

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used. 
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()   
                                
        if isinstance(dataset, DataFrame):
            expected_type_inferred = ""
            # when it is classifier, infer the datatype from label columns
            if expected_type_inferred == "" and 'predict' in self.model_signatures:
                # Batch inference takes a single expected output column type. Use the first columns type for now.
                label_cols_signatures = [
                    row for row in self.model_signatures['predict'].outputs if row.name in self.output_cols
                ]
                if len(label_cols_signatures) == 0:
                    error_str = f"Output columns {self.output_cols} do not match model signatures {self.model_signatures['predict'].outputs}."
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_ATTRIBUTE,
                        original_exception=ValueError(error_str),
                    )

                expected_type_inferred = convert_sp_to_sf_type(label_cols_signatures[0].as_snowpark_type())
            
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()
            assert isinstance(
                dataset._session, Session
            )  # mypy does not recognize the check in _batch_inference_validate_snowpark()

            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_type=expected_type_inferred,
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols= self.output_cols,
            **transform_kwargs
        )

        return output_df

    @available_if(original_estimator_has_callable("transform"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    def transform(self, dataset: Union[DataFrame, pd.DataFrame]) -> Union[DataFrame, pd.DataFrame]:
        """Impute all missing values in `X`
        For more details on this function, see [sklearn.impute.IterativeImputer.transform]
        (https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer.transform)


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.

        Returns:
            Transformed dataset.
        """
        super()._check_dataset_type(dataset)
        inference_method = "transform"

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used. 
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()
        if isinstance(dataset, DataFrame):
            expected_dtype = ""
            if False:  # is child of _BaseHeterogeneousEnsemble
                # transform() method of HeterogeneousEnsemble estimators return responses of varying shapes
                # from (n_samples, n_estimators) to (n_samples, n_estimators * n_classes) (and everything in between)
                # based on init param values. We will convert that to pandas dataframe of shape (n_samples, 1) with
                # each row containing a list of values.
                expected_dtype = "array"

            # If we were unable to assign a type to this transform in the factory, infer the type here.
            if expected_dtype == "":
                # If this is a clustering transformer, if the number of output columns does not equal the number of clusters the response will be an "array"
                if hasattr(self._sklearn_object, "n_clusters") and getattr(self._sklearn_object, "n_clusters") != len(self.output_cols):
                    expected_dtype = "array"
                # If this is a decomposition transformer, if the number of output columns does not equal the number of components the response will be an "array"
                elif hasattr(self._sklearn_object, "n_components") and getattr(self._sklearn_object, "n_components") != len(self.output_cols):
                    expected_dtype = "array"
                else:
                    output_types = [signature.as_snowpark_type() for signature in _infer_signature(dataset[self.input_cols], "output", use_snowflake_identifiers=True)]
                    # We can only infer the output types from the input types if the following two statemetns are true:
                    # 1) All of the output types are the same. Otherwise, we still have to fall back to variant because `_sklearn_inference` only accepts one type.
                    # 2) The length of the input columns equals the length of the output columns. Otherwise the transform will likely result in an `ARRAY`.
                    if all(x == output_types[0] for x in output_types) and len(output_types) == len(self.output_cols):
                        expected_dtype = convert_sp_to_sf_type(output_types[0])
            
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()
            assert isinstance(dataset._session, Session) # mypy does not recognize the check in _batch_inference_validate_snowpark()

            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_type=expected_dtype,
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=self.output_cols,
            **transform_kwargs
        )
        return output_df
    
    @available_if(original_estimator_has_callable("fit_predict"))  # type: ignore[misc]
    def fit_predict(
        self,
        dataset: Union[DataFrame, pd.DataFrame],
        output_cols_prefix: str = "fit_predict_",
    ) -> Union[DataFrame, pd.DataFrame]:
        """ Method not supported for this class.


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.
        output_cols_prefix: Prefix for the response columns
        Returns:
            Predicted dataset.
        """
        self._infer_input_output_cols(dataset)
        super()._check_dataset_type(dataset)
        model_trainer = ModelTrainerBuilder.build_fit_predict(
            estimator=self._sklearn_object,
            dataset=dataset,
            input_cols=self.input_cols,
            autogenerated=self._autogenerated,
            subproject=_SUBPROJECT,
        )
        expected_output_cols = (
            self.output_cols if self.output_cols else self._get_output_column_names(output_cols_prefix)
        )
        if isinstance(dataset, DataFrame):
            expected_output_cols, example_output_pd_df = self._align_expected_output(
                "fit_predict", dataset, expected_output_cols, output_cols_prefix
            )
            output_result, fitted_estimator = model_trainer.train_fit_predict(
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_list=expected_output_cols,
                example_output_pd_df=example_output_pd_df,
            )
        else:
            output_result, fitted_estimator = model_trainer.train_fit_predict(
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_list=expected_output_cols,
            )
        self._sklearn_object = fitted_estimator
        self._is_fitted = True
        return output_result

            
    @available_if(original_estimator_has_callable("fit_transform"))  # type: ignore[misc]
    def fit_transform(self, dataset: Union[DataFrame, pd.DataFrame], output_cols_prefix: str = "fit_transform_",) -> Union[DataFrame, pd.DataFrame]:
        """ Fit the imputer on `X` and return the transformed `X`
        For more details on this function, see [sklearn.impute.IterativeImputer.fit_transform]
        (https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer.fit_transform)


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.
        output_cols_prefix: Prefix for the response columns
        Returns:
            Transformed dataset.
        """
        self._infer_input_output_cols(dataset)
        super()._check_dataset_type(dataset)

        model_trainer = ModelTrainerBuilder.build_fit_transform(
            estimator=self._sklearn_object,
            dataset=dataset,
            input_cols=self.input_cols,
            label_cols=self.label_cols,
            sample_weight_col=self.sample_weight_col,
            autogenerated=self._autogenerated,
            subproject=_SUBPROJECT,
        )
        output_result, fitted_estimator = model_trainer.train_fit_transform(
            drop_input_cols=self._drop_input_cols,
            expected_output_cols_list=self.output_cols,
        )
        self._sklearn_object = fitted_estimator
        self._is_fitted = True
        return output_result


    def _get_output_column_names(self, output_cols_prefix: str, output_cols: Optional[List[str]] = None) -> List[str]:
        """ Returns the list of output columns for predict_proba(), decision_function(), etc.. functions.
        Returns a list with output_cols_prefix as the only element if the estimator is not a classifier.
        """
        output_cols_prefix = identifier.resolve_identifier(output_cols_prefix)
        # The following condition is introduced for kneighbors methods, and not used in other methods
        if output_cols:
            output_cols = [
                identifier.concat_names([output_cols_prefix, identifier.resolve_identifier(c)])
                for c in output_cols
            ]
        elif getattr(self._sklearn_object, "classes_", None) is None:
            output_cols = [output_cols_prefix]
        elif self._sklearn_object is not None:
            classes = self._sklearn_object.classes_
            if isinstance(classes, numpy.ndarray):
                output_cols = [f'{output_cols_prefix}{str(c)}' for c in classes.tolist()]
            elif isinstance(classes, list) and len(classes) > 0 and isinstance(classes[0], numpy.ndarray):
                # If the estimator is a multioutput estimator, classes_ will be a list of ndarrays.
                output_cols = []
                for i, cl in enumerate(classes):
                    # For binary classification, there is only one output column for each class
                    # ndarray as the two classes are complementary.
                    if len(cl) == 2:
                        output_cols.append(f'{output_cols_prefix}{i}_{cl[0]}')
                    else:
                        output_cols.extend([
                            f'{output_cols_prefix}{i}_{c}' for c in cl.tolist()
                        ])
        else:
            output_cols = []

        # Make sure column names are valid snowflake identifiers.
        assert output_cols is not None  # Make MyPy happy
        rv = [identifier.rename_to_valid_snowflake_identifier(c) for c in output_cols]

        return rv

    def _align_expected_output(
        self, method: str, dataset: DataFrame, expected_output_cols_list: List[str], output_cols_prefix: str,
    ) -> Tuple[List[str], pd.DataFrame]:
        """ Run 1 line of data with the desired method, and return one tuple that consists of the output column names 
        and output dataframe with 1 line.
        If the method is fit_predict, run 2 lines of data. 
        """
        # in case the inferred output column names dimension is different 
        # we use one line of snowpark dataframe and put it into sklearn estimator using pandas

        # For fit_predict method, a minimum of 2 is required by MinCovDet, BayesianGaussianMixture
        # so change the minimum of number of rows to 2
        num_examples = 2
        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=_SUBPROJECT,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), IterativeImputer.__class__.__name__
            ),
            api_calls=[Session.call],
            custom_tags={"autogen": True} if self._autogenerated else None,
        )
        if output_cols_prefix == "fit_predict_":
            if hasattr(self._sklearn_object, "n_clusters"):
                # cluster classes such as BisectingKMeansTest requires # of examples >= n_clusters
                num_examples = self._sklearn_object.n_clusters
            elif hasattr(self._sklearn_object, "min_samples"):
                # OPTICS default min_samples 5, which requires at least 5 lines of data
                num_examples = self._sklearn_object.min_samples
            elif hasattr(self._sklearn_object, "n_neighbors") and hasattr(self._sklearn_object, "n_samples"):
                # LocalOutlierFactor expects n_neighbors <= n_samples
                num_examples = self._sklearn_object.n_neighbors
            sample_pd_df = dataset.select(self.input_cols).limit(num_examples).to_pandas(statement_params=statement_params)
        else:
            sample_pd_df = dataset.select(self.input_cols).limit(1).to_pandas(statement_params=statement_params)

        # Rename the pandas df column names to snowflake identifiers and reorder columns to match the order
        # seen during the fit.
        snowpark_column_names = dataset.select(self.input_cols).columns
        sample_pd_df.columns = snowpark_column_names

        output_df_pd = getattr(self, method)(sample_pd_df, output_cols_prefix)
        output_df_columns = list(output_df_pd.columns)
        output_df_columns_set: Set[str] = set(output_df_columns) - set(dataset.columns)
        if self.sample_weight_col:
            output_df_columns_set -= set(self.sample_weight_col)
        
        # if the dimension of inferred output column names is correct; use it
        if len(expected_output_cols_list) == len(output_df_columns_set):
            return expected_output_cols_list, output_df_pd
        # otherwise, use the sklearn estimator's output
        else:
            expected_output_cols_list = sorted(list(output_df_columns_set), key=lambda x: output_df_columns.index(x))
            return expected_output_cols_list, output_df_pd[expected_output_cols_list]

    @available_if(original_estimator_has_callable("predict_proba"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    def predict_proba(
        self, dataset: Union[DataFrame, pd.DataFrame], output_cols_prefix: str = "predict_proba_"
    ) -> Union[DataFrame, pd.DataFrame]:
        """Method not supported for this class.


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.
            output_cols_prefix: Prefix for the response columns

        Returns:
            Output dataset with probability of the sample for each class in the model.
        """
        super()._check_dataset_type(dataset)
        inference_method = "predict_proba"

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used. 
        transform_kwargs: BatchInferenceKwargsTypedDict = dict() 

        expected_output_cols = self._get_output_column_names(output_cols_prefix)

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()
            assert isinstance(
                dataset._session, Session
            )  # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_type="float",
            )
            expected_output_cols, _ = self._align_expected_output(
                inference_method, dataset, expected_output_cols, output_cols_prefix
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated
        )
        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=expected_output_cols,
            **transform_kwargs
        )
        return output_df

    @available_if(original_estimator_has_callable("predict_log_proba"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    def predict_log_proba(
        self, dataset: Union[DataFrame, pd.DataFrame], output_cols_prefix: str = "predict_log_proba_"
    ) -> Union[DataFrame, pd.DataFrame]:
        """Method not supported for this class.


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.
            output_cols_prefix: str
                Prefix for the response columns

        Returns:
            Output dataset with log probability of the sample for each class in the model.
        """
        super()._check_dataset_type(dataset)
        inference_method = "predict_log_proba"
        expected_output_cols = self._get_output_column_names(output_cols_prefix)

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used. 
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()  

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()
            assert isinstance(
                dataset._session, Session
            )  # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_type="float",
            )
            expected_output_cols, _ = self._align_expected_output(
                inference_method, dataset, expected_output_cols, output_cols_prefix
            )
        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=expected_output_cols,
            **transform_kwargs
        )
        return output_df


    @available_if(original_estimator_has_callable("decision_function"))  # type: ignore[misc]
    def decision_function(
        self, dataset: Union[DataFrame, pd.DataFrame], output_cols_prefix: str = "decision_function_"
    ) -> Union[DataFrame, pd.DataFrame]:
        """Method not supported for this class.


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.
            output_cols_prefix: str
                Prefix for the response columns

        Returns:
            Output dataset with results of the decision function for the samples in input dataset.
        """
        super()._check_dataset_type(dataset)
        inference_method = "decision_function"

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used. 
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()  

        expected_output_cols = self._get_output_column_names(output_cols_prefix)

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()
            assert isinstance(
                dataset._session, Session
            )  # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols=self._drop_input_cols,
                expected_output_cols_type="float",
            )
            expected_output_cols, _ = self._align_expected_output(
                inference_method, dataset, expected_output_cols, output_cols_prefix
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(snowpark_input_cols=self._snowpark_cols, drop_input_cols=self._drop_input_cols)

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=expected_output_cols,
            **transform_kwargs
        )
        return output_df

    @available_if(original_estimator_has_callable("score_samples"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    def score_samples(
        self, dataset: Union[DataFrame, pd.DataFrame], output_cols_prefix: str = "score_samples_"
    ) -> Union[DataFrame, pd.DataFrame]:
        """Method not supported for this class.


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.
            output_cols_prefix: Prefix for the response columns

        Returns:
            Output dataset with probability of the sample for each class in the model.
        """
        super()._check_dataset_type(dataset)
        inference_method = "score_samples"

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used. 
        transform_kwargs: BatchInferenceKwargsTypedDict = dict()

        expected_output_cols = self._get_output_column_names(output_cols_prefix)

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()
            assert isinstance(dataset._session, Session) # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                drop_input_cols = self._drop_input_cols,
                expected_output_cols_type="float",
            )
            expected_output_cols, _ = self._align_expected_output(
                inference_method, dataset, expected_output_cols, output_cols_prefix
            )

        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(
                snowpark_input_cols = self._snowpark_cols,
                drop_input_cols = self._drop_input_cols
                )

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=expected_output_cols,
            **transform_kwargs
        )
        return output_df

    @available_if(original_estimator_has_callable("score"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    def score(self, dataset: Union[DataFrame, pd.DataFrame]) -> float:
        """Method not supported for this class.


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.

        Returns:
            Score.
        """
        self._infer_input_output_cols(dataset)
        super()._check_dataset_type(dataset)

        # This dictionary contains optional kwargs for scoring. These kwargs
        # are specific to the type of dataset used. 
        transform_kwargs: ScoreKwargsTypedDict = dict()  

        if isinstance(dataset, DataFrame):
            self._batch_inference_validate_snowpark(dataset=dataset, inference_method="score")
            self._deps = self._get_dependencies()
            selected_cols = self._get_active_columns()
            if len(selected_cols) > 0:
                dataset = dataset.select(selected_cols)
            assert isinstance(dataset._session, Session) # keep mypy happy
            transform_kwargs = dict(
                session=dataset._session,
                dependencies=self._deps,
                score_sproc_imports=['sklearn'],
            )
        elif isinstance(dataset, pd.DataFrame):
            # pandas_handler.score() does not require any extra kwargs.
            transform_kwargs = dict()

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated
        )

        output_score = transform_handlers.score(
            input_cols=self.input_cols,
            label_cols=self.label_cols,
            sample_weight_col=self.sample_weight_col,
            **transform_kwargs
        )

        return output_score


    @available_if(original_estimator_has_callable("kneighbors"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        custom_tags=dict([("autogen", True)]),
    )
    def kneighbors(
        self,
        dataset: Union[DataFrame, pd.DataFrame],
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
        output_cols_prefix: str = "kneighbors_",
    ) -> Union[DataFrame, pd.DataFrame]:
        """Method not supported for this class.


        Raises:
            TypeError: Supported dataset types: snowpark.DataFrame, pandas.DataFrame.

        Args:
            dataset: Union[snowflake.snowpark.DataFrame, pandas.DataFrame]
                Snowpark or Pandas DataFrame.
            output_cols_prefix: str
                Prefix for the response columns

        Returns:
            Output dataset with results of the K-neighbors for the samples in input dataset.
        """
        super()._check_dataset_type(dataset)
        inference_method="kneighbors"

        # This dictionary contains optional kwargs for batch inference. These kwargs
        # are specific to the type of dataset used. 
        transform_kwargs: BatchInferenceKwargsTypedDict = dict() 
        output_cols = ["neigh_ind"]
        if return_distance:
            output_cols.insert(0, "neigh_dist")

        if isinstance(dataset, DataFrame):

            self._batch_inference_validate_snowpark(dataset=dataset, inference_method=inference_method)
            self._deps = self._get_dependencies()
            assert isinstance(dataset._session, Session) # mypy does not recognize the check in _batch_inference_validate_snowpark()
            transform_kwargs = dict(
                session = dataset._session,
                dependencies = self._deps,
                drop_input_cols = self._drop_input_cols,
                expected_output_cols_type="array",
                n_neighbors = n_neighbors,
                return_distance =  return_distance
            )
        elif isinstance(dataset, pd.DataFrame):
            transform_kwargs = dict(
                n_neighbors = n_neighbors,
                return_distance = return_distance,
                snowpark_input_cols = self._snowpark_cols
            )

        transform_handlers = ModelTransformerBuilder.build(
            dataset=dataset,
            estimator=self._sklearn_object,
            class_name=self._class_name,
            subproject=self._subproject,
            autogenerated=self._autogenerated
        )

        output_df: DATAFRAME_TYPE = transform_handlers.batch_inference(
            inference_method=inference_method,
            input_cols=self.input_cols,
            expected_output_cols=self._get_output_column_names(output_cols_prefix, output_cols),
            **transform_kwargs
        )
        return output_df

    

    def to_sklearn(self) -> Any:
        """Get sklearn.impute.IterativeImputer object.
        """
        if self._sklearn_object is None:
            self._sklearn_object = self._create_sklearn_object()
        return self._sklearn_object

    def to_xgboost(self) -> Any:
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.METHOD_NOT_ALLOWED,
            original_exception=AttributeError(
                modeling_error_messages.UNSUPPORTED_MODEL_CONVERSION.format(
                    "to_xgboost()", 
                    "to_sklearn()"
                )
            ),
        )

    def to_lightgbm(self) -> Any:
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.METHOD_NOT_ALLOWED,
            original_exception=AttributeError(
                modeling_error_messages.UNSUPPORTED_MODEL_CONVERSION.format(
                    "to_lightgbm()", 
                    "to_sklearn()"
                )
            ),
        )

    def _get_dependencies(self) -> List[str]:
        return self._deps


    def _generate_model_signatures(self, dataset: Union[DataFrame, pd.DataFrame]) -> None:
        self._model_signature_dict = dict()

        PROB_FUNCTIONS = ["predict_log_proba", "predict_proba", "decision_function"]

        inputs = list(_infer_signature(dataset[self.input_cols], "input", use_snowflake_identifiers=True))
        outputs: List[BaseFeatureSpec] = []
        if hasattr(self, "predict"):
            # keep mypy happy
            assert self._sklearn_object is not None and hasattr(self._sklearn_object, "_estimator_type")
            # For classifier, the type of predict is the same as the type of label
            if self._sklearn_object._estimator_type == "classifier":
                # label columns is the desired type for output
                outputs = list(_infer_signature(dataset[self.label_cols], "output", use_snowflake_identifiers=True))
                # rename the output columns
                outputs = list(model_signature_utils.rename_features(outputs, self.output_cols))
                self._model_signature_dict["predict"] = ModelSignature(
                    inputs, ([] if self._drop_input_cols else inputs) + outputs
                )
            # For mixture models that use the density mixin, `predict` returns the argmax of the log prob.
            # For outlier models, returns -1 for outliers and 1 for inliers.
            # Clusterer returns int64 cluster labels.
            elif self._sklearn_object._estimator_type in ["DensityEstimator", "clusterer", "outlier_detector"]:
                outputs = [FeatureSpec(dtype=DataType.INT64, name=c) for c in self.output_cols]
                self._model_signature_dict["predict"] = ModelSignature(
                    inputs, ([] if self._drop_input_cols else inputs) + outputs
                )

            # For regressor, the type of predict is float64
            elif self._sklearn_object._estimator_type == "regressor":
                outputs = [FeatureSpec(dtype=DataType.DOUBLE, name=c) for c in self.output_cols]
                self._model_signature_dict["predict"] = ModelSignature(
                    inputs, ([] if self._drop_input_cols else inputs) + outputs
                )

        for prob_func in PROB_FUNCTIONS:
            if hasattr(self, prob_func):
                output_cols_prefix: str = f"{prob_func}_"
                output_column_names = self._get_output_column_names(output_cols_prefix)
                outputs = [FeatureSpec(dtype=DataType.DOUBLE, name=c) for c in output_column_names]
                self._model_signature_dict[prob_func] = ModelSignature(
                    inputs, ([] if self._drop_input_cols else inputs) + outputs
                )

        # Output signature names may still need to be renamed, since they were not created with `_infer_signature`.
        items = list(self._model_signature_dict.items())
        for method, signature in items:
            signature._outputs = _rename_signature_with_snowflake_identifiers(signature._outputs)
            self._model_signature_dict[method] = signature

    @property
    def model_signatures(self) -> Dict[str, ModelSignature]:
        """Returns model signature of current class.

        Raises:
            SnowflakeMLException: If estimator is not fitted, then model signature cannot be inferred

        Returns:
            Dict with each method and its input output signature
        """
        if self._model_signature_dict is None:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=RuntimeError("Estimator not fitted before accessing property model_signatures!"),
            )
        return self._model_signature_dict
