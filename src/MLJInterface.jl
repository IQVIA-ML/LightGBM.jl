module MLJInterface

import MLJModelInterface
import LightGBM


const LGBM_METRICS = (
    "None", "l1", "l2", "rmse", "quantile", "mape", "huber", "fair", "poisson", "gamma", "gamma_deviance",
    "tweedie", "ndcg", "lambdarank", "map", "mean_average_precision", "auc", "average_precision", "binary_logloss",
    "binary_error", "auc_mu", "multi_logloss", "multi_error", "cross_entropy", "xentropy", "multi_logloss","cross_entropy_lambda",
    "xentlambda", "kullback_leibler", "kldiv",
)

const CLASSIFICATION_OBJECTIVES = (
    "binary", "multiclass", "softmax",
)

const REGRESSION_OBJECTIVES = (
    "regression", "regression_l2", "l2", "mean_squared_error", "mse", "l2_root", "root_mean_squared_error", "rmse",
    "regression_l1", "l1", "mean_absolute_error", "mae", "huber", "fair", "poisson", "quantile", "mape",
    "mean_absolute_percentage_error", "gamma", "tweedie",
)

const NON_LIGHTGBM_PARAMETERS = (
    :truncate_booster,
)

MLJModelInterface.@mlj_model mutable struct LGBMRegressor <: MLJModelInterface.Deterministic

    # Hyperparameters, see https://lightgbm.readthedocs.io/en/latest/Parameters.html for defaults
    boosting::String = "gbdt"::(_ in ("gbdt", "goss", "rf", "dart"))
    num_iterations::Int = 100::(_ >= 0)
    learning_rate::Float64 = 0.1::(_ > 0.)
    num_leaves::Int = 31::(1 < _ <= 131072)
    max_depth::Int = -1;#::(_ != 0);
    tree_learner::String = "serial"::(_ in ("serial", "feature", "data", "voting"))
    histogram_pool_size::Float64 = -1.0;#::(_ != 0.0);
    min_data_in_leaf::Int = 20::(_ >= 0)
    min_sum_hessian_in_leaf::Float64 = 1e-3::(_ >= 0.0)
    max_delta_step::Float64 = 0.0
    lambda_l1::Float64 = 0.0::(_ >= 0.0)
    lambda_l2::Float64 = 0.0::(_ >= 0.0)
    min_gain_to_split::Float64 = 0.0::(_ >= 0.0)
    feature_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    feature_fraction_bynode::Float64 = 1.0::(0.0 < _ <= 1.0)
    feature_fraction_seed::Int = 2
    bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    bagging_freq::Int = 0::(_ >= 0)
    bagging_seed::Int = 3
    early_stopping_round::Int = 0
    extra_trees::Bool = false
    extra_seed::Int = 6
    max_bin::Int = 255::(_ > 1)
    bin_construct_sample_cnt::Int = 200000::(_ > 0)
    drop_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)
    max_drop::Int = 50
    skip_drop:: Float64 = 0.5::(0.0 <= _ <= 1)
    xgboost_dart_mode::Bool = false
    uniform_drop::Bool = false
    drop_seed::Int = 4
    top_rate::Float64 = 0.2::(0.0 <= _ <= 1.0)
    other_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)
    min_data_per_group::Int = 100::(_ > 0)
    max_cat_threshold::Int = 32::(_ > 0)
    cat_l2::Float64 = 10.0::(_ >= 0)
    cat_smooth::Float64 = 10.0::(_ >= 0)

    # Model properties
    objective::String = "regression"::(_ in REGRESSION_OBJECTIVES)
    categorical_feature::Vector{Int} = Vector{Int}()
    data_random_seed::Int = 1
    is_enable_sparse::Bool = true
    is_unbalance::Bool = false
    boost_from_average::Bool = true
    use_missing::Bool = true
    linear_tree::Bool = false
    feature_pre_filter::Bool = true

    alpha::Float64 = 0.9::(_ > 0.0 )

    # Metrics
    metric::Vector{String} = ["l2"]::(all(in.(_, (LGBM_METRICS, ))))
    metric_freq::Int = 1::(_ > 0)
    is_provide_training_metric::Bool = false
    eval_at::Vector{Int} = Vector{Int}([1, 2, 3, 4, 5])::(all(_ .> 0))

    # Implementation parameters
    num_machines::Int = 1::(_ > 0)
    num_threads::Int  = 0::(_ >= 0)
    local_listen_port::Int = 12400::(_ > 0)
    time_out::Int = 120::(_ > 0)
    machine_list_filename::String = ""
    save_binary::Bool = false
    device_type::String = "cpu"::(_ in ("cpu", "gpu"))
    gpu_use_dp::Bool = false
    gpu_platform_id::Int = -1
    gpu_device_id::Int = -1
    num_gpu::Int = 1
    force_col_wise::Bool = false
    force_row_wise::Bool = false
    truncate_booster::Bool = true

end


MLJModelInterface.@mlj_model mutable struct LGBMClassifier <: MLJModelInterface.Probabilistic

    # Hyperparameters, see https://lightgbm.readthedocs.io/en/latest/Parameters.html for defaults
    boosting::String = "gbdt"::(_ in ("gbdt", "goss", "rf", "dart"))
    num_iterations::Int = 100::(_ >= 0)
    learning_rate::Float64 = 0.1::(_ > 0.)
    num_leaves::Int = 31::(1 < _ <= 131072)
    max_depth::Int = -1;#::(_ != 0);
    tree_learner::String = "serial"::(_ in ("serial", "feature", "data", "voting"))
    histogram_pool_size::Float64 = -1.0;#::(_ != 0.0);
    min_data_in_leaf::Int = 20::(_ >= 0)
    min_sum_hessian_in_leaf::Float64 = 1e-3::(_ >= 0.0)
    max_delta_step::Float64 = 0.0
    lambda_l1::Float64 = 0.0::(_ >= 0.0)
    lambda_l2::Float64 = 0.0::(_ >= 0.0)
    min_gain_to_split::Float64 = 0.0::(_ >= 0.0)
    feature_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    feature_fraction_bynode::Float64 = 1.0::(0.0 < _ <= 1.0)
    feature_fraction_seed::Int = 2
    bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    pos_bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    neg_bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    bagging_freq::Int = 0::(_ >= 0)
    bagging_seed::Int = 3
    early_stopping_round::Int = 0
    extra_trees::Bool = false
    extra_seed::Int = 6
    max_bin::Int = 255::(_ > 1)
    bin_construct_sample_cnt::Int = 200000::(_ > 0)
    drop_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)
    max_drop::Int = 50
    skip_drop:: Float64 = 0.5::(0.0 <= _ <= 1)
    xgboost_dart_mode::Bool = false
    uniform_drop::Bool = false
    drop_seed::Int = 4
    top_rate::Float64 = 0.2::(0.0 <= _ <= 1.0)
    other_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)
    min_data_per_group::Int = 100::(_ > 0)
    max_cat_threshold::Int = 32::(_ > 0)
    cat_l2::Float64 = 10.0::(_ >= 0)
    cat_smooth::Float64 = 10.0::(_ >= 0)

    # For documentation purposes: A calibration scaling factor for the output probabilities for binary and multiclass OVA
    sigmoid::Float64 = 1.0::(_ > 0.0 )

    # Model properties
    objective::String = "multiclass"::(_ in CLASSIFICATION_OBJECTIVES)
    categorical_feature::Vector{Int} = Vector{Int}();
    data_random_seed::Int = 1
    is_enable_sparse::Bool = true
    is_unbalance::Bool = false
    boost_from_average::Bool = true
    scale_pos_weight = 1.0
    use_missing::Bool = true
    linear_tree::Bool = false
    feature_pre_filter::Bool = true

    # Metrics
    metric::Vector{String} = ["None"]::(all(in.(_, (LGBM_METRICS, ))))
    metric_freq::Int = 1::(_ > 0)
    is_provide_training_metric::Bool = false
    eval_at::Vector{Int} = Vector{Int}([1, 2, 3, 4, 5])::(all(_ .> 0))

    # Implementation parameters
    num_machines::Int = 1::(_ > 0)
    num_threads::Int  = 0::(_ >= 0)
    local_listen_port::Int = 12400::(_ > 0)
    time_out::Int = 120::(_ > 0)
    machine_list_filename::String = ""
    save_binary::Bool = false
    device_type::String = "cpu"::(_ in ("cpu", "gpu"))
    gpu_use_dp::Bool = false
    gpu_platform_id::Int = -1
    gpu_device_id::Int = -1
    num_gpu::Int = 1
    force_col_wise::Bool = false
    force_row_wise::Bool = false
    truncate_booster::Bool = true

end


MODELS = Union{LGBMClassifier, LGBMRegressor}


function mlj_to_kwargs(model::MODELS)

    return Dict{Symbol, Any}(
        name => getfield(model, name)
        for name in fieldnames(typeof(model))
        if !(name in NON_LIGHTGBM_PARAMETERS)
    )

end


function mlj_to_kwargs(model::LGBMClassifier, classes)

    num_class = length(classes)
    if model.objective == "binary"
        if num_class != 2
            throw(ArgumentError("binary objective number of classes $num_class greater than 2"))
        end
        # munge num_class for LightGBM
        num_class = 1
    end

    retval = mlj_to_kwargs(model)
    retval[:num_class] = num_class
    return retval

end


# X and y and w must be untyped per MLJ docs
function fit(mlj_model::MODELS, verbosity::Int, X, y, w=AbstractFloat[])

    # MLJ docs are clear that 0 means silent. but 0 in LightGBM world means "warnings"
    # and < 0 means fatal logs only, so we put intended silence to -1 (which is probably the closest we get)
    verbosity = if verbosity == 0; -1 else verbosity end

    y_lgbm, classes = prepare_targets(y, mlj_model)
    model = model_init(mlj_model, classes)
    X = MLJModelInterface.matrix(X)
    # The FFI wrapper wants Float32 for these
    w = Float32.(w)
    # slice of y_lgbm required to converts it from a SubArray to a copy of an actual Array
    train_results = LightGBM.fit!(model, X, y_lgbm[:]; verbosity=verbosity, weights=w, truncate_booster=mlj_model.truncate_booster)

    fitresult = (model, classes, deepcopy(mlj_model))
    # because update needs access to the older version of training metrics we keep them in the cache
    # so the update can merge old and additional metrics as necessary.
    cache = (
        num_boostings_done=[LightGBM.get_iter_number(model)],
        training_metrics=train_results["metrics"],
    )
    report = user_fitreport(model, train_results)

    return (fitresult, cache, report)

end


function update(mlj_model::MLJInterface.MODELS, verbosity::Int, fitresult, cache, X, y, w=AbstractFloat[])

    old_lgbm_model, old_classes, old_mlj_model = fitresult

    # we can continue boosting if and only if num_iterations has changed
    if !MLJModelInterface.is_same_except(old_mlj_model, mlj_model, :num_iterations)
        # if we get here it's just means we need to call fit directly
        return MLJInterface.fit(mlj_model, verbosity, X, y, w)
    end

    additional_iterations = mlj_model.num_iterations - old_mlj_model.num_iterations

    if additional_iterations < 0
        # less iterations isn't very valid so re-fit from scratch
        # TODO: I think theres a LightGBM API where you can prune
        # the boosting, so we would just do that instead of wasting time re-fitting
        # although we'd still have to consider fit-report etc
        return MLJInterface.fit(mlj_model, verbosity, X, y, w)
    end

    if verbosity >= 1
        @info("Not refitting from scratch", additional_iterations)
    end

    # splice the data into the estimator -- we need to update num_iterations,
    # taking into account it may have stopped early previously
    # It might well stop early again too, but give it a chance
    # Also ideally early stopping would be implemented via this mechanism and MLJ anyway
    num_iterations = sum(cache.num_boostings_done)
    old_lgbm_model.num_iterations = num_iterations + additional_iterations

    # eh this is ugly, possibly prompts a need for some refactoring
    train_results = LightGBM.train!(
        old_lgbm_model,
        additional_iterations,
        String[],
        verbosity,
        LightGBM.Dates.now();
        truncate_booster=old_mlj_model.truncate_booster
    )
    fitresult = (old_lgbm_model, old_classes, deepcopy(mlj_model))

    final_num_iter = LightGBM.get_iter_number(old_lgbm_model)
    iteration_history = deepcopy(cache.num_boostings_done)
    push!(iteration_history, final_num_iter - num_iterations)

    report = user_fitreport(old_lgbm_model, cache.training_metrics, train_results)
    newcache = (
        num_boostings_done=iteration_history,
        training_metrics=report.training_metrics,
    )

    return (fitresult, newcache, report)

end


# This does prep for classification tasks
@inline function prepare_targets(targets, model::LGBMClassifier)

    classes = MLJModelInterface.classes(first(targets))
    # -1 because these will be 1,2 and LGBM uses the 0/1 boundary
    # -This also works for multiclass because the classes are 0 indexed
    targets = Float64.(MLJModelInterface.int(targets) .- 1)
    return targets, classes

end
# This does prep for Regression, which is basically a no-op (or rather, just creation of an empty placeholder classes object
@inline prepare_targets(targets::AbstractVector, model::LGBMRegressor) = targets, []


function predict_classifier((fitted_model, classes, _), Xnew)

    Xnew = MLJModelInterface.matrix(Xnew)
    predicted = LightGBM.predict(fitted_model, Xnew)
    # when the objective == binary, lightgbm internally has classes = 1 and spits out only probability of positive class
    if size(predicted, 2) == 1
        predicted = hcat(1. .- predicted, predicted)
    end

    return MLJModelInterface.UnivariateFinite(classes, predicted)

end


function predict_regression((fitted_model, classes, _), Xnew)

    Xnew = MLJModelInterface.matrix(Xnew)
    return dropdims(LightGBM.predict(fitted_model, Xnew), dims=2)

end

# This function returns a user accessible report and therefore needs to not be a source of breaking changes
# Keep all of the functinality required to generate this into a single function so that it can be
# kept under control via tests and so on. It's not a lot for now but is likely to grow.
function user_fitreport(estimator::LightGBM.LGBMEstimator, fit_metrics::Dict)

    importance = (gain = LightGBM.gain_importance(estimator), split = LightGBM.split_importance(estimator))
    return (training_metrics = deepcopy(fit_metrics["metrics"]), importance = importance, best_iter = fit_metrics["best_iter"])

end
function user_fitreport(estimator::LightGBM.LGBMEstimator, cached_training_metrics::Dict, new_fit_metrics::Dict)

    metrics = LightGBM.merge_metrics(cached_training_metrics, new_fit_metrics["metrics"])

    merged_fit_metrics = Dict(
        "best_iter" => new_fit_metrics["best_iter"],
        "metrics" => metrics
    )

    return user_fitreport(estimator, merged_fit_metrics)

end


# multiple dispatch the various signatures for each model and args combo
MLJModelInterface.fit(model::MLJInterface.MODELS, verbosity::Int, X, y) = MLJInterface.fit(model, verbosity, X, y)
MLJModelInterface.fit(model::MLJInterface.MODELS, verbosity::Int, X, y, w::Nothing) = MLJInterface.fit(model, verbosity, X, y)
MLJModelInterface.fit(model::MLJInterface.MODELS, verbosity::Int, X, y, w) = MLJInterface.fit(model, verbosity, X, y, w)

# Add `update` method
MLJModelInterface.update(model::MLJInterface.MODELS, verbosity::Int, fitresult, cache, X, y) = MLJInterface.update(model, verbosity, fitresult, cache, X, y)
MLJModelInterface.update(model::MLJInterface.MODELS, verbosity::Int, fitresult, cache, X, y, w::Nothing) = MLJInterface.update(model, verbosity, fitresult, cache, X, y)
MLJModelInterface.update(model::MLJInterface.MODELS, verbosity::Int, fitresult, cache, X, y, w) = MLJInterface.update(model, verbosity, fitresult, cache, X, y, w)

MLJModelInterface.predict(model::MLJInterface.LGBMClassifier, fitresult, Xnew) = MLJInterface.predict_classifier(fitresult, Xnew)
MLJModelInterface.predict(model::MLJInterface.LGBMRegressor, fitresult, Xnew) = MLJInterface.predict_regression(fitresult, Xnew)

# multiple dispatch the model initialiser functions
model_init(mlj_model::MLJInterface.LGBMClassifier, classes) = LightGBM.LGBMClassification(; mlj_to_kwargs(mlj_model, classes)...)
model_init(mlj_model::MLJInterface.LGBMRegressor, targets) = LightGBM.LGBMRegression(; mlj_to_kwargs(mlj_model)...)


# metadata
MLJModelInterface.metadata_pkg.(
    (LGBMClassifier, LGBMRegressor); # end positional args
    name="LightGBM",
    uuid="7acf609c-83a4-11e9-1ffb-b912bcd3b04a",
    url="https://github.com/IQVIA-ML/LightGBM.jl",
    julia=false,
    license="MIT Expat",
    is_wrapper=false,
)


MLJModelInterface.metadata_model(
    LGBMClassifier; # end positional args
    path="LightGBM.MLJInterface.LGBMClassifier",
    input=MLJModelInterface.Table(MLJModelInterface.Continuous),
    target=AbstractVector{<:MLJModelInterface.Finite},
    weights=true,
    human_name="LightGBM classifier",
)

MLJModelInterface.metadata_model(
    LGBMRegressor; # end positional args
    path="LightGBM.MLJInterface.LGBMRegressor",
    input=MLJModelInterface.Table(MLJModelInterface.Continuous),
    target=AbstractVector{MLJModelInterface.Continuous},
    weights=true,
    human_name="LightGBM regressor",
)

"""
$(MLJModelInterface.doc_header(LGBMRegressor))

LightGBM, short for light gradient-boosting machine, is a
framework for gradient boosting based on decision tree algorithms and used for
classification, regression and other machine learning tasks, with a focus on
performance and scalability. This model in particular is used for various types of
regression tasks.

# Training data 

In MLJ or MLJBase, bind an instance `model` to data with 

  mach = machine(model, X, y) 

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of
  scitype `Continuous`; check the column scitypes with `schema(X)`; alternatively,
  `X` is any `AbstractMatrix` with `Continuous` elements; check the scitype with
  `scitype(X)`.
- y is a vector of targets whose items are of scitype `Continuous`. Check the
  scitype with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features
  `Xnew`, which should have the same scitype as `X` above.

# Hyper-parameters

- `boosting::String = "gbdt"`: Which boosting algorithm to use. One of:
    - gbdt: traditional gradient boosting
    - rf: random forest
    - dart: dropout additive regression trees
    - goss: gradient one side sampling
- `num_iterations::Int = 10::(_ >= 0)`: Number of iterations to run the boosting algorithm.
- `learning_rate::Float64 = 0.1::(_ > 0.)`: The update or shrinkage rate. In `dart`
  boosting, also affects the normalization weights of dropped trees.
- `num_leaves::Int = 31::(1 < _ <= 131072)`: The maximum number of leaves in one tree.
- `max_depth::Int = -1`: The limit on the maximum depth of a tree. Used to reduce
  overfitting. Set to `≤0` for unlimited depth
- `tree_learner::String = "serial"`: The tree learning mode. One of:
    - "serial":: Single machine tree learner.
    - "feature": feature parallel tree learner.
    - "data": data parallel tree learner
    - "voting": voting parallel tree learner. see the [LightGBM distributed
      learning
      guide](https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html)
      for details
- `histogram_pool_size::Float64 = -1.0`: Max size in MB for the historical
  histogram. Set to `≤0` for an unlimited size.
- `min_data_in_leaf::Int = 20::(_ >= 0)`: Minimal number of data in one leaf. Can be used to
  deal with over-fitting.
- `min_sum_hessian_in_leaf::Float64 = 1e-3::(_ >= 0.0)`: Minimal sum hessian in one leaf. Like
  `min_data_in_leaf`, it can be used to deal with over-fitting.
- `max_delta_step::Float64 = 0.0`: Used to limit the max output of tree leaves.
  The final maximum amount of leaves is `max_delta_step * learning_rate`. A value
  less than 0 means no limit on the max output.
- `lambda_l1::Float64 = 0.0::(_ >= 0.0)`: L1 regularization.
- `lambda_l2::Float64 = 0.0::(_ >= 0.0)`: L2 regularization.
- `min_gain_to_split::Float64 = 0.0::(_ >= 0.0)`: The minimal gain required to perform a
  split. Can be used to speed up training.
- `feature_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`: The fraction of features to select before
  fitting a tree. Can be used to speed up training and reduce over-fitting.
- `feature_fraction_bynode::Float64 = 1.0::(0.0 < _ <= 1.0)`: The fraction of features to select for
  each tree node. Can be used to reduce over-fitting.
- `feature_fraction_seed::Int = 2`: Random seed to use for the gesture fraction
- `bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`: The fraction of samples to use before
- `pos_bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`: The fraction of positive samples to use
  before fitting a tree. Can be used to speed up training and reduce over-fitting.
- `neg_bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`: The fraction of negative samples to use
  before fitting a tree. Can be used to speed up training and reduce over-fitting.
- `bagging_freq::Int = 0::(_ >= 0)`: The frequency to perform bagging at. At frequency `k`,
  every `k` samples select `bagging_fraction` of the data and use that data for
  the next `k` iterations.
- `bagging_seed::Int = 3`: The random seed to use for bagging.
- `early_stopping_round::Int = 0`: Will stop training if a validation metric does
  not improve over `early_stopping_round` rounds.
- `extra_trees::Bool = false`: Use extremely randomized trees. If true, will only
  check one randomly chosen threshold before splitting. Can be used to speed up
  training and reduce over-fitting.
- `extra_seed::Int = 6`: The random seed to use for `extra_trees`.
- `max_bin::Int = 255::(_ > 1)`: Number of bins feature values will be bucketed in. Smaller
  values may reduce training accuracy and help alleviate over-fitting.
- `bin_construct_sample_cnt = 200000::(_ > 0)`: Number of samples to use to construct bins.
  Larger values will give better results but may increase data loading time.
- `drop_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)`: The dropout rate for `dart`.
- `max_drop::Int = 50`: The maximum number of trees to drop in `dart`.
- `skip_drop:: Float64 = 0.5::(0.0 <= _ <= 1)`: Probability of skipping dropout in `dart`.
- `xgboost_dart_mode::Bool`: Set to true if you want to use xgboost dart mode in
  dart.
- `uniform_drop::Bool`: Set to true if you want to use uniform dropout in `dart`.
- `drop_seed::Int = 4`: Random seed for `dart` dropout.
- `top_rate::Float64 = 0.2::(0.0 <= _ <= 1.0)`: The retain ratio of large gradient data in `goss`.
- `other_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)`: The retain ratio of large gradient data in `goss`.
- `min_data_per_group::Int = 100::(_ > 0)`: Minimal amount of data per categorical group.
- `max_cat_threshold::Int = 32::(_ > 0)`: Limits the number of split points considered for
  categorical features.
- `cat_l2::Float64 = 10.0::(_ >= 0)`: L2 regularization for categorical splits
- `cat_smooth::Float64 = 10.0::(_ >= 0)`: Reduces noise in categorical features,
  particularly useful when there are categories with little data
- `sigmoid::Float64 = 1.0::(_ > 0.0 )`: A calibration scaling factor for the output 
  probabilities for binary and multiclass OVA
- `objective::String = "regression"`: The objective function to use. One of:
    - "regression": L2 loss or mse.
    - "regression_l1": L1 loss or mae.
    - "huber": Huber loss.
    - "fair": Fair loss.
    - "poisson": poisson regression.
    - "quantile": Quantile regression.
    - "mape": MAPE (mean mean_absolute_percentage_error) loss.
    - "gamma": Gamma regression with log-link.
    - "tweedie": Tweedie regression with log-link.
- `categorical_feature::Vector{Int} = Vector{Int}()`: Used to specify the
  categorical features. Items in the vector are column indices representing which
  features should be interpreted as categorical.
- `data_random_seed::Int = 1`: Random seed used when constructing histogram bins.
- `is_sparse::Bool = true`: Enable/disable sparse optimization.
- `is_unbalance::Bool = false`: Set to true if training data is unbalanced.
- `boost_from_average::Bool = true`: Adjusts the initial score to the mean of
  labels for faster convergence.
- `scale_pos_weight::Float64 = 1.0`: Control the balance of positive and negative
  weights. Useful for unbalanced classes.
- `use_missing::Bool = true`: Whether or not to handle missing values.
- `linear_tree::Bool = false`: Set to true to use linear splits.
- `feature_pre_filter::Bool = true`: Whether or not to ignore unsplittable
  features.
- `alpha::Float64 = 0.9::(_ > 0.0 )`: Parameter used for huber and quantile regression.
- `metric::Vector{String} = ["l2"]`: Metric(s) to be used when evaluating on
  evaluation set. For detailed information, see [the official
  documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters)
- `metric_freq::Int = 1::(_ > 0)`: The frequency to run metric evaluation at.
- `is_provide_training_metric::Bool = false`: Set to `true` to output metric result on
  training dataset.
- `eval_at::Vector{Int} = Vector{Int}([1, 2, 3, 4, 5])::(all(_ .> 0))`: Evaluation positions for
  ndcg and map metrics.
- `num_machines::Int = 1::(_ > 0)`: Number of machines to use when doing distributed
  learning.
- `num_threads::Int  = 0::(_ >= 0)`: Number of threads to use.
- `local_listen_port::Int = 12400::(_ > 0)`: TCP listen port.
- `time_out::Int = 120::(_ > 0)`: Socket timeout.
- `machine_list_file::String = ""`: Path of files that lists the machines used for
  distributed learning.
- `save_binary::Bool = false`: Whether or not to save the dataset to a binary file
- `device_type::String = "cpu"`: The type of device being used. One of `cpu` or
  `gpu`
- `gpu_use_dp::Bool = false`: Whether or not to use double precision on the GPU.
- `gpu_platform_id::Int = -1`: The platform ID of the GPU to use.
- `gpu_device_id::Int = -1`: The device ID of the GPU to use.
- `num_gpu::Int = 1`: The number of GPUs to use.
- `force_col_wise::Bool = false`: Force column wise histogram building. Only
  applicable on cpu.
- `force_row_wise::Bool = false`: Force row wise histogram building. Only
  applicable on cpu.
- `truncate_booster::Bool = true`: Whether or not to truncate the booster.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `fitresult`: Fitted model information, contains a `LGBMRegression` object, an
  empty vector, and the regressor with all its parameters

# Report

The fields of `report(mach)` are:

- `training_metrics`: A dictionary containing all training metrics.
- `importance`: A `namedtuple` containing:
    - `gain`: The total gain of each split used by the model
    - `split`: The number of times each feature is used by the model.

# Examples

```julia

using DataFrames
using MLJ

# load the model (make sure to Pkg.add LightGBM to the environment)
LGBMRegressor = @load LGBMRegressor

X, y = @load_boston # a table and a vector 
X = DataFrame(X)
train, test = partition(collect(eachindex(y)), 0.70, shuffle=true)

first(X, 3)
lgb = LGBMRegressor() #initialised a model with default params
mach = machine(lgb, X[train, :], y[train]) |> fit!

predict(mach, X[test, :])

# Access feature importances
model_report = report(mach)
gain_importance = model_report.importance.gain
split_importance = model_report.importance.split
```

"""
LGBMRegressor


"""
$(MLJModelInterface.doc_header(LGBMClassifier))

`LightGBM, short for light gradient-boosting machine, is a
framework for gradient boosting based on decision tree algorithms and used for
classification and other machine learning tasks, with a focus on
performance and scalability. This model in particular is used for various types of
classification tasks.

# Training data In MLJ or MLJBase, bind an instance `model` to data with 

  mach = machine(model, X, y) 

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of
  scitype `Continuous`; check the column scitypes with `schema(X)`; alternatively,
  `X` is any `AbstractMatrix` with `Continuous` elements; check the scitype with
  `scitype(X)`.
- y is a vector of targets whose items are of scitype `Continuous`. Check the
  scitype with scitype(y).

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features
  `Xnew`, which should have the same scitype as `X` above.

# Hyper-parameters

- `boosting::String = "gbdt"`: Which boosting algorithm to use. One of:
    - gbdt: traditional gradient boosting
    - rf: random forest
    - dart: dropout additive regression trees
    - goss: gradient one side sampling
- `num_iterations::Int = 10::(_ >= 0)`: Number of iterations to run the boosting algorithm.
- `learning_rate::Float64 = 0.1::(_ > 0.)`: The update or shrinkage rate. In `dart`
  boosting, also affects the normalization weights of dropped trees.
- `num_leaves::Int = 31::(1 < _ <= 131072)`: The maximum number of leaves in one tree.
- `max_depth::Int = -1`: The limit on the maximum depth of a tree.
- `tree_learner::String = "serial"`: The tree learning mode. One of:
    - serial: Single machine tree learner.
    - feature: feature parallel tree learner.
    - data: data parallel tree learner
    - voting: voting parallel tree learner. see the [LightGBM distributed learning
      guide](https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html)
      for details
- `histogram_pool_size::Float64 = -1.0`: Max size in MB for the historical
  histogram.
- `min_data_in_leaf::Int = 20::(_ >= 0)`: Minimal number of data in one leaf. Can be used to
  deal with over-fitting.
- `min_sum_hessian_in_leaf::Float64 = 1e-3::(_ >= 0.0)`: Minimal sum hessian in one leaf. Like
  min_data_in_leaf, it can be used to deal with over-fitting.
- `max_delta_step::Float64 = 0.0`: Used to limit the max output of tree leaves.
  The final maximum amount of leaves is `max_delta_step * learning_rate`.
- `lambda_l1::Float64 = 0.0::(_ >= 0.0)`: L1 regularization.
- `lambda_l2::Float64 = 0.0::(_ >= 0.0)`: L2 regularization.
- `min_gain_to_split::Float64 = 0.0::(_ >= 0.0)`: The minimal gain required to perform a
  split. Can be used to speed up training.
- `feature_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`: The fraction of features to select before
  fitting a tree. Can be used to speed up training and reduce over-fitting.
- `feature_fraction_bynode::Float64 = 1.0::(0.0 < _ <= 1.0)`: The fraction of features to select for
  each tree node. Can be used to reduce over-fitting.
- `feature_fraction_seed::Int = 2`: Random seed to use for the gesture fraction
- `bagging_fraction::Float64 = 1.0`: The fraction of samples to use before
  fitting a tree. Can be used to speed up training and reduce over-fitting.
- `bagging_freq::Int = 0::(_ >= 0)`: The frequency to perform bagging at. At frequency `k`,
  every `k` samples select `bagging_fraction` of the data and use that data for
  the next `k` iterations.
- `bagging_seed::Int = 3`: The random seed to use for bagging.
- `early_stopping_round::Int = 0`: Will stop training if a validation metric does
  not improve over `early_stopping_round` rounds.
- `extra_trees::Bool = false`: Use extremely randomized trees. If true, will only
  check one randomly chosen threshold before splitting. Can be used to speed up
  training and reduce over-fitting.
- `extra_seed::Int = 6`: The random seed to use for `extra_trees`.
- `max_bin::Int = 255::(_ > 1)`: Number of bins feature values will be bucketed in. Smaller
  values may reduce training accuracy and help alleviate over-fitting.
- `bin_construct_sample_cnt = 200000::(_ > 0)`: Number of samples to use to construct bins.
  Larger values will give better results but may increase data loading time.
- `drop_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)`: The dropout rate for `dart`.
- `max_drop::Int = 50`: The maximum number of trees to drop in `dart`.
- `skip_drop:: Float64 = 0.5::(0.0 <= _ <= 1)`: Probability of skipping dropout in `dart`.
- `xgboost_dart_mode::Bool`: Set to true if you want to use xgboost dart mode in
  dart.
- `uniform_drop::Bool`: Set to true if you want to use uniform dropout in `dart`.
- `drop_seed::Int = 4`: Random seed for `dart` dropout.
- `top_rate::Float64 = 0.2::(0.0 <= _ <= 1.0)`: The retain ratio of large gradient data in `goss`.
- `other_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)`: The retain ratio of large gradient data in `goss`.
- `min_data_per_group::Int = 100::(_ > 0)`: Minimal amount of data per categorical group.
- `max_cat_threshold::Int = 32::(_ > 0)`: Limits the number of split points considered for
  categorical features.
- `cat_l2::Float64 = 10.0::(_ >= 0)`: L2 regularization for categorical splits
- `cat_smooth::Float64 = 10.0::(_ >= 0)`: Reduces noise in categorical features,
  particularly useful when there are categories with little data
- `objective::String = "multiclass"`: The objective function to use. One of:
    - binary: Binary log loss classification.
    - multiclass: Softmax classification.
    - multiclassova: One verse all multiclass classification. `num_class` should
      be set as well
    - cross_entropy: Cross-entropy objective function.
    - cross_entropy_lambda: Alternative parametrized form of the cross-entropy
      objective function.
    - lambdarank: The lambdarank objective function, for use in ranking
      applications.
    - rank_xendcg: The XE_NDCG_MART ranking objective function. Faster than
      lambdarank with same peroformance.
- `categorical_feature::Vector{Int} = Vector{Int}()`: Used to specify the
  categorical features. Items in the vector are column indices representing which
  features should be interpreted as categorical.
- `data_random_seed::Int = 1`: Random seed used when constructing histogram bins.
- `is_sparse::Bool = true`: Enable/disable sparse optimization.
- `is_unbalance::Bool = false`: Set to true if training data is unbalanced.
- `boost_from_average::Bool = true`: Adjusts the initial score to the mean of
  labels for faster convergence.
- `use_missing::Bool = true`: Whether or not to handle missing values.
- `linear_tree::Bool = false`: Set to true to use linear splits.
- `feature_pre_filter::Bool = true`: Whether or not to ignore unsplittable
  features.
- `metric::Vector{String} = ["none"]`: Metric(s) to be used when evaluating on
  evaluation set. For detailed information, see [the official
  documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters)
- `metric_freq::Int = 1::(_ > 0)`: The frequency to run metric evaluation at.
- `is_provide_training_metric::Bool = false`: Set true to output metric result on training
  dataset.
  - `eval_at::Vector{Int} = Vector{Int}([1, 2, 3, 4, 5])::(all(_ .> 0))`: Evaluation positions for
  ndcg and map metrics.
- `num_machines::Int = 1::(_ > 0)`: Number of machines to use when doing distributed
  learning.
- `num_threads::Int  = 0::(_ >= 0)`: Number of threads to use.
- `local_listen_port::Int = 12400::(_ > 0)`: TCP listen port.
- `time_out::Int = 120::(_ > 0)`: Socket timeout.
- `machine_list_file::String = ""`: Path of files that lists the machines used for
  distributed learning.
- `save_binary::Bool = false`: Whether or not to save the dataset to a binary file
- `device_type::String = "cpu"`: The type of device being used. One of `cpu` or
  `gpu`
- `gpu_use_dp::Bool = false`: Whether or not to use double precision on the GPU.
- `gpu_platform_id::Int = -1`: The platform ID of the GPU to use.
- `gpu_device_id::Int = -1`: The device ID of the GPU to use.
- `num_gpu::Int = 1`: The number of GPUs to use.
- `force_col_wise::Bool = false`: Force column wise histogram building. Only
  applicable on cpu.
- `force_row_wise::Bool = false`: Force row wise histogram building. Only
  applicable on cpu.
- `truncate_booster::Bool = true`: Whether or not to truncate the booster.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `fitresult`: Fitted model information, contains a `LGBMClassification` object, a
  `CategoricalArray` of the input class names, and the classifier with all its
  parameters

# Report

The fields of `report(mach)` are:

- `training_metrics`: A dictionary containing all training metrics.
- `importance`: A `namedtuple` containing:
    - `gain`: The total gain of each split used by the model
    - `split`: The number of times each feature is used by the model.


# Examples

```julia

using DataFrames
using MLJ

# load the model (make sure to Pkg.add LightGBM to the environment)
LGBMClassifier = @load LGBMClassifier

X, y = @load_iris 
X = DataFrame(X)
train, test = partition(collect(eachindex(y)), 0.70, shuffle=true)

first(X, 3)
lgb = LGBMClassifier() #initialised a model with default params
mach = machine(lgb, X[train, :], y[train]) |> fit!

predict(mach, X[test, :])

# Access feature importances
model_report = report(mach)
gain_importance = model_report.importance.gain
split_importance = model_report.importance.split
```

"""
LGBMClassifier

end # module
