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


MLJModelInterface.@mlj_model mutable struct LGBMRegressor <: MLJModelInterface.Deterministic

    # Hyperparameters, see https://lightgbm.readthedocs.io/en/latest/Parameters.html for defaults
    boosting::String = "gbdt"::(_ in ("gbdt", "goss", "rf", "dart"))
    num_iterations::Int = 10::(_ >= 0)
    learning_rate::Float64 = 0.1::(_ > 0.)
    num_leaves::Int = 31::(1 < _ <= 131072)
    max_depth::Int = -1;#::(_ != 0);
    tree_learner::String = "serial"::(_ in ("serial", "feature", "data", "voting"))
    histogram_pool_size::Float64 = -1.0;#::(_ != 0.0);
    min_data_in_leaf::Int = 20::(_ >= 0)
    min_sum_hessian_in_leaf::Float64 = 1e-3::(_ >= 0.0)
    lambda_l1::Float64 = 0.0::(_ >= 0.0)
    lambda_l2::Float64 = 0.0::(_ >= 0.0)
    min_gain_to_split::Float64 = 0.0::(_ >= 0.0)
    feature_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    feature_fraction_seed::Int = 2
    bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    bagging_freq::Int = 0::(_ >= 0)
    bagging_seed::Int = 3
    early_stopping_round::Int = 0
    max_bin::Int = 255::(_ > 1)
    bin_construct_sample_cnt = 200000::(_ > 0)
    init_score::String = ""
    drop_rate::Float64 = 0.1 :: (0.0 <= _ <= 1.0)
    max_drop::Int = 50
    skip_drop:: Float64 = 0.5 :: (0.0 <= _ <= 1)
    xgboost_dart_mode::Bool
    uniform_drop::Bool
    drop_seed::Int = 4
    top_rate::Float64 = 0.2 :: (0.0 <= _ <= 1.0)
    other_rate::Float64 = 0.1 :: (0.0 <= _ <= 1.0)

    # Model properties
    objective::String = "regression"::(_ in REGRESSION_OBJECTIVES)
    categorical_feature::Vector{Int} = Vector{Int}()
    data_random_seed::Int = 1
    is_sparse::Bool = true
    is_unbalance::Bool = false
    boost_from_average::Bool = true
    use_missing::Bool = true

    # Metrics
    metric::Vector{String} = ["l2"]::(all(in.(_, (LGBM_METRICS, ))))
    metric_freq::Int = 1::(_ > 0)
    is_training_metric::Bool = false
    ndcg_at::Vector{Int} = Vector{Int}([1, 2, 3, 4, 5])::(all(_ .> 0))

    # Implementation parameters
    num_machines::Int = 1::(_ > 0)
    num_threads::Int  = 0::(_ >= 0)
    local_listen_port::Int = 12400::(_ > 0)
    time_out::Int = 120::(_ > 0)
    machine_list_file::String = ""
    save_binary::Bool = false
    device_type::String = "cpu"::(_ in ("cpu", "gpu"))

end


MLJModelInterface.@mlj_model mutable struct LGBMClassifier <: MLJModelInterface.Probabilistic

    # Hyperparameters, see https://lightgbm.readthedocs.io/en/latest/Parameters.html for defaults
    boosting::String = "gbdt"::(_ in ("gbdt", "goss", "rf", "dart"))
    num_iterations::Int = 10::(_ >= 0)
    learning_rate::Float64 = 0.1::(_ > 0.)
    num_leaves::Int = 31::(1 < _ <= 131072)
    max_depth::Int = -1;#::(_ != 0);
    tree_learner::String = "serial"::(_ in ("serial", "feature", "data", "voting"))
    histogram_pool_size::Float64 = -1.0;#::(_ != 0.0);
    min_data_in_leaf::Int = 20::(_ >= 0)
    min_sum_hessian_in_leaf::Float64 = 1e-3::(_ >= 0.0)
    lambda_l1::Float64 = 0.0::(_ >= 0.0)
    lambda_l2::Float64 = 0.0::(_ >= 0.0)
    min_gain_to_split::Float64 = 0.0::(_ >= 0.0)
    feature_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    feature_fraction_seed::Int = 2
    bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    bagging_freq::Int = 0::(_ >= 0)
    bagging_seed::Int = 3
    early_stopping_round::Int = 0
    max_bin::Int = 255::(_ > 1)
    bin_construct_sample_cnt = 200000::(_ > 0)
    init_score::String = ""
    drop_rate::Float64 = 0.1 :: (0.0 <= _ <= 1.0)
    max_drop::Int = 50
    skip_drop:: Float64 = 0.5 :: (0.0 <= _ <= 1)
    xgboost_dart_mode::Bool
    uniform_drop::Bool
    drop_seed::Int = 4
    top_rate::Float64 = 0.2 :: (0.0 <= _ <= 1.0)
    other_rate::Float64 = 0.1 :: (0.0 <= _ <= 1.0)

    # For documentation purposes: A calibration scaling factor for the output probabilities for binary and multiclass OVA
    # Not included above because this is only present for the binary model in the FFI wrapper, hence commented out
    # sigmoid::Float64 = 1.0::(_ > 0.0 )

    # Model properties
    objective::String = "multiclass"::(_ in CLASSIFICATION_OBJECTIVES)
    categorical_feature::Vector{Int} = Vector{Int}();
    data_random_seed::Int = 1
    is_sparse::Bool = true
    is_unbalance::Bool = false
    boost_from_average::Bool = true
    scale_pos_weight = 1.0
    use_missing::Bool = true

    # Metrics
    metric::Vector{String} = ["None"]::(all(in.(_, (LGBM_METRICS, ))))
    metric_freq::Int = 1::(_ > 0)
    is_training_metric::Bool = false
    ndcg_at::Vector{Int} = Vector{Int}([1, 2, 3, 4, 5])::(all(_ .> 0))

    # Implementation parameters
    num_machines::Int = 1::(_ > 0)
    num_threads::Int  = 0::(_ >= 0)
    local_listen_port::Int = 12400::(_ > 0)
    time_out::Int = 120::(_ > 0)
    machine_list_file::String = ""
    save_binary::Bool = false
    device_type::String = "cpu"::(_ in ("cpu", "gpu"))

end


MODELS = Union{LGBMClassifier, LGBMRegressor}


function mlj_to_kwargs(model::LGBMRegressor)

    return Dict{Symbol, Any}(
        name => getfield(model, name)
        for name in fieldnames(typeof(model))
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

    retval = Dict{Symbol, Any}(
        name => getfield(model, name)
        for name in fieldnames(typeof(model))
    )

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
    train_metrics = LightGBM.fit!(model, X, y_lgbm; verbosity=verbosity, weights=w)

    fitresult = (model, classes, deepcopy(mlj_model))
    # because update needs access to the older version of training metrics we keep them in the cache
    # so the update can merge old and additional metrics as necessary.
    cache = (num_boostings_done=[LightGBM.get_iter_number(model)], training_metrics=train_metrics)
    report = user_fitreport(model, train_metrics)

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
    train_metrics = LightGBM.train!(old_lgbm_model, additional_iterations, String[], verbosity, LightGBM.Dates.now())
    fitresult = (old_lgbm_model, old_classes, deepcopy(mlj_model))

    final_num_iter = LightGBM.get_iter_number(old_lgbm_model)
    iteration_history = deepcopy(cache.num_boostings_done)
    push!(iteration_history, final_num_iter - num_iterations)
    report = user_fitreport(old_lgbm_model, cache.training_metrics, train_metrics)
    newcache = (num_boostings_done=iteration_history, training_metrics=report.training_metrics)

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
function user_fitreport(estimator::LightGBM.LGBMEstimator, eval_metrics::Dict)

    importance = (gain = LightGBM.gain_importance(estimator), split = LightGBM.split_importance(estimator))
    return (training_metrics = deepcopy(eval_metrics), importance = importance)

end
function user_fitreport(estimator::LightGBM.LGBMEstimator, eval_metrics::Dict, new_eval_metrics::Dict)

    metrics = LightGBM.merge_scores(eval_metrics, new_eval_metrics)
    return user_fitreport(estimator, metrics)

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
    descr="Microsoft LightGBM FFI wrapper: Classifier",
)

MLJModelInterface.metadata_model(
    LGBMRegressor; # end positional args
    path="LightGBM.MLJInterface.LGBMRegressor",
    input=MLJModelInterface.Table(MLJModelInterface.Continuous),
    target=AbstractVector{MLJModelInterface.Continuous},
    weights=true,
    descr="Microsoft LightGBM FFI wrapper: Regressor",
)

end # module
