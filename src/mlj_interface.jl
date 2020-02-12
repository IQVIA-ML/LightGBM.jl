module mlj_interface

using MLJBase
using ScientificTypes

import LightGBM


const LGBM_METRICS = (
                        "None", "l1", "l2", "rmse", "quantile", "mape", "huber", "fair", "poisson", "gamma", "gamma_deviance",
                        "tweedie", "ndcg", "lambdarank", "map", "mean_average_precision", "auc", "binary_logloss", "binary",
                        "binary_error", "auc_mu", "multi_logloss", "multi_error", "cross_entropy", "xentropy", "multi_logloss",
                        "multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr", "cross_entropy_lambda",
                        "xentlambda", "kullback_leibler", "kldiv",
)


@mlj_model mutable struct LGBMRegression <: MLJBase.Deterministic

    # Hyperparameters, see https://lightgbm.readthedocs.io/en/latest/Parameters.html for defaults
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
    init_score::String = ""

    # Model properties
    categorical_feature::Vector{Int} = Vector{Int}()
    data_random_seed::Int = 1
    is_sparse::Bool = true
    is_unbalance::Bool = false

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


@mlj_model mutable struct LGBMBinary <: MLJBase.Probabilistic

    # Hyperparameters, see https://lightgbm.readthedocs.io/en/latest/Parameters.html for defaults
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
    init_score::String = ""

    # For documentation purposes: A calibration scaling factor for the output probabilities for binary and multiclass OVA
    # Not included above because this is only present for the binary model in the FFI wrapper
    sigmoid::Float64 = 1.0::(_ > 0.0 )

    # Model properties
    categorical_feature::Vector{Int} = Vector{Int}();
    data_random_seed::Int = 1
    is_sparse::Bool = true
    is_unbalance::Bool = false

    # Only accepted in the interface for Multiclass
    #num_class::Int = 1::(_ > 0)

    # Metrics
    metric::Vector{String} = ["binary_error"]::(all(in.(_, (LGBM_METRICS, ))))
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


@mlj_model mutable struct LGBMClassifier <: MLJBase.Probabilistic

    # Hyperparameters, see https://lightgbm.readthedocs.io/en/latest/Parameters.html for defaults
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
    init_score::String = ""

    # For documentation purposes: A calibration scaling factor for the output probabilities for binary and multiclass OVA
    # Not included above because this is only present for the binary model in the FFI wrapper, hence commented out
    # sigmoid::Float64 = 1.0::(_ > 0.0 )

    # Model properties
    categorical_feature::Vector{Int} = Vector{Int}();
    data_random_seed::Int = 1
    is_sparse::Bool = true
    is_unbalance::Bool = false

    # Only accepted in the interface for Multiclass
    # LightGBM itself will throw if left at the default value
    num_class::Int = 1::(_ > 0)

    # Metrics
    metric::Vector{String} = ["multi_error"]::(all(in.(_, (LGBM_METRICS, ))))
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


CLASSIFIERS = Union{LGBMBinary, LGBMClassifier}
MODELS = Union{LGBMBinary, LGBMClassifier, LGBMRegression}


function mlj_to_kwargs(model::MLJBase.Supervised)

    return Dict{Symbol, Any}(
        name => getfield(model, name)
        for name in fieldnames(typeof(model))
    )

end


# X and y must be untyped per MLJ docs (and probably w too, getting nothing by default is zzz though)
function fit(mlj_model::MODELS, verbosity::Int, X, y, w=Vector{AbstractFloat}())

    # MLJ docs are clear that 0 means silent. but 0 in LightGBM world means "warnings"
    # and < 0 means fatal logs only, so we put intended silence to -1 (which is probably the closest we get)
    verbosity = if verbosity == 0; -1 else verbosity end

    y_lgbm, classes = prepare_targets(y, mlj_model)
    model = model_init(mlj_model)
    X = MLJBase.matrix(X)
    # The FFI wrapper wants Float32 for these
    w = Float32.(w)
    report = LightGBM.fit(model, X, y_lgbm; verbosity=verbosity, weights=w)

    model = (model, classes)
    cache = nothing
    report = (report,)

    # Caution: The model is a pointer to a memory location including its training data
    # which is probably bum ... fix in LightGBM.jl?
    return (model, cache, report)

end


# This does prep for classification tasks
@inline function prepare_targets(
    targets::MLJBase.CategoricalArrays.CategoricalArray,
    model::CLASSIFIERS,
)::Tuple{Vector{Float64}, MLJBase.CategoricalArrays.CategoricalArray}

    classes = MLJBase.classes(first(targets))
    if isa(model, mlj_interface.LGBMBinary)
        # We just need to check this one because other cases throw by LightGBM
        # Whereas this will run and give incorrect (unexpected) results
        if length(classes) > 2
            throw(ErrorException("Binary classification with $(length(classes)) categories"))
        end
    end
    # -1 because these will be 1,2 and LGBM uses the 0/1 boundary
    # -This also works for multiclass because the classes ae 0 indexed
    targets = Float64.(MLJBase.int(targets) .- 1)
    return targets, classes
end

# This does prep for Regression, which is basically a no-op (or rather, just creation of an empty placeholder classes object
@inline function prepare_targets(
    targets::AbstractVector,
    model::LGBMRegression,
)::Tuple{Vector{Float64}, MLJBase.CategoricalArrays.CategoricalArray}

    return targets, MLJBase.CategoricalArrays.CategoricalArray(undef)

end


function predict_binary((fitted_model, classes), Xnew)

    Xnew = MLJBase.matrix(Xnew)
    predicted = LightGBM.predict(fitted_model, Xnew)
    return [MLJBase.UnivariateFinite(classes, [1 - pred, pred]) for pred in predicted]

end

function predict_multi((fitted_model, classes), Xnew)

    Xnew = MLJBase.matrix(Xnew)
    predicted = LightGBM.predict(fitted_model, Xnew)
    # much rather use `eachrow` here but it requires julia >= 1.1 and thats probably not cool
    return [MLJBase.UnivariateFinite(classes, predicted[row, :]) for row in 1:size(predicted, 1)]

end

function predict_regression((fitted_model, classes), Xnew)

    Xnew = MLJBase.matrix(Xnew)
    return LightGBM.predict(fitted_model, Xnew)

end

# multiple dispatch the various signatures for each model and args combo
MLJBase.fit(model::mlj_interface.MODELS, verbosity::Int, X, y) = mlj_interface.fit(model, verbosity, X, y)
MLJBase.fit(model::mlj_interface.MODELS, verbosity::Int, X, y, w::Nothing) = mlj_interface.fit(model, verbosity, X, y)
MLJBase.fit(model::mlj_interface.MODELS, verbosity::Int, X, y, w) = mlj_interface.fit(model, verbosity, X, y, w)

MLJBase.predict(model::mlj_interface.LGBMBinary, fitresult, Xnew) = mlj_interface.predict_binary(fitresult, Xnew)
MLJBase.predict(model::mlj_interface.LGBMClassifier, fitresult, Xnew) = mlj_interface.predict_multi(fitresult, Xnew)
MLJBase.predict(model::mlj_interface.LGBMRegression, fitresult, Xnew) = mlj_interface.predict_regression(fitresult, Xnew)

# multiple dispatch the model initialiser functions
model_init(mlj_model::mlj_interface.LGBMBinary) = LightGBM.LGBMBinary(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::mlj_interface.LGBMClassifier) = LightGBM.LGBMMulticlass(; mlj_to_kwargs(mlj_model)...)
model_init(mlj_model::mlj_interface.LGBMRegression) = LightGBM.LGBMRegression(; mlj_to_kwargs(mlj_model)...)


# metadata
MLJBase.load_path(::Type{<:LGBMBinary})          = "LightGBM.mlj_interface.LGBMBinary"
MLJBase.load_path(::Type{<:LGBMClassifier})      = "LightGBM.mlj_interface.LGBMMulticlass"
MLJBase.load_path(::Type{<:LGBMRegression})      = "LightGBM.mlj_interface.LGBMRegression"

MLJBase.package_name(::Type{<:MODELS})           = "LightGBM"
MLJBase.package_uuid(::Type{<:MODELS})           = "50415d55-5a07-4c42-a30b-abdb22ba6b8f"
MLJBase.is_pure_julia(::Type{<:MODELS})          = false
MLJBase.is_wrapper(::Type{<:MODELS})             = true
MLJBase.package_url(::Type{<:MODELS})            = "https://github.com/IQVIA-ML/LightGBM.jl"
MLJBase.input_scitype(::Type{<:MODELS})          = Table(Continuous)
MLJBase.target_scitype(::Type{<:CLASSIFIERS})    = AbstractVector{<:Finite}
MLJBase.target_scitype(::Type{<:LGBMRegression}) = AbstractVector{Continuous}
MLJBase.supports_weights(::Type{<:MODELS})       = true

end # module
