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
    :feature_importance,
)

struct LGBMFrontEndData
    matrix::AbstractMatrix
    dataset::LightGBM.Dataset
    feature_names::Union{Nothing, Tuple}
    dataset_params::String
    # When selectrows is called with unsorted row indices, LightGBM's LGBM_DatasetGetSubset
    # requires sorted indices. We then store invperm(sortperm(rows)) so predictions can be
    # reordered to match the requested row order.
    row_invperm::Union{Nothing, AbstractVector{Int}}
end

function extract_feature_names(X)
    feature_names = try
        Tuple(propertynames(X))
    catch
        try
            Tuple(propertynames(first(X)))
        catch
            nothing
        end
    end
    return feature_names === nothing || isempty(feature_names) ? nothing : feature_names
end

MLJModelInterface.@mlj_model mutable struct LGBMRegressor <: MLJModelInterface.Deterministic

    # Hyperparameters, see https://lightgbm.readthedocs.io/en/latest/Parameters.html for defaults
    # Core parameters
    objective::String = "regression"::(_ in REGRESSION_OBJECTIVES)
    boosting::String = "gbdt"::(_ in ("gbdt", "goss", "rf", "dart"))
    num_iterations::Int = 100::(_ >= 0)
    learning_rate::Float64 = 0.1::(_ > 0.)
    num_leaves::Int = 31::(1 < _ <= 131072)
    tree_learner::String = "serial"::(_ in ("serial", "feature", "data", "voting"))
    num_threads::Int  = 0::(_ >= 0)
    device_type::String = "cpu"::(_ in ("cpu", "gpu"))
    seed::Int = 0
    deterministic::Bool = false

    # Learning control parameters
    force_col_wise::Bool = false
    force_row_wise::Bool = false
    histogram_pool_size::Float64 = -1.0;#::(_ != 0.0);
    max_depth::Int = -1;#::(_ != 0);
    min_data_in_leaf::Int = 20::(_ >= 0)
    min_sum_hessian_in_leaf::Float64 = 1e-3::(_ >= 0.0)
    bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    bagging_freq::Int = 0::(_ >= 0)
    bagging_seed::Int = 3
    feature_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    feature_fraction_bynode::Float64 = 1.0::(0.0 < _ <= 1.0)
    feature_fraction_seed::Int = 2
    extra_trees::Bool = false
    extra_seed::Int = 6
    early_stopping_round::Int = 0
    first_metric_only::Bool = false
    max_delta_step::Float64 = 0.0
    lambda_l1::Float64 = 0.0::(_ >= 0.0)
    lambda_l2::Float64 = 0.0::(_ >= 0.0)
    linear_lambda::Float64 = 0.0::(_ >= 0.0)
    min_gain_to_split::Float64 = 0.0::(_ >= 0.0)
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
    max_cat_to_onehot::Int = 4::(_ > 0)
    top_k::Int = 20::(_ > 0)
    monotone_constraints::Vector{Int} = Vector{Int}()
    monotone_constraints_method::String = "basic"::(_ in ("basic", "intermediate", "advanced"))
    monotone_penalty::Float64 = 0.0::(_ >= 0.0)
    feature_contri::Vector{Float64} = Vector{Float64}()
    forcedsplits_filename::String = ""
    refit_decay_rate::Float64 = 0.9::(0.0 <= _ <= 1.0)
    cegb_tradeoff::Float64 = 1.0::(_ >= 0.0)
    cegb_penalty_split::Float64 = 0.0::(_ >= 0.0)
    cegb_penalty_feature_lazy::Vector{Float64} = Vector{Float64}()
    cegb_penalty_feature_coupled::Vector{Float64} = Vector{Float64}()
    path_smooth::Float64 = 0.0::(_ >= 0.0)
    interaction_constraints::Vector{Vector{Int}} = Vector{Vector{Int}}()
    verbosity::Int = 1
    
    # Dataset parameters
    linear_tree::Bool = false
    max_bin::Int = 255::(_ > 1)
    max_bin_by_feature::Vector{Int} = Vector{Int}()
    min_data_in_bin::Int = 3::(_ > 0)
    bin_construct_sample_cnt::Int = 200000::(_ > 0)
    data_random_seed::Int = 1
    is_enable_sparse::Bool = true
    enable_bundle::Bool = true
    use_missing::Bool = true
    zero_as_missing::Bool = false
    feature_pre_filter::Bool = true
    pre_partition::Bool = false
    two_round::Bool = false
    header::Bool = false
    label_column::String = ""   
    weight_column::String = ""
    ignore_column::String  = ""
    categorical_feature::Vector{Int} = Vector{Int}()
    forcedbins_filename::String = ""
    precise_float_parser::Bool = false

    # Predict parameters
    start_iteration_predict::Int = 0
    num_iteration_predict::Int = -1
    predict_raw_score::Bool = false
    predict_leaf_index::Bool = false
    predict_contrib::Bool = false
    predict_disable_shape_check::Bool = false
    
    # Objective parameters
    is_unbalance::Bool = false
    boost_from_average::Bool = true
    reg_sqrt::Bool = false
    alpha::Float64 = 0.9::(_ > 0.0 )
    fair_c::Float64 = 1.0::(_ > 0.0 )
    poisson_max_delta_step::Float64 = 0.7::(_ > 0.0 )
    tweedie_variance_power::Float64 = 1.5::(1.0 <= _ < 2.0)

    # Metrics
    metric::Vector{String} = ["l2"]::(all(in.(_, (LGBM_METRICS, ))))
    metric_freq::Int = 1::(_ > 0)
    is_provide_training_metric::Bool = false
    eval_at::Vector{Int} = Vector{Int}([1, 2, 3, 4, 5])::(all(_ .> 0))

    # Network parameters
    num_machines::Int = 1::(_ > 0)
    local_listen_port::Int = 12400::(_ > 0)
    time_out::Int = 120::(_ > 0)
    machine_list_filename::String = ""
    machines::String = ""

    # GPU parameters
    gpu_platform_id::Int = -1
    gpu_device_id::Int = -1
    gpu_use_dp::Bool = false
    num_gpu::Int = 1

    # Other (non-lightbm) parameters
    truncate_booster::Bool = true
    
    # Feature importance method for MLJModelInterface.feature_importances
    # :gain - importance based on information gain (default)
    # :split - importance based on number of times feature was used in splits
    feature_importance::Symbol = :gain::(_ in (:gain, :split))

end


MLJModelInterface.@mlj_model mutable struct LGBMClassifier <: MLJModelInterface.Probabilistic

    # Hyperparameters, see https://lightgbm.readthedocs.io/en/latest/Parameters.html for defaults
    # Core parameters
    objective::String = "multiclass"::(_ in CLASSIFICATION_OBJECTIVES)
    boosting::String = "gbdt"::(_ in ("gbdt", "goss", "rf", "dart"))
    num_iterations::Int = 100::(_ >= 0)
    learning_rate::Float64 = 0.1::(_ > 0.)
    num_leaves::Int = 31::(1 < _ <= 131072)
    tree_learner::String = "serial"::(_ in ("serial", "feature", "data", "voting"))
    num_threads::Int  = 0::(_ >= 0)
    device_type::String = "cpu"::(_ in ("cpu", "gpu"))
    seed::Int = 0
    deterministic::Bool = false
    
    # Learning control parameters
    force_col_wise::Bool = false
    force_row_wise::Bool = false
    histogram_pool_size::Float64 = -1.0;#::(_ != 0.0);
    max_depth::Int = -1;#::(_ != 0);
    min_data_in_leaf::Int = 20::(_ >= 0)
    min_sum_hessian_in_leaf::Float64 = 1e-3::(_ >= 0.0)
    bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    pos_bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    neg_bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    bagging_freq::Int = 0::(_ >= 0)
    bagging_seed::Int = 3
    feature_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)
    feature_fraction_bynode::Float64 = 1.0::(0.0 < _ <= 1.0)
    feature_fraction_seed::Int = 2
    extra_trees::Bool = false
    extra_seed::Int = 6
    early_stopping_round::Int = 0
    first_metric_only::Bool = false
    max_delta_step::Float64 = 0.0
    lambda_l1::Float64 = 0.0::(_ >= 0.0)
    lambda_l2::Float64 = 0.0::(_ >= 0.0)
    linear_lambda::Float64 = 0.0::(_ >= 0.0)
    min_gain_to_split::Float64 = 0.0::(_ >= 0.0)
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
    max_cat_to_onehot::Int = 4::(_ > 0)
    top_k::Int = 20::(_ > 0)
    monotone_constraints::Vector{Int} = Vector{Int}()
    monotone_constraints_method::String = "basic"::(_ in ("basic", "intermediate", "advanced"))
    monotone_penalty::Float64 = 0.0::(_ >= 0.0)
    feature_contri::Vector{Float64} = Vector{Float64}()
    forcedsplits_filename::String = ""
    refit_decay_rate::Float64 = 0.9::(0.0 <= _ <= 1.0)
    cegb_tradeoff::Float64 = 1.0::(_ >= 0.0)
    cegb_penalty_split::Float64 = 0.0::(_ >= 0.0)
    cegb_penalty_feature_lazy::Vector{Float64} = Vector{Float64}()
    cegb_penalty_feature_coupled::Vector{Float64} = Vector{Float64}()
    path_smooth::Float64 = 0.0::(_ >= 0.0)
    interaction_constraints::Vector{Vector{Int}} = Vector{Vector{Int}}()
    verbosity::Int = 1
    
    # Dateset parameters
    linear_tree::Bool = false
    max_bin::Int = 255::(_ > 1)
    max_bin_by_feature::Vector{Int} = Vector{Int}()
    min_data_in_bin::Int = 3::(_ > 0)
    bin_construct_sample_cnt::Int = 200000::(_ > 0)
    data_random_seed::Int = 1
    is_enable_sparse::Bool = true
    enable_bundle::Bool = true
    use_missing::Bool = true
    zero_as_missing::Bool = false
    feature_pre_filter::Bool = true
    pre_partition::Bool = false
    two_round::Bool = false
    header::Bool = false
    label_column::String = ""   
    weight_column::String = ""
    ignore_column::String  = ""
    categorical_feature::Vector{Int} = Vector{Int}()
    forcedbins_filename::String = ""
    precise_float_parser::Bool = false

    # Predict parameters
    start_iteration_predict::Int = 0
    num_iteration_predict::Int = -1
    predict_raw_score::Bool = false
    predict_leaf_index::Bool = false
    predict_contrib::Bool = false
    predict_disable_shape_check::Bool = false
    pred_early_stop::Bool = false
    pred_early_stop_freq::Int = 10
    pred_early_stop_margin::Float64 = 10.0
    
    # Objective parameters
    is_unbalance::Bool = false
    scale_pos_weight = 1.0
    # For documentation purposes: A calibration scaling factor for the output probabilities for binary and multiclass OVA
    sigmoid::Float64 = 1.0::(_ > 0.0 )
    boost_from_average::Bool = true

    # Metric parameters
    metric::Vector{String} = ["None"]::(all(in.(_, (LGBM_METRICS, ))))
    metric_freq::Int = 1::(_ > 0)
    is_provide_training_metric::Bool = false
    eval_at::Vector{Int} = Vector{Int}([1, 2, 3, 4, 5])::(all(_ .> 0))
    multi_error_top_k::Int = 1::(_ > 0)
    auc_mu_weights::Vector{Float64} = Vector{Float64}()

    # Network parameters
    num_machines::Int = 1::(_ > 0)
    local_listen_port::Int = 12400::(_ > 0)
    time_out::Int = 120::(_ > 0)
    machine_list_filename::String = ""
    machines::String = ""

    # GPU parameters
    gpu_platform_id::Int = -1
    gpu_device_id::Int = -1
    gpu_use_dp::Bool = false
    num_gpu::Int = 1

    # Other (non-lightbm) parameters
    truncate_booster::Bool = true
    
    # Feature importance method for MLJModelInterface.feature_importances
    # :gain - importance based on information gain (default)
    # :split - importance based on number of times feature was used in splits
    feature_importance::Symbol = :gain::(_ in (:gain, :split))

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


# Build dataset parameters string. For LGBMClassifier, num_class is required for correct
# dataset creation (and for LGBM_DatasetGetSubset during resampling). When classes are
# provided, they are used; otherwise the estimator default (num_class=2) is used.
function dataset_params(mlj_model::LGBMClassifier)
    estimator = LightGBM.LGBMClassification(; mlj_to_kwargs(mlj_model)...)
    return LightGBM.stringifyparams(estimator)
end
function dataset_params(mlj_model::LGBMClassifier, classes)
    estimator = LightGBM.LGBMClassification(; mlj_to_kwargs(mlj_model, classes)...)
    return LightGBM.stringifyparams(estimator)
end
function dataset_params(mlj_model::LGBMRegressor)
    estimator = LightGBM.LGBMRegression(; mlj_to_kwargs(mlj_model)...)
    return LightGBM.stringifyparams(estimator)
end

function build_frontend_data(mlj_model::MODELS, X)
    matrix = MLJModelInterface.matrix(X)
    feature_names = extract_feature_names(X)
    ds_params = dataset_params(mlj_model)
    dataset = LightGBM.dataset_constructor(matrix, ds_params, false)
    if feature_names !== nothing
        LightGBM.LGBM_DatasetSetFeatureNames(dataset, collect(String.(feature_names)))
    end
    return LGBMFrontEndData(matrix, dataset, feature_names, ds_params, nothing)
end

# Classifier: when y is available, include num_class in dataset params for correct dataset/subset creation.
function build_frontend_data(mlj_model::LGBMClassifier, X, y)
    matrix = MLJModelInterface.matrix(X)
    feature_names = extract_feature_names(X)
    classes = MLJModelInterface.classes(first(y))
    ds_params = dataset_params(mlj_model, classes)
    dataset = LightGBM.dataset_constructor(matrix, ds_params, false)
    if feature_names !== nothing
        LightGBM.LGBM_DatasetSetFeatureNames(dataset, collect(String.(feature_names)))
    end
    return LGBMFrontEndData(matrix, dataset, feature_names, ds_params, nothing)
end

"""
    MLJModelInterface.reformat(model, X)
    MLJModelInterface.reformat(model, X, y)
    MLJModelInterface.reformat(model, X, y, w)

Construct LightGBM front-end data for MLJ. Always returns a tuple.
For LGBMRegressor, `reformat(model, X, y)[1]` equals `reformat(model, X)[1]`.
For LGBMClassifier, when y is provided the first element is built with num_class from y
so that dataset params are correct for resampling and LGBM_DatasetGetSubset; otherwise
the default num_class is used.
"""
function MLJModelInterface.reformat(model::MODELS, X)
    data = build_frontend_data(model, X)
    return (data,)
end
function MLJModelInterface.reformat(model::MODELS, X, y)
    data = MLJModelInterface.reformat(model, X)[1]
    return (data, y)
end
function MLJModelInterface.reformat(model::MODELS, X, y, w)
    data = MLJModelInterface.reformat(model, X)[1]
    return (data, y, w)
end
# LGBMClassifier with y: use build_frontend_data(model, X, y) so dataset_params include num_class.
function MLJModelInterface.reformat(model::LGBMClassifier, X, y)
    data = build_frontend_data(model, X, y)
    return (data, y)
end
function MLJModelInterface.reformat(model::LGBMClassifier, X, y, w)
    data = build_frontend_data(model, X, y)
    return (data, y, w)
end


# MLJBase uses the result as predict(model, fitresult, selectrows(...)...), so we return
# a tuple so that splatting yields the correct number of arguments (one for X, or (X, y) etc.).
function MLJModelInterface.selectrows(::MODELS, rows, data::LGBMFrontEndData)
    return (selectrows_lgbm(data, rows),)
end
function MLJModelInterface.selectrows(::MODELS, rows, data::LGBMFrontEndData, y)
    return (selectrows_lgbm(data, rows), MLJModelInterface.selectrows(y, rows))
end
function MLJModelInterface.selectrows(::MODELS, rows, data::LGBMFrontEndData, y, w)
    return (
        selectrows_lgbm(data, rows),
        MLJModelInterface.selectrows(y, rows),
        MLJModelInterface.selectrows(w, rows),
    )
end

# Internal: subset LGBMFrontEndData by row indices.
# LightGBM's LGBM_DatasetGetSubset requires used_row_indices to be sorted; we sort when
# necessary and store row_invperm so predictions can be reordered to match the requested order.
function selectrows_lgbm(data::LGBMFrontEndData, rows)
    if rows isa Colon || rows isa Function
        return data
    end
    row_indices = collect(rows)
    if issorted(row_indices)
        sorted_indices = row_indices
        row_invperm = nothing
    else
        perm = sortperm(row_indices)
        sorted_indices = row_indices[perm]
        row_invperm = invperm(perm)
    end
    subset_dataset = LightGBM.LGBM_DatasetGetSubset(data.dataset, sorted_indices, data.dataset_params)
    if data.feature_names !== nothing
        LightGBM.LGBM_DatasetSetFeatureNames(subset_dataset, collect(String.(data.feature_names)))
    end
    matrix_view = @view data.matrix[sorted_indices, :]
    return LGBMFrontEndData(matrix_view, subset_dataset, data.feature_names, data.dataset_params, row_invperm)
end

# selectrows(data, rows) / selectrows(tuple, rows) — for direct use (e.g. tests)
function MLJModelInterface.selectrows(data::LGBMFrontEndData, rows)
    return selectrows_lgbm(data, rows)
end
function MLJModelInterface.selectrows(data_tuple::Tuple{LGBMFrontEndData}, rows)
    data = data_tuple[1]
    return (MLJModelInterface.selectrows(data, rows),)
end
function MLJModelInterface.selectrows(data_tuple::Tuple{LGBMFrontEndData, Any}, rows)
    data, y = data_tuple
    return (MLJModelInterface.selectrows(data, rows), MLJModelInterface.selectrows(y, rows))
end
function MLJModelInterface.selectrows(data_tuple::Tuple{LGBMFrontEndData, Any, Any}, rows)
    data, y, w = data_tuple
    return (
        MLJModelInterface.selectrows(data, rows),
        MLJModelInterface.selectrows(y, rows),
        MLJModelInterface.selectrows(w, rows),
    )
end

# X and y and w must be untyped per MLJ docs
function fit(mlj_model::MODELS, verbosity::Int, X, y, w=AbstractFloat[])

    # MLJ docs: 0 means silent. LightGBM: 0 = warnings, <0 = fatal only. Use -1 for silence.
    # Respect both fit!(verbosity) and the model's verbosity hyperparameter (model wins if either is 0).
    lightgbm_verbosity = (verbosity == 0 || mlj_model.verbosity == 0) ? -1 : verbosity

    y_lgbm, classes = prepare_targets(y, mlj_model)
    model = model_init(mlj_model, classes)
    # Booster is created with stringifyparams(model); C API reads verbosity from there and prints warnings.
    model.verbosity = lightgbm_verbosity

    feature_names = extract_feature_names(X)

    X = MLJModelInterface.matrix(X)
    # The FFI wrapper wants Float32 for these
    w = Float32.(w)
    # slice of y_lgbm required to converts it from a SubArray to a copy of an actual Array
    train_results = LightGBM.fit!(model, X, y_lgbm[:]; verbosity=lightgbm_verbosity, weights=w, truncate_booster=mlj_model.truncate_booster)

    fitresult = (model, classes, deepcopy(mlj_model), feature_names)
    # because update needs access to the older version of training metrics we keep them in the cache
    # so the update can merge old and additional metrics as necessary.
    cache = (
        num_boostings_done=[LightGBM.get_iter_number(model)],
        training_metrics=train_results["metrics"],
    )
    report = user_fitreport(model, train_results)

    return (fitresult, cache, report)

end

function fit(mlj_model::MODELS, verbosity::Int, data::LGBMFrontEndData, y, w=AbstractFloat[])
    lightgbm_verbosity = (verbosity == 0 || mlj_model.verbosity == 0) ? -1 : verbosity

    y_lgbm, classes = prepare_targets(y, mlj_model)
    model = model_init(mlj_model, classes)
    # Booster is created with stringifyparams(model); C API reads verbosity from there.
    model.verbosity = lightgbm_verbosity

    if length(w) > 0
        w = Float32.(w)
        LightGBM.LGBM_DatasetSetField(data.dataset, "weight", w)
    end
    LightGBM.LGBM_DatasetSetField(data.dataset, "label", y_lgbm[:])

    train_results = LightGBM.fit!(
        model,
        data.dataset;
        verbosity=lightgbm_verbosity,
        truncate_booster=mlj_model.truncate_booster,
    )

    fitresult = (model, classes, deepcopy(mlj_model), data.feature_names)
    cache = (
        num_boostings_done=[LightGBM.get_iter_number(model)],
        training_metrics=train_results["metrics"],
    )
    report = user_fitreport(model, train_results)

    return (fitresult, cache, report)
end


function update(mlj_model::MLJInterface.MODELS, verbosity::Int, fitresult, cache, X, y, w=AbstractFloat[])
    lightgbm_verbosity = (verbosity == 0 || mlj_model.verbosity == 0) ? -1 : verbosity

    old_lgbm_model, old_classes, old_mlj_model, feature_names = fitresult

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
        lightgbm_verbosity,
        LightGBM.Dates.now();
        truncate_booster=old_mlj_model.truncate_booster
    )
    fitresult = (old_lgbm_model, old_classes, deepcopy(mlj_model), feature_names)

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


function predict_classifier((fitted_model, classes, _, _), Xnew)
    row_invperm = Xnew isa LGBMFrontEndData ? Xnew.row_invperm : nothing
    Xnew = Xnew isa LGBMFrontEndData ? Matrix(Xnew.matrix) : MLJModelInterface.matrix(Xnew)
    predicted = LightGBM.predict(fitted_model, Xnew)
    # when the objective == binary, lightgbm internally has classes = 1 and spits out only probability of positive class
    if size(predicted, 2) == 1
        predicted = hcat(1. .- predicted, predicted)
    end
    if row_invperm !== nothing
        predicted = predicted[row_invperm, :]
    end
    return MLJModelInterface.UnivariateFinite(classes, predicted)
end


function predict_regression((fitted_model, classes, _, _), Xnew)
    row_invperm = Xnew isa LGBMFrontEndData ? Xnew.row_invperm : nothing
    Xnew = Xnew isa LGBMFrontEndData ? Matrix(Xnew.matrix) : MLJModelInterface.matrix(Xnew)
    predicted = dropdims(LightGBM.predict(fitted_model, Xnew), dims=2)
    if row_invperm !== nothing
        predicted = predicted[row_invperm]
    end
    return predicted
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


# Helper function to get feature names from a fitted model.
# 
# Requires a fitted model - the booster handle must be initialized (not C_NULL).
# Once initialized, LightGBM always provides feature names - either custom names
# (if set via LGBM_DatasetSetFeatureNames) or default names like "Column_0", "Column_1", etc.
function get_feature_names(fitted_model::LightGBM.LGBMEstimator)
    # Prevent segfault by checking if booster is initialized
    if fitted_model.booster.handle == C_NULL
        throw(ErrorException("Estimator does not contain a fitted model."))
    end
    
    return LightGBM.LGBM_BoosterGetFeatureNames(fitted_model.booster)
end

# Implementation of feature_importances for MLJModelInterface
function MLJModelInterface.feature_importances(model::MODELS, fitresult, report)
    fitted_model, _, _, feature_names = fitresult
    
    # Get the appropriate importance values based on the model's hyperparameter
    importance_values = if model.feature_importance == :gain
        LightGBM.gain_importance(fitted_model)
    elseif model.feature_importance == :split
        LightGBM.split_importance(fitted_model)
    else
        error("Unsupported feature importance method: $(model.feature_importance)")
    end
    
    # If feature_names is nothing (X was a plain matrix), fall back to LightGBM's internal names
    # Otherwise use the original table column names for compatibility with MLJ tools
    if feature_names === nothing
        lgbm_names = get_feature_names(fitted_model)
        return [Symbol(name) => Float64(importance) for (name, importance) in zip(lgbm_names, importance_values)]
    else
        return [name => Float64(importance) for (name, importance) in zip(feature_names, importance_values)]
    end
end


# Set the trait to indicate that both models support feature importances
MLJModelInterface.reports_feature_importances(::Type{<:LGBMClassifier}) = true
MLJModelInterface.reports_feature_importances(::Type{<:LGBMRegressor}) = true


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

include("docstrings.jl")

end # module
