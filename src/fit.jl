"""
    fit!(estimator, num_iterations, X, y[, test...]; [verbosity = 1, is_row_major = false])
    fit!(estimator, X, y[, test...]; [verbosity = 1, is_row_major = false])
    fit!(estimator, X, y, train_indices[, test_indices...]; [verbosity = 1, is_row_major = false])
    fit!(estimator, train_dataset[, test_datasets...]; [verbosity = 1])

Fit the `estimator` with features data `X` and label `y` using the X-y pairs in `test` as
validation sets.
Alternatively, Fit the `estimator` with `train_dataset` and `test_datasets` in the form of Dataset class(es)

Return a dictionary with an entry for each validation set. Each entry of the dictionary is another
dictionary with an entry for each validation metric in the `estimator`. Each of these entries is an
array that holds the validation metric's value at each iteration.

# Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `test::Tuple{Matrix{TX},Vector{Ty}}...`: optionally contains one or more tuples of X-y pairs of
    the same types as `X` and `y` that should be used as validation sets.
* `train_dataset::Dataset`: prepared train_dataset
* `test_datasets::Vector{Dataset}`: (optional) prepared test_datasets
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
* `is_row_major::Bool`: keyword argument that indicates whether or not `X` is row-major. `true`
    indicates that it is row-major, `false` indicates that it is column-major (Julia's default).
* `weights::Vector{Tw<:Real}`: the training weights.
* `init_score::Vector{Ti<:Real}`: the init scores.
"""
function fit!(
    estimator::LGBMEstimator, X::Matrix{TX}, y::Vector{Ty}, test::Tuple{Matrix{TX},Vector{Ty}}...;
    verbosity::Integer = 1,
    is_row_major = false,
    weights::Vector{Tw} = Float32[],
    init_score::Vector{Ti} = Float64[],
) where {TX<:Real,Ty<:Real,Tw<:Real,Ti<:Real}

    start_time = now()

    log_debug(verbosity, "Started creating LGBM training dataset\n")
    ds_parameters = stringifyparams(estimator, DATASETPARAMS)
    train_ds = LGBM_DatasetCreateFromMat(X, ds_parameters, is_row_major)
    LGBM_DatasetSetField(train_ds, "label", y)
    if length(weights) > 0
        LGBM_DatasetSetField(train_ds, "weight", weights)
    end
    if length(init_score) > 0
        LGBM_DatasetSetField(train_ds, "init_score", init_score)
    end

    test_dss = []

    for test_entry in test
        test_ds = LGBM_DatasetCreateFromMat(test_entry[1], ds_parameters, train_ds, is_row_major)
        LGBM_DatasetSetField(test_ds, "label", test_entry[2])
        push!(test_dss, test_ds)
    end

    return fit!(estimator, train_ds, test_dss..., verbosity=verbosity)
end


# Pass Dataset class directly. This will speed up the process if it is part of an iterative process and pre-constructed dataset(s) are available
function fit!(
    estimator::LGBMEstimator, 
    train_dataset::Dataset, 
    test_datasets::Dataset...;
    verbosity::Integer = 1,
)

    start_time = now()
    log_debug(verbosity, "Started creating LGBM booster\n")
    bst_parameters = stringifyparams(estimator, BOOSTERPARAMS) * " verbosity=$verbosity"
    estimator.booster = LGBM_BoosterCreate(train_dataset, bst_parameters)

    n_tests = length(test_datasets)
    tests_names = Array{String}(undef,n_tests)

    for (testset_enum, test_dataset) in enumerate(test_datasets)
        tests_names[testset_enum] = "test_$(testset_enum)"
        LGBM_BoosterAddValidData(estimator.booster, test_dataset)
    end

    log_debug(verbosity, "Started training...\n")
    results = train!(estimator, tests_names, verbosity, start_time)

    return results
end


function train!(
    estimator::LGBMEstimator, 
    num_iterations::Int, 
    tests_names::Vector{String}, 
    verbosity::Integer, 
    start_time::DateTime
)
    results = Dict{String,Dict{String,Vector{Float64}}}()
    metrics = LGBM_BoosterGetEvalNames(estimator.booster)

    bigger_is_better = Dict(metric => ifelse(in(metric, MAXIMIZE_METRICS), 1., -1.) for metric in metrics)
    best_score = Dict{String,Dict{String,Real}}()
    best_iter = Dict{String,Dict{String,Real}}()

    for metric in metrics
        best_score[metric] = Dict{String,Real}()
        best_iter[metric] = Dict{String,Real}()
        for tests_name in tests_names
            best_score[metric][tests_name] = -Inf
            best_iter[metric][tests_name] = 1
        end
    end

    start_iter = get_iter_number(estimator) + 1
    end_iter = start_iter + num_iterations - 1

    metrics_idx_sequence = (((start_iter:end_iter) .- 1) .% estimator.metric_freq) .== 0
    total_metrics_evals = sum(metrics_idx_sequence)

    for (idx, iter) in enumerate(start_iter:end_iter)

        is_finished = LGBM_BoosterUpdateOneIter(estimator.booster)

        log_debug(verbosity, Dates.CompoundPeriod(now() - start_time), " elapsed, finished iteration ", iter, "\n")

        if is_finished == 0
            is_finished = eval_metrics!(
                results, estimator, tests_names, iter, verbosity, 
                bigger_is_better, best_score, best_iter, metrics,
            )
        end

        if is_finished == 1
            break
        end
    end

    # save the model in serialised form, in case we should be deepcopied or serialised elsewhere
    estimator.model = LGBM_BoosterSaveModelToString(estimator.booster)

    return results
end
# Old signature, pass through args
function train!(
    estimator::LGBMEstimator, tests_names::Vector{String}, verbosity::Integer, start_time::DateTime
)
    return train!(estimator, estimator.num_iterations, tests_names, verbosity, start_time)
end


function eval_metrics!(
    results::Dict{String,Dict{String,Vector{Float64}}},
    estimator::LGBMEstimator,
    tests_names::Vector{String},
    iter::Integer,
    verbosity::Integer,
    bigger_is_better::Dict{String,Float64},
    best_score::Dict{String,Dict{String,Real}},
    best_iter::Dict{String,Dict{String,Real}},
    metrics::Vector{String},
)
    now_scores = Dict{String,Vector{Float64}}()

    if (iter - 1) % estimator.metric_freq == 0
        if estimator.is_training_metric
            now_scores["training"] = LGBM_BoosterGetEval(estimator.booster, 0)
        end
        for (test_idx, tests_name) in enumerate(tests_names)
            now_scores[tests_name] = LGBM_BoosterGetEval(estimator.booster, test_idx)
        end
    end

    # check early stopping condition
    if estimator.early_stopping_round > 0
        for tests_name in tests_names
            for (metric_idx, metric) in enumerate(metrics)
                maximize_score = bigger_is_better[metric] * now_scores[tests_name][metric_idx]
                # All good if maximize_score is better than previous best
                if maximize_score > best_score[metric][tests_name]
                    best_score[metric][tests_name] = maximize_score
                    best_iter[metric][tests_name] = iter
                    continue
                # All good if difference between current and best iter is within early_stopping_round
                elseif (iter - best_iter[metric][tests_name]) < estimator.early_stopping_round
                    continue
                end                
                log_info(verbosity, "Early stopping at iteration ", iter, ", the best iteration round is ", best_iter[metric][tests_name], "\n")
                return 1
            end
        
        end
    end

    for (metric_idx, metric_name) in enumerate(metrics)
	    for dataset_key in keys(now_scores)
            store_scores!(results, dataset_key, metric_name, now_scores[dataset_key][metric_idx])

            # print scores
            log_info(verbosity, "Iteration: ", iter, ", ", dataset_key, "'s ")
            log_info(verbosity, metric_name, ": ", now_scores[dataset_key][metric_idx])
                metric_idx < length(metrics) && log_info(verbosity, ", ")
            log_info(verbosity, "\n")
        
        end
    end
    return 0
end

function store_scores!(
    results::Dict{String,Dict{String,Vector{Float64}}},
    dataset_key::String,
    metric_name::String,
    value_to_add::Real,
)
    if !haskey(results, dataset_key)
        results[dataset_key] = Dict{String,Vector{Float64}}()
        results[dataset_key][metric_name] = Vector{Float64}()
    elseif !haskey(results[dataset_key], metric_name)
        results[dataset_key][metric_name] = Vector{Float64}()
    end
    push!(results[dataset_key][metric_name], value_to_add)
end


function merge_scores(
    old_scores::Dict{String,Dict{String,Vector{Float64}}},
    additional_scores::Dict{String,Dict{String,Vector{Float64}}},
)

    if keys(old_scores) != keys(additional_scores)
        throw(ErrorException("Tried to merge metrics with different data sets:\n    a=> $(keys(old_scores))\n    b=> $(keys(additional_scores))"))
    end

    new_scores = Dict{String,Dict{String,Vector{Float64}}}()
    for (key, oldvals) in old_scores
        new_scores[key] = merge_scores(old_scores[key], additional_scores[key])
    end

    return new_scores
end


function merge_scores(
    old_scores::Dict{String,Vector{Float64}},
    additional_scores::Dict{String,Vector{Float64}},
)

    if keys(old_scores) != keys(additional_scores)
        throw(ErrorException("Tried to merge metrics with different metrics:\n    a=> $(keys(old_scores))\n    b=> $(keys(additional_scores))"))
    end

    new_scores = Dict{String,Vector{Float64}}()
    for (key, oldvals) in old_scores
        new_scores[key] = cat(oldvals, additional_scores[key]; dims=1)
    end

    return new_scores
end


function stringifyparams(estimator::LGBMEstimator, params::Vector{Symbol})
    paramstring = ""
    n_params = length(params)
    valid_names = fieldnames(typeof(estimator))
    for (param_idx, param_name) in enumerate(params)
        if in(param_name, valid_names)
            param_value = getfield(estimator, param_name)
            if !isempty(param_value)
                # Convert parameters that contain indices to C's zero-based indices.
                if in(param_name, INDEXPARAMS)
                    param_value .-= 1
                end

                if typeof(param_value) <: Array
                    n_entries = length(param_value)
                    if n_entries >= 1
                        paramstring = string(paramstring, param_name, "=", param_value[1])
                        for entry_idx in 2:n_entries
                            paramstring = string(paramstring, ",", param_value[entry_idx])
                        end
                        paramstring = string(paramstring, " ")
                    end
                else
                    paramstring = string(paramstring, param_name, "=", param_value, " ")
                end
            end
        end
    end
    return paramstring[1:end - 1]
end
