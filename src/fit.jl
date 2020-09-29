"""
    fit!(estimator, num_iterations, X, y[, test...]; [verbosity = 1, is_row_major = false])
    fit!(estimator, X, y[, test...]; [verbosity = 1, is_row_major = false])

Fit the `estimator` with features data `X` and label `y` using the X-y pairs in `test` as
validation sets.

Return a dictionary with an entry for each validation set. Each entry of the dictionary is another
dictionary with an entry for each validation metric in the `estimator`. Each of these entries is an
array that holds the validation metric's value at each iteration.

# Arguments
* `estimator::LGBMEstimator`: the estimator to be fit.
* `num_iterations::Int`: OPTIONAL -- defaults to estimator.num_iterations if not provided
* `X::Matrix{TX<:Real}`: the features data.
* `y::Vector{Ty<:Real}`: the labels.
* `test::Tuple{Matrix{TX},Vector{Ty}}...`: optionally contains one or more tuples of X-y pairs of
    the same types as `X` and `y` that should be used as validation sets.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
* `is_row_major::Bool`: keyword argument that indicates whether or not `X` is row-major. `true`
    indicates that it is row-major, `false` indicates that it is column-major (Julia's default).
* `weights::Vector{Tw<:Real}`: the training weights.
* `init_score::Vector{Ti<:Real}`: the init scores.
"""
function fit!(
    estimator::LGBMEstimator, num_iterations::Int, X::Matrix{TX}, y::Vector{Ty}, test::Tuple{Matrix{TX},Vector{Ty}}...;
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

    log_debug(verbosity, "Started creating LGBM booster\n")
    bst_parameters = stringifyparams(estimator, BOOSTERPARAMS) * " verbosity=$verbosity"
    estimator.booster = LGBM_BoosterCreate(train_ds, bst_parameters)

    n_tests = length(test)
    tests_names = Array{String}(undef,n_tests)

    if n_tests > 0
        log_debug(verbosity, "Started creating LGBM test datasets\n")
        @inbounds for (test_idx, test_entry) in enumerate(test)
            tests_names[test_idx] = "test_$(test_idx)"
            test_ds = LGBM_DatasetCreateFromMat(test_entry[1], ds_parameters, train_ds, is_row_major)
            LGBM_DatasetSetField(test_ds, "label", test_entry[2])
            LGBM_BoosterAddValidData(estimator.booster, test_ds)
        end
    end

    log_debug(verbosity, "Started training...\n")
    results = train!(estimator, tests_names, verbosity, start_time)

    return results
end
# Old signature, pass through args
function fit!(
    estimator::LGBMEstimator,
    X::Matrix{TX},
    y::Vector{Ty},
    test::Tuple{Matrix{TX},Vector{Ty}}...;
    kwargs...
)  where {TX<:Real,Ty<:Real}
    return fit!(estimator, estimator.num_iterations, X, y, test...; kwargs...)
end


function train!(
    estimator::LGBMEstimator, num_iterations::Int, tests_names::Vector{String}, verbosity::Integer, start_time::DateTime
)
    results = Dict{String,Dict{String,Vector{Float64}}}()
    n_tests = length(tests_names)
    metrics = LGBM_BoosterGetEvalNames(estimator.booster)
    n_metrics = length(metrics)
    bigger_is_better = [ifelse(in(metric, MAXIMIZE_METRICS), 1., -1.) for metric in metrics]
    best_score = fill(-Inf, (n_metrics, n_tests))
    best_iter = fill(1, (n_metrics, n_tests))

    start_iter = get_iter_number(estimator) + 1
    end_iter = start_iter + num_iterations - 1

    metrics_idx_sequence = (((start_iter:end_iter) .- 1) .% estimator.metric_freq) .== 0
    total_metrics_evals = sum(metrics_idx_sequence)
    metric_idx = 0

    for (idx, iter) in enumerate(start_iter:end_iter)

        is_finished = LGBM_BoosterUpdateOneIter(estimator.booster)

        log_debug(verbosity, Dates.CompoundPeriod(now() - start_time), " elapsed, finished iteration ", iter, "\n")

        metric_idx = metric_idx + metrics_idx_sequence[idx]

        if is_finished == 0
            is_finished = eval_metrics!(
                results, estimator, tests_names, iter, metric_idx, total_metrics_evals, n_metrics,
                verbosity, bigger_is_better, best_score, best_iter, metrics,
            )
        else
            shrinkresults!(results, metric_idx)
            log_info(verbosity, "Stopped training because there are no more leaves that meet the ",
                     "split requirements.")
        end
        if is_finished == 1
            break
        end
    end

    # save the model in serialised form, in case we should be deepcopied or serialised elsewhere
    estimator.model = LGBM_BoosterSaveModelToString(estimator.booster, 0, 0, 0)

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
    metrics_store_idx::Integer,
    metrics_stored_count::Integer,
    n_metrics::Integer,
    verbosity::Integer,
    bigger_is_better::Vector{Float64},
    best_score::Matrix{Float64},
    best_iter::Matrix{Int},
    metrics::Vector{String},
)

    if (iter - 1) % estimator.metric_freq == 0
        if estimator.is_training_metric
            scores = LGBM_BoosterGetEval(estimator.booster, 0)
            store_scores!(results, estimator, metrics_store_idx, metrics_stored_count, "training", scores, metrics)
            print_scores(estimator, iter, "training", n_metrics, scores, metrics, verbosity)
        end
    end

    # Metrics for test sets
    if (iter - 1) % estimator.metric_freq == 0 || estimator.early_stopping_round > 0
        for (test_idx, test_name) in enumerate(tests_names)
            scores = LGBM_BoosterGetEval(estimator.booster, test_idx)
            # Check if progress should be stored and/or printed
            if (iter - 1) % estimator.metric_freq == 0
                store_scores!(results, estimator, metrics_store_idx, metrics_stored_count, test_name, scores, metrics)
                print_scores(estimator, iter, test_name, n_metrics, scores, metrics, verbosity)
            end

            # Check if early stopping is called for
            @inbounds for metric_idx in eachindex(metrics)
                maximize_score = bigger_is_better[metric_idx] * scores[metric_idx]
                if maximize_score > best_score[metric_idx, test_idx]
                    best_score[metric_idx, test_idx] = maximize_score
                    best_iter[metric_idx, test_idx] = iter
                elseif iter - best_iter[metric_idx, test_idx] >= estimator.early_stopping_round
                    # This will shrink it up to the current stored metric
                    shrinkresults!(results, metrics_store_idx)
                    log_info(verbosity, "Early stopping at iteration ", iter,
                             ", the best iteration round is ", best_iter[metric_idx, test_idx], "\n")
                    return 1
                end
            end
        end
    end

    return 0
end


function store_scores!(
    results::Dict{String,Dict{String,Vector{Float64}}},
    estimator::LGBMEstimator,
    store_idx::Integer,
    num_evals::Integer,
    evalname::String,
    scores::Vector{Cdouble},
    metrics::Vector{String},
)

    for (metric_idx, metric_name) in enumerate(metrics)
        if !haskey(results, evalname)
            results[evalname] = Dict{String,Vector{Float64}}()
            results[evalname][metric_name] = Array{Float64}(undef,num_evals)
        elseif !haskey(results[evalname], metric_name)
            results[evalname][metric_name] = Array{Float64}(undef,num_evals)
        end
        results[evalname][metric_name][store_idx] = scores[metric_idx]
    end

    return nothing
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



function print_scores(estimator::LGBMEstimator, iter::Integer, name::String, n_metrics::Integer,
                      scores::Vector{Cdouble}, metrics::Vector{String}, verbosity::Integer)
    log_info(verbosity, "Iteration: ", iter, ", ", name, "'s ")
    for (metric_idx, metric_name) in enumerate(metrics)
        log_info(verbosity, metric_name, ": ", scores[metric_idx])
        metric_idx < n_metrics && log_info(verbosity, ", ")
    end
    log_info(verbosity, "\n")
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
                    param_value -= 1
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
