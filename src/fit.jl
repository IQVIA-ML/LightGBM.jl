# Currently unsupported DS parameters:
# label, ignore_column, weight_column, group_column, has_header, bin_construct_sample_cnt
const datasetparams = [:is_pre_partition, :num_class, :two_round, :is_sparse, :max_bin,
                       :data_random_seed]

const boosterparams = [:application, :learning_rate, :num_leaves, :max_depth, :tree_learner,
                       :num_threads, :histogram_pool_size, :min_data_in_leaf,
                       :min_sum_hessian_in_leaf, :lambda_l1, :lambda_l2, :min_gain_to_split,
                       :feature_fraction, :feature_fraction_seed, :bagging_fraction,
                       :bagging_freq, :bagging_seed, :early_stopping_round, :is_sigmoid,
                       :sigmoid, :is_unbalance, :max_position, :metric,
                       :is_training_metric, :ndcg_at, :num_machines, :local_listen_port,
                       :time_out, :machine_list_file]

const maximize_metrics = ["auc", "ndcg"]

# Fit the `estimator` using the CLI of LightGBM.
function api_fit{TX<:Real,Ty<:Real}(estimator::LGBMEstimator, X::Matrix{TX}, y::Vector{Ty},
                                    test::Tuple{Matrix{TX},Vector{Ty}}...; verbosity::Integer = 1)
    start_time = now()

    log_debug(verbosity, "Started creating LGBM training dataset\n")
    ds_parameters = getparamstring(estimator, datasetparams)
    train_ds = LGBM_CreateDatasetFromMat(X, ds_parameters)
    LGBM_DatasetSetField(train_ds, "label", y)

    n_tests = length(test)
    if n_tests > 0
        log_debug(verbosity, "Started creating LGBM test datasets\n")
        tests_ds = Array(Dataset, n_tests)
        tests_names = ["test_$(test_idx)" for test_idx in 1:n_tests]
        @inbounds for (test_idx, test_entry) in enumerate(test)
            test_ds = LGBM_CreateDatasetFromMat(test_entry[1], ds_parameters, train_ds)
            LGBM_DatasetSetField(test_ds, "label", test_entry[2])
            tests_ds[test_idx] = test_ds
        end
    end

    log_debug(verbosity, "Started creating LGBM booster\n")
    bst_parameters = getparamstring(estimator, boosterparams) * " verbosity=$verbosity"
    estimator.booster = LGBM_BoosterCreate(train_ds, tests_ds, tests_names, bst_parameters)

    log_debug(verbosity, "Started training...\n")
    results = fit_train(estimator, tests_names, verbosity, start_time)
    # estimator.model = readlines("$(tempdir)/model.txt")

    return results
end

function fit_train(estimator::LGBMEstimator, tests_names::Vector{String}, verbosity::Integer,
                   start_time::DateTime)
    results = Dict{String,Dict{String,Vector{Float64}}}()
    n_tests = length(tests_names)
    n_metrics = length(estimator.metric)
    bigger_is_better = [ifelse(in(metric, maximize_metrics), 1., -1.) for metric in estimator.metric]
    best_score = fill(-Inf, (n_metrics, n_tests))
    best_iter = fill(1, (n_metrics, n_tests))

    for iter in 1:estimator.num_iterations
        is_finished = LGBM_BoosterUpdateOneIter(estimator.booster)
        log_debug(verbosity, Base.Dates.CompoundPeriod(now() - start_time),
                  " elapsed, finished iteration ", iter, "\n")
        if is_finished == 0
            is_finished = eval_metrics!(results, estimator, tests_names, iter, n_tests, n_metrics,
                                        verbosity, bigger_is_better, best_score, best_iter)
        else
            log_info(verbosity, "Stopped training because there are no more leaves that meet the ",
                     "split requirements.")
        end
        is_finished == 1 && return results
    end
    return results
end

function eval_metrics!(results::Dict{String,Dict{String,Vector{Float64}}},
                       estimator::LGBMEstimator, tests_names::Vector{String}, iter::Integer,
                       n_tests::Integer, n_metrics::Integer, verbosity::Integer,
                       bigger_is_better::Vector{Float64}, best_score::Matrix{Float64},
                       best_iter::Matrix{Int})
    if (iter - 1) % estimator.metric_freq == 0
        if estimator.is_training_metric
            scores = LGBM_BoosterEval(estimator.booster, 0, n_metrics)
            store_scores!(results, estimator, iter, "training", n_metrics, scores)
            print_scores(estimator, iter, "training", n_metrics, scores, verbosity)
        end
    end

    if (iter - 1) % estimator.metric_freq == 0 || estimator.early_stopping_round > 0
        for (test_idx, test_name) in enumerate(tests_names)
            scores = LGBM_BoosterEval(estimator.booster, test_idx, n_metrics)

            # Check if progress should be stored and/or printed
            if (iter - 1) % estimator.metric_freq == 0
                store_scores!(results, estimator, iter, test_name, n_metrics, scores)
                print_scores(estimator, iter, test_name, n_metrics, scores, verbosity)
            end

            # Check if early stopping is called for
            @inbounds for metric_idx in eachindex(estimator.metric)
                maximize_score = bigger_is_better[metric_idx] * scores[metric_idx]
                if maximize_score > best_score[metric_idx, test_idx]
                    best_score[metric_idx, test_idx] = maximize_score
                    best_iter[metric_idx, test_idx] = iter
                elseif iter - best_iter[metric_idx, test_idx] >= estimator.early_stopping_round
                    shrinkresults!(results, best_iter[metric_idx, test_idx])
                    log_info(verbosity, "Early stopping at iteration ", iter,
                             ", the best iteration round is ", best_iter[metric_idx, test_idx], "\n")
                    return 1
                end
            end
        end
    end

    return 0
end

function store_scores!(results::Dict{String,Dict{String,Vector{Float64}}},
                       estimator::LGBMEstimator, iter::Integer, evalname::String,
                       n_metrics::Integer, scores::Vector{Cfloat})
    for (metric_idx, metric_name) in enumerate(estimator.metric)
        if !haskey(results, evalname)
            num_evals = cld(estimator.num_iterations, estimator.metric_freq)
            results[evalname] = Dict{String,Vector{Float64}}()
            results[evalname][metric_name] = Array(Float64, num_evals)
        elseif !haskey(results[evalname], metric_name)
            num_evals = cld(estimator.num_iterations, estimator.metric_freq)
            results[evalname][metric_name] = Array(Float64, num_evals)
        end
        eval_idx = cld(iter, estimator.metric_freq)
        results[evalname][metric_name][eval_idx] = scores[metric_idx]
    end

    return nothing
end

function print_scores(estimator::LGBMEstimator, iter::Integer, name::String, n_metrics::Integer,
                      scores::Vector{Cfloat}, verbosity::Integer)
    log_info(verbosity, "Iteration: ", iter, ", ", name, "'s ")
    for (metric_idx, metric_name) in enumerate(estimator.metric)
        log_info(verbosity, metric_name, ": ", scores[metric_idx])
        metric_idx < n_metrics && log_info(verbosity, ", ")
    end
    log_info(verbosity, "\n")
end

function getparamstring(estimator::LGBMEstimator, params::Vector{Symbol})
    paramstring = ""
    n_params = length(params)
    for (param_idx, param_name) in enumerate(params)
        param_value = getfield(estimator, param_name)
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
    return paramstring[1:end-1]
end
