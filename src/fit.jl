# DS parameters:
# label
# ignore_column
# weight_column
# group_column
# has_header
# bin_construct_sample_cnt
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

# Fit the `estimator` using the CLI of LightGBM.
function api_fit{TX<:Real,Ty<:Real}(estimator::LGBMEstimator, X::Array{TX,2}, y::Array{Ty,1},
                                    test::Tuple{Array{TX,2},Array{Ty,1}}...;
                                    verbosity::Integer = 1)
    ds_parameters = getparamstring(estimator, datasetparams)
    bst_parameters = getparamstring(estimator, boosterparams)
    n_metrics = length(estimator.metric)

    # Init train dataset
    train_ds = LGBM_CreateDatasetFromMat(X, ds_parameters)
    LGBM_DatasetSetField(train_ds, "label", y)

    # Init test datasets
    n_tests = length(test)
    tests_ds = Array(DatasetHandle, n_tests)
    tests_names = Array(String, n_tests)
    for test_idx in 1:n_tests
        test_ds = LGBM_CreateDatasetFromMat(test[test_idx][1], ds_parameters)
        LGBM_DatasetSetField(test_ds, "label", test[test_idx][2])
        tests_ds[test_idx] = test_ds
        tests_names[test_idx] = "test_$test_idx"
    end

    bst = LGBM_BoosterCreate(train_ds, tests_ds, tests_names, bst_parameters)

    results = Dict{String,Dict{String,Array{Float64,1}}}()
    for iteration in 1:estimator.num_iterations
        LGBM_BoosterUpdateOneIter(bst)
        evalprogress!(results, estimator, bst, tests_names, iteration)
    end

    estimator.booster = bst
    # estimator.model = readlines("$(tempdir)/model.txt")
    return results
end

function evalprogress!(results::Dict{String,Dict{String,Array{Float64,1}}},
                        estimator::LGBMEstimator, bst::BoosterHandle, tests_names::Array{String,1},
                        iteration::Integer)
    if mod(iteration - 1, estimator.metric_freq) == 0
        n_metrics = length(estimator.metric)
        n_tests = length(tests_names)

        if estimator.is_training_metric
            eval = LGBM_BoosterEval(bst, 0, n_metrics)
            storeeval!(results, estimator, iteration, "training", n_metrics, eval)
            printeval(estimator, iteration, "training", n_metrics, eval)
        end

        for test_idx in 1:n_tests
            eval = LGBM_BoosterEval(bst, test_idx, n_metrics)
            storeeval!(results, estimator, iteration, tests_names[test_idx], n_metrics, eval)
            printeval(estimator, iteration, tests_names[test_idx], n_metrics, eval)
        end
    end
    return nothing
end

function storeeval!(results::Dict{String,Dict{String,Array{Float64,1}}}, estimator::LGBMEstimator,
                    iteration::Integer, evalname::String, n_metrics::Integer,
                    eval::Array{Cfloat,1})
    for metric_idx in 1:n_metrics
        metricname = estimator.metric[metric_idx]
        if !haskey(results, evalname)
            num_evals = cld(estimator.num_iterations, estimator.metric_freq)
            results[evalname] = Dict{String,Array{Float64,1}}()
            results[evalname][metricname] = Array(Float64, num_evals)
        elseif !haskey(results[evalname], metricname)
            num_evals = cld(estimator.num_iterations, estimator.metric_freq)
            results[evalname][metricname] = Array(Float64, num_evals)
        end
        eval_idx = cld(iteration, estimator.metric_freq)
        # Reverse eval order, because LightGBM stores evals in reverse order
        results[evalname][metricname][eval_idx] = eval[n_metrics - metric_idx + 1]
    end

    return nothing
end

function printeval(estimator::LGBMEstimator, iteration::Integer, name::String, n_metrics::Integer,
                   eval::Array{Cfloat,1})
    print("Iteration: ", iteration, ", ", name, "'s ")
    for metric_idx in 1:n_metrics
        # Reverse eval order, because LightGBM stores evals in reverse order
        print(estimator.metric[metric_idx], ": ", eval[n_metrics - metric_idx + 1])
        metric_idx < n_metrics && print(", ")
    end
    print("\n")
end

function getparamstring(estimator::LGBMEstimator, params::Array{Symbol,1})
    paramstring = ""
    n_params = length(params)
    for param_idx in 1:n_params
        param = params[param_idx]
        param_value = getfield(estimator, param)
        if typeof(param_value) <: Array
            n_entries = length(param_value)
            if n_entries == 1
                paramstring = string(paramstring, param, "=", param_value[1], " ")
            elseif n_entries > 1
                paramstring = string(paramstring, param, "=", param_value[1])
                for entry_idx in 2:n_entries
                    paramstring = string(paramstring, ",", param_value[entry_idx])
                end
                paramstring = string(paramstring, " ")
            end
        else
            paramstring = string(paramstring, param, "=", param_value, " ")
        end
    end
    return paramstring[1:end-1]
end
