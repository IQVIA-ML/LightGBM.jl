const DATASETPARAMS = [:is_sparse, :max_bin, :data_random_seed, :categorical_feature]

const BOOSTERPARAMS = [:application, :learning_rate, :num_leaves, :max_depth, :tree_learner,
                       :num_threads, :histogram_pool_size, :min_data_in_leaf,
                       :min_sum_hessian_in_leaf, :lambda_l1, :lambda_l2, :min_gain_to_split,
                       :feature_fraction, :feature_fraction_seed, :bagging_fraction,
                       :bagging_freq, :bagging_seed, :early_stopping_round, :sigmoid,
                       :is_unbalance, :metric, :is_training_metric, :ndcg_at, :num_machines,
                       :local_listen_port, :time_out, :machine_list_file, :num_class,:device]

const INDEXPARAMS = [:categorical_feature]

const MAXIMIZE_METRICS = ["auc", "ndcg"]

"""
    savemodel(estimator, filename; [num_iteration = -1])

Save the fitted model in `estimator` as `filename`.

# Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `filename::String`: the name of the file to save the model in.
* `num_iteration::Integer`: keyword argument that sets the number of iterations of the model that
    should be saved. `< 0` for all iterations.
"""
function savemodel(estimator::LGBMEstimator, filename::String; num_iteration::Integer = -1)
    @assert(estimator.booster.handle != C_NULL, "Estimator does not contain a fitted model.")
    LGBM_BoosterSaveModel(estimator.booster, num_iteration, filename)
    return nothing
end

"""
    loadmodel(estimator, filename)

Load the fitted model `filename` into `estimator`. Note that this only loads the fitted model—not
the parameters or data of the estimator whose model was saved as `filename`.

# Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `filename::String`: the name of the file that contains the model.
"""
function loadmodel(estimator::LGBMEstimator, filename::String)
    estimator.booster = LGBM_BoosterCreateFromModelfile(filename)
    return nothing
end

function log_fatal(verbosity::Integer, msg...)
    warn(msg...)
end

function log_warning(verbosity::Integer, msg...)
    verbosity >= 0 && warn(msg...)
end

function log_info(verbosity::Integer, msg...)
    verbosity >= 1 && print(msg...)
end

function log_debug(verbosity::Integer, msg...)
    verbosity >= 2 && print(msg...)
end

function shrinkresults!(results, last_retained_iter::Integer)
    for test_key in keys(results)
        test = results[test_key]
        for metric_key in keys(test)
            test[metric_key] = test[metric_key][1:last_retained_iter]
        end
    end
    return nothing
end
