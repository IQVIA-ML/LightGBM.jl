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
