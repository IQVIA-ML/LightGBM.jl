abstract Estimator

abstract LightGBMEstimator <: Estimator

type LightGBMRegression <: LightGBMEstimator
    model::Array{String,1}
    application::String

    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    tree_learner::String
    num_threads::Int

    min_data_in_leaf::Int
    min_sum_hessian_in_leaf::Float64
    feature_fraction::Float64
    feature_fraction_seed::Int
    bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    early_stopping_round::Int

    max_bin::Int
    data_random_seed::Int
    is_sigmoid::Bool
    init_score::String
    is_pre_partition::Bool
    is_sparse::Bool
    two_round::Bool
    save_binary::Bool

    sigmoid::Float64
    is_unbalance::Bool
    max_position::Int
    label_gain::Array{Float64,1}

    metric::Array{String,1}
    metric_freq::Int
    is_training_metric::Bool
    ndcg_at::Array{Int,1}

    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_file::String
end

"""
    LightGBMRegression(; [num_iterations = 10,
                          learning_rate = .1,
                          num_leaves = 127,
                          tree_learner = \"serial\",
                          num_threads = Sys.CPU_CORES,
                          min_data_in_leaf = 100,
                          min_sum_hessian_in_leaf = 10.,
                          feature_fraction = 1.,
                          feature_fraction_seed = 2,
                          bagging_fraction = 1.,
                          bagging_freq = 0,
                          bagging_seed = 3,
                          early_stopping_round = 0,
                          max_bin = 255,
                          data_random_seed = 1,
                          is_sigmoid = true,
                          init_score = \"\",
                          is_pre_partition = false,
                          is_sparse = true,
                          two_round = false,
                          save_binary = false,
                          sigmoid = 1.,
                          is_unbalance = false,
                          max_position = 20,
                          label_gain = Float64[],
                          metric = [\"l2\"],
                          metric_freq = 1,
                          is_training_metric = false,
                          ndcg_at = Int[],
                          num_machines = 1,
                          local_listen_port = 12400,
                          time_out = 120,
                          machine_list_file = \"\"])

Return a LightGBMRegression estimator.
"""
function LightGBMRegression(; num_iterations = 10,
                            learning_rate = .1,
                            num_leaves = 127,
                            tree_learner = "serial",
                            num_threads = Sys.CPU_CORES,
                            min_data_in_leaf = 100,
                            min_sum_hessian_in_leaf = 10.,
                            feature_fraction = 1.,
                            feature_fraction_seed = 2,
                            bagging_fraction = 1.,
                            bagging_freq = 0,
                            bagging_seed = 3,
                            early_stopping_round = 0,
                            max_bin = 255,
                            data_random_seed = 1,
                            is_sigmoid = true,
                            init_score = "",
                            is_pre_partition = false,
                            is_sparse = true,
                            two_round = false,
                            save_binary = false,
                            sigmoid = 1.,
                            is_unbalance = false,
                            max_position = 20,
                            label_gain = Float64[],
                            metric = ["l2"],
                            metric_freq = 1,
                            is_training_metric = false,
                            ndcg_at = Int[],
                            num_machines = 1,
                            local_listen_port = 12400,
                            time_out = 120,
                            machine_list_file = "")

    @assert(in(tree_learner, ("serial", "feature", "data")),
            "Unknown tree_learner, got $tree_learner")
    foreach(m -> @assert(in(m, ("l1", "l2", "ndcg", "auc", "binary_logloss", "binary_error")),
                         "Unknown metric, got $m"), metric)

    return LightGBMRegression(String[], "regression", num_iterations, learning_rate,
                              num_leaves, tree_learner, num_threads, min_data_in_leaf,
                              min_sum_hessian_in_leaf, feature_fraction, feature_fraction_seed,
                              bagging_fraction, bagging_freq, bagging_seed, early_stopping_round,
                              max_bin, data_random_seed, is_sigmoid, init_score, is_pre_partition,
                              is_sparse, two_round, save_binary, sigmoid, is_unbalance,
                              max_position, label_gain, metric, metric_freq, is_training_metric,
                              ndcg_at, num_machines, local_listen_port, time_out,
                              machine_list_file)
end

type LightGBMBinary <: LightGBMEstimator
    model::Array{String,1}
    application::String

    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    tree_learner::String
    num_threads::Int

    min_data_in_leaf::Int
    min_sum_hessian_in_leaf::Float64
    feature_fraction::Float64
    feature_fraction_seed::Int
    bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    early_stopping_round::Int

    max_bin::Int
    data_random_seed::Int
    is_sigmoid::Bool
    init_score::String
    is_pre_partition::Bool
    is_sparse::Bool
    two_round::Bool
    save_binary::Bool

    sigmoid::Float64
    is_unbalance::Bool
    max_position::Int
    label_gain::Array{Float64,1}

    metric::Array{String,1}
    metric_freq::Int
    is_training_metric::Bool
    ndcg_at::Array{Int,1}

    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_file::String
end

"""
    LightGBMBinary(; [num_iterations = 10,
                      learning_rate = .1,
                      num_leaves = 127,
                      tree_learner = \"serial\",
                      num_threads = Sys.CPU_CORES,
                      min_data_in_leaf = 100,
                      min_sum_hessian_in_leaf = 10.,
                      feature_fraction = 1.,
                      feature_fraction_seed = 2,
                      bagging_fraction = 1.,
                      bagging_freq = 0,
                      bagging_seed = 3,
                      early_stopping_round = 0,
                      max_bin = 255,
                      data_random_seed = 1,
                      is_sigmoid = true,
                      init_score = \"\",
                      is_pre_partition = false,
                      is_sparse = true,
                      two_round = false,
                      save_binary = false,
                      sigmoid = 1.,
                      is_unbalance = false,
                      max_position = 20,
                      label_gain = Float64[],
                      metric = [\"binary_logloss\"],
                      metric_freq = 1,
                      is_training_metric = false,
                      ndcg_at = Int[],
                      num_machines = 1,
                      local_listen_port = 12400,
                      time_out = 120,
                      machine_list_file = \"\"])

Return a LightGBMBinary estimator.
"""
function LightGBMBinary(; num_iterations = 10,
                        learning_rate = .1,
                        num_leaves = 127,
                        tree_learner = "serial",
                        num_threads = Sys.CPU_CORES,
                        min_data_in_leaf = 100,
                        min_sum_hessian_in_leaf = 10.,
                        feature_fraction = 1.,
                        feature_fraction_seed = 2,
                        bagging_fraction = 1.,
                        bagging_freq = 0,
                        bagging_seed = 3,
                        early_stopping_round = 0,
                        max_bin = 255,
                        data_random_seed = 1,
                        is_sigmoid = true,
                        init_score = "",
                        is_pre_partition = false,
                        is_sparse = true,
                        two_round = false,
                        save_binary = false,
                        sigmoid = 1.,
                        is_unbalance = false,
                        max_position = 20,
                        label_gain = Float64[],
                        metric = ["binary_logloss"],
                        metric_freq = 1,
                        is_training_metric = false,
                        ndcg_at = Int[],
                        num_machines = 1,
                        local_listen_port = 12400,
                        time_out = 120,
                        machine_list_file = "")

    @assert(in(tree_learner, ("serial", "feature", "data")),
            "Unknown tree_learner, got $tree_learner")
    foreach(m -> @assert(in(m, ("l1", "l2", "ndcg", "auc", "binary_logloss", "binary_error")),
                         "Unknown metric, got $m"), metric)

    return LightGBMBinary(String[], "binary", num_iterations, learning_rate, num_leaves,
                          tree_learner, num_threads, min_data_in_leaf, min_sum_hessian_in_leaf,
                          feature_fraction, feature_fraction_seed, bagging_fraction, bagging_freq,
                          bagging_seed, early_stopping_round, max_bin, data_random_seed,
                          is_sigmoid, init_score, is_pre_partition, is_sparse, two_round,
                          save_binary, sigmoid, is_unbalance, max_position, label_gain, metric,
                          metric_freq, is_training_metric, ndcg_at, num_machines,
                          local_listen_port, time_out, machine_list_file)
end

type LightGBMLambdaRank <: LightGBMEstimator
    model::Array{String,1}
    application::String

    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    tree_learner::String
    num_threads::Int

    min_data_in_leaf::Int
    min_sum_hessian_in_leaf::Float64
    feature_fraction::Float64
    feature_fraction_seed::Int
    bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    early_stopping_round::Int

    max_bin::Int
    data_random_seed::Int
    is_sigmoid::Bool
    init_score::String
    is_pre_partition::Bool
    is_sparse::Bool
    two_round::Bool
    save_binary::Bool

    sigmoid::Float64
    is_unbalance::Bool
    max_position::Int
    label_gain::Array{Float64,1}

    metric::Array{String,1}
    metric_freq::Int
    is_training_metric::Bool
    ndcg_at::Array{Int,1}

    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_file::String
end

"""
    LightGBMLambdaRank(; [num_iterations = 10,
                          learning_rate = .1,
                          num_leaves = 127,
                          tree_learner = \"serial\",
                          num_threads = Sys.CPU_CORES,
                          min_data_in_leaf = 100,
                          min_sum_hessian_in_leaf = 10.,
                          feature_fraction = 1.,
                          feature_fraction_seed = 2,
                          bagging_fraction = 1.,
                          bagging_freq = 0,
                          bagging_seed = 3,
                          early_stopping_round = 0,
                          max_bin = 255,
                          data_random_seed = 1,
                          is_sigmoid = true,
                          init_score = \"\",
                          is_pre_partition = false,
                          is_sparse = true,
                          two_round = false,
                          save_binary = false,
                          sigmoid = 1.,
                          is_unbalance = false,
                          max_position = 20,
                          label_gain = Float64[],
                          metric = [\"ndcg\"],
                          metric_freq = 1,
                          is_training_metric = false,
                          ndcg_at = Int[],
                          num_machines = 1,
                          local_listen_port = 12400,
                          time_out = 120,
                          machine_list_file = \"\"])

Return a LightGBMBinary estimator.
"""
function LightGBMLambdaRank(; num_iterations = 10,
                            learning_rate = .1,
                            num_leaves = 127,
                            tree_learner = "serial",
                            num_threads = Sys.CPU_CORES,
                            min_data_in_leaf = 100,
                            min_sum_hessian_in_leaf = 10.,
                            feature_fraction = 1.,
                            feature_fraction_seed = 2,
                            bagging_fraction = 1.,
                            bagging_freq = 0,
                            bagging_seed = 3,
                            early_stopping_round = 0,
                            max_bin = 255,
                            data_random_seed = 1,
                            is_sigmoid = true,
                            init_score = "",
                            is_pre_partition = false,
                            is_sparse = true,
                            two_round = false,
                            save_binary = false,
                            sigmoid = 1.,
                            is_unbalance = false,
                            max_position = 20,
                            label_gain = Float64[],
                            metric = ["ndcg"],
                            metric_freq = 1,
                            is_training_metric = false,
                            ndcg_at = Int[],
                            num_machines = 1,
                            local_listen_port = 12400,
                            time_out = 120,
                            machine_list_file = "")

    @assert(in(tree_learner, ("serial", "feature", "data")),
            "Unknown tree_learner, got $tree_learner")
    foreach(m -> @assert(in(m, ("l1", "l2", "ndcg", "auc", "binary_logloss", "binary_error")),
                         "Unknown metric, got $m"), metric)

    return LightGBMLambdaRank(String[], "lambdarank", num_iterations, learning_rate,
                              num_leaves, tree_learner, num_threads, min_data_in_leaf,
                              min_sum_hessian_in_leaf, feature_fraction, feature_fraction_seed,
                              bagging_fraction, bagging_freq, bagging_seed, early_stopping_round,
                              max_bin, data_random_seed, is_sigmoid, init_score, is_pre_partition,
                              is_sparse, two_round, save_binary, sigmoid, is_unbalance,
                              max_position, label_gain, metric, metric_freq, is_training_metric,
                              ndcg_at, num_machines, local_listen_port, time_out,
                              machine_list_file)
end
