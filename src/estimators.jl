abstract Estimator

abstract LGBMEstimator <: Estimator

type LGBMRegression <: LGBMEstimator
    model::Array{String,1}
    application::String

    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    max_depth::Int
    tree_learner::String
    num_threads::Int
    histogram_pool_size::Float64

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
    LGBMRegression(; [num_iterations = 10,
                      learning_rate = .1,
                      num_leaves = 127,
                      max_depth = -1,
                      tree_learner = \"serial\",
                      num_threads = Sys.CPU_CORES,
                      histogram_pool_size = -1.,
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

Return a LGBMRegression estimator.
"""
function LGBMRegression(; num_iterations = 10,
                        learning_rate = .1,
                        num_leaves = 127,
                        max_depth = 1,
                        tree_learner = "serial",
                        num_threads = Sys.CPU_CORES,
                        histogram_pool_size = -1.,
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
    foreach(m -> @assert(in(m, ("l1", "l2", "ndcg", "auc", "binary_logloss", "binary_error",
                                "multi_logloss", "multi_error")),
                         "Unknown metric, got $m"), metric)

    return LGBMRegression(String[], "regression", num_iterations, learning_rate, num_leaves,
                          max_depth, tree_learner, num_threads, histogram_pool_size,
                          min_data_in_leaf, min_sum_hessian_in_leaf, feature_fraction,
                          feature_fraction_seed, bagging_fraction, bagging_freq, bagging_seed,
                          early_stopping_round, max_bin, data_random_seed, is_sigmoid, init_score,
                          is_pre_partition, is_sparse, two_round, save_binary, sigmoid,
                          is_unbalance, max_position, label_gain, metric, metric_freq,
                          is_training_metric, ndcg_at, num_machines, local_listen_port, time_out,
                          machine_list_file)
end

type LGBMBinary <: LGBMEstimator
    model::Array{String,1}
    application::String

    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    max_depth::Int
    tree_learner::String
    num_threads::Int
    histogram_pool_size::Float64

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
    LGBMBinary(; [num_iterations = 10,
                  learning_rate = .1,
                  num_leaves = 127,
                  max_depth = -1,
                  tree_learner = \"serial\",
                  num_threads = Sys.CPU_CORES,
                  histogram_pool_size = -1.,
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

Return a LGBMBinary estimator.
"""
function LGBMBinary(; num_iterations = 10,
                    learning_rate = .1,
                    num_leaves = 127,
                    max_depth = -1,
                    tree_learner = "serial",
                    num_threads = Sys.CPU_CORES,
                    histogram_pool_size = -1.,
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
    foreach(m -> @assert(in(m, ("l1", "l2", "ndcg", "auc", "binary_logloss", "binary_error",
                                "multi_logloss", "multi_error")),
                         "Unknown metric, got $m"), metric)

    return LGBMBinary(String[], "binary", num_iterations, learning_rate, num_leaves, max_depth,
                      tree_learner, num_threads, histogram_pool_size, min_data_in_leaf,
                      min_sum_hessian_in_leaf, feature_fraction, feature_fraction_seed,
                      bagging_fraction, bagging_freq, bagging_seed, early_stopping_round, max_bin,
                      data_random_seed, is_sigmoid, init_score, is_pre_partition, is_sparse,
                      two_round, save_binary, sigmoid, is_unbalance, max_position, label_gain,
                      metric, metric_freq, is_training_metric, ndcg_at, num_machines,
                      local_listen_port, time_out, machine_list_file)
end

type LGBMLambdaRank <: LGBMEstimator
    model::Array{String,1}
    application::String

    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    max_depth::Int
    tree_learner::String
    num_threads::Int
    histogram_pool_size::Float64

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
    LGBMLambdaRank(; [num_iterations = 10,
                      learning_rate = .1,
                      num_leaves = 127,
                      max_depth = -1,
                      tree_learner = \"serial\",
                      num_threads = Sys.CPU_CORES,
                      histogram_pool_size = -1.,
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

Return a LGBMBinary estimator.
"""
function LGBMLambdaRank(; num_iterations = 10,
                        learning_rate = .1,
                        num_leaves = 127,
                        max_depth = -1,
                        tree_learner = "serial",
                        num_threads = Sys.CPU_CORES,
                        histogram_pool_size = -1.,
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
    foreach(m -> @assert(in(m, ("l1", "l2", "ndcg", "auc", "binary_logloss", "binary_error",
                                "multi_logloss", "multi_error")),
                         "Unknown metric, got $m"), metric)

    return LGBMLambdaRank(String[], "lambdarank", num_iterations, learning_rate, num_leaves,
                          max_depth, tree_learner, num_threads, histogram_pool_size,
                          min_data_in_leaf, min_sum_hessian_in_leaf, feature_fraction,
                          feature_fraction_seed, bagging_fraction, bagging_freq, bagging_seed,
                          early_stopping_round, max_bin, data_random_seed, is_sigmoid, init_score,
                          is_pre_partition, is_sparse, two_round, save_binary, sigmoid,
                          is_unbalance, max_position, label_gain, metric, metric_freq,
                          is_training_metric, ndcg_at, num_machines, local_listen_port, time_out,
                          machine_list_file)
end

type LGBMMulticlass <: LGBMEstimator
    model::Array{String,1}
    application::String

    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    max_depth::Int
    tree_learner::String
    num_threads::Int
    histogram_pool_size::Float64

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

    num_class::Int
end

"""
    LGBMMulticlass(; [num_iterations = 10,
                      learning_rate = .1,
                      num_leaves = 127,
                      max_depth = -1,
                      tree_learner = \"serial\",
                      num_threads = Sys.CPU_CORES,
                      histogram_pool_size = -1.,
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
                      metric = [\"multi_logloss\"],
                      metric_freq = 1,
                      is_training_metric = false,
                      ndcg_at = Int[],
                      num_machines = 1,
                      local_listen_port = 12400,
                      time_out = 120,
                      machine_list_file = \"\",
                      num_class = 1])

Return a LGBMMulticlass estimator.
"""
function LGBMMulticlass(; num_iterations = 10,
                        learning_rate = .1,
                        num_leaves = 127,
                        max_depth = -1,
                        tree_learner = "serial",
                        num_threads = Sys.CPU_CORES,
                        histogram_pool_size = -1.,
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
                        metric = ["multi_logloss"],
                        metric_freq = 1,
                        is_training_metric = false,
                        ndcg_at = Int[],
                        num_machines = 1,
                        local_listen_port = 12400,
                        time_out = 120,
                        machine_list_file = "",
                        num_class = 1)

    @assert(in(tree_learner, ("serial", "feature", "data")),
            "Unknown tree_learner, got $tree_learner")
    foreach(m -> @assert(in(m, ("l1", "l2", "ndcg", "auc", "binary_logloss", "binary_error",
                                "multi_logloss", "multi_error")),
                         "Unknown metric, got $m"), metric)

    return LGBMMulticlass(String[], "multiclass", num_iterations, learning_rate, num_leaves,
                          max_depth, tree_learner, num_threads, histogram_pool_size,
                          min_data_in_leaf, min_sum_hessian_in_leaf, feature_fraction,
                          feature_fraction_seed, bagging_fraction, bagging_freq, bagging_seed,
                          early_stopping_round, max_bin, data_random_seed, is_sigmoid, init_score,
                          is_pre_partition, is_sparse, two_round, save_binary, sigmoid,
                          is_unbalance, max_position, label_gain, metric, metric_freq,
                          is_training_metric, ndcg_at, num_machines, local_listen_port, time_out,
                          machine_list_file, num_class)
end