abstract type Estimator end
abstract type LGBMEstimator <: Estimator end

mutable struct LGBMRegression <: LGBMEstimator
    booster::Booster
    model::Vector{String}
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
    lambda_l1::Float64
    lambda_l2::Float64
    min_gain_to_split::Float64
    feature_fraction::Float64
    feature_fraction_seed::Int
    bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    early_stopping_round::Int

    max_bin::Int
    data_random_seed::Int
    init_score::String
    is_sparse::Bool
    save_binary::Bool
    categorical_feature::Vector{Int}

    is_unbalance::Bool

    metric::Vector{String}
    metric_freq::Int
    is_training_metric::Bool
    ndcg_at::Vector{Int}

    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_file::String

    num_class::Int
end

"""
    LGBMRegression(; [num_iterations = 10,
                      learning_rate = .1,
                      num_leaves = 127,
                      max_depth = -1,
                      tree_learner = \"serial\",
                      num_threads = Sys.CPU_THREADS,
                      histogram_pool_size = -1.,
                      min_data_in_leaf = 100,
                      min_sum_hessian_in_leaf = 10.,
                      lambda_l1 = 0.,
                      lambda_l2 = 0.,
                      min_gain_to_split = 0.,
                      feature_fraction = 1.,
                      feature_fraction_seed = 2,
                      bagging_fraction = 1.,
                      bagging_freq = 0,
                      bagging_seed = 3,
                      early_stopping_round = 0,
                      max_bin = 255,
                      data_random_seed = 1,
                      init_score = \"\",
                      is_sparse = true,
                      save_binary = false,
                      categorical_feature = Int[],
                      is_unbalance = false,
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
                        max_depth = -1,
                        tree_learner = "serial",
                        num_threads = Sys.CPU_THREADS,
                        histogram_pool_size = -1.,
                        min_data_in_leaf = 100,
                        min_sum_hessian_in_leaf = 10.,
                        lambda_l1 = 0.,
                        lambda_l2 = 0.,
                        min_gain_to_split = 0.,
                        feature_fraction = 1.,
                        feature_fraction_seed = 2,
                        bagging_fraction = 1.,
                        bagging_freq = 0,
                        bagging_seed = 3,
                        early_stopping_round = 0,
                        max_bin = 255,
                        data_random_seed = 1,
                        init_score = "",
                        is_sparse = true,
                        save_binary = false,
                        categorical_feature = Int[],
                        is_unbalance = false,
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

    return LGBMRegression(Booster(), String[], "regression", num_iterations, learning_rate, num_leaves,
                          max_depth, tree_learner, num_threads, histogram_pool_size,
                          min_data_in_leaf, min_sum_hessian_in_leaf, lambda_l1, lambda_l2,
                          min_gain_to_split, feature_fraction, feature_fraction_seed,
                          bagging_fraction, bagging_freq, bagging_seed, early_stopping_round,
                          max_bin, data_random_seed, init_score,
                          is_sparse, save_binary, categorical_feature,
                          is_unbalance, metric, metric_freq,
                          is_training_metric, ndcg_at, num_machines, local_listen_port, time_out,
                          machine_list_file,1)
end

mutable struct LGBMBinary <: LGBMEstimator
    booster::Booster
    model::Vector{String}
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
    lambda_l1::Float64
    lambda_l2::Float64
    min_gain_to_split::Float64
    feature_fraction::Float64
    feature_fraction_seed::Int
    bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    early_stopping_round::Int

    max_bin::Int
    data_random_seed::Int
    init_score::String
    is_sparse::Bool
    save_binary::Bool
    categorical_feature::Vector{Int}

    sigmoid::Float64
    is_unbalance::Bool

    metric::Vector{String}
    metric_freq::Int
    is_training_metric::Bool
    ndcg_at::Vector{Int}

    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_file::String

    num_class::Int
end

"""
    LGBMBinary(; [num_iterations = 10,
                  learning_rate = .1,
                  num_leaves = 127,
                  max_depth = -1,
                  tree_learner = \"serial\",
                  num_threads = Sys.CPU_THREADS,
                  histogram_pool_size = -1.,
                  min_data_in_leaf = 100,
                  min_sum_hessian_in_leaf = 10.,
                  lambda_l1 = 0.,
                  lambda_l2 = 0.,
                  min_gain_to_split = 0.,
                  feature_fraction = 1.,
                  feature_fraction_seed = 2,
                  bagging_fraction = 1.,
                  bagging_freq = 0,
                  bagging_seed = 3,
                  early_stopping_round = 0,
                  max_bin = 255,
                  data_random_seed = 1,
                  init_score = \"\",
                  is_sparse = true,
                  save_binary = false,
                  categorical_feature = Int[],
                  sigmoid = 1.,
                  is_unbalance = false,
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
                    num_threads = Sys.CPU_THREADS,
                    histogram_pool_size = -1.,
                    min_data_in_leaf = 100,
                    min_sum_hessian_in_leaf = 10.,
                    lambda_l1 = 0.,
                    lambda_l2 = 0.,
                    min_gain_to_split = 0.,
                    feature_fraction = 1.,
                    feature_fraction_seed = 2,
                    bagging_fraction = 1.,
                    bagging_freq = 0,
                    bagging_seed = 3,
                    early_stopping_round = 0,
                    max_bin = 255,
                    data_random_seed = 1,
                    init_score = "",
                    is_sparse = true,
                    save_binary = false,
                    categorical_feature = Int[],
                    sigmoid = 1.,
                    is_unbalance = false,
                    metric = ["binary_logloss"],
                    metric_freq = 1,
                    is_training_metric = false,
                    ndcg_at = Int[],
                    num_machines = 1,
                    local_listen_port = 12400,
                    time_out = 120,
                    machine_list_file = ""
                    )

    @assert(in(tree_learner, ("serial", "feature", "data")),
            "Unknown tree_learner, got $tree_learner")
    foreach(m -> @assert(in(m, ("l1", "l2", "ndcg", "auc", "binary_logloss", "binary_error",
                                "multi_logloss", "multi_error")),
                         "Unknown metric, got $m"), metric)

    return LGBMBinary(Booster(), String[], "binary", num_iterations, learning_rate, num_leaves,
                      max_depth, tree_learner, num_threads, histogram_pool_size, min_data_in_leaf,
                      min_sum_hessian_in_leaf, lambda_l1, lambda_l2, min_gain_to_split,
                      feature_fraction, feature_fraction_seed, bagging_fraction, bagging_freq,
                      bagging_seed, early_stopping_round, max_bin, data_random_seed,
                      init_score, is_sparse, save_binary,
                      categorical_feature, sigmoid, is_unbalance, metric,
                      metric_freq, is_training_metric, ndcg_at, num_machines, local_listen_port,
                      time_out, machine_list_file, 1)
end

mutable struct LGBMMulticlass <: LGBMEstimator
    booster::Booster
    model::Vector{String}
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
    lambda_l1::Float64
    lambda_l2::Float64
    min_gain_to_split::Float64
    feature_fraction::Float64
    feature_fraction_seed::Int
    bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    early_stopping_round::Int

    max_bin::Int
    data_random_seed::Int
    init_score::String
    is_sparse::Bool
    save_binary::Bool
    categorical_feature::Vector{Int}

    is_unbalance::Bool

    metric::Vector{String}
    metric_freq::Int
    is_training_metric::Bool
    ndcg_at::Vector{Int}

    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_file::String

    num_class::Int

    device_type::String
end

"""
    LGBMMulticlass(; [num_iterations = 10,
                      learning_rate = .1,
                      num_leaves = 127,
                      max_depth = -1,
                      tree_learner = \"serial\",
                      num_threads = Sys.CPU_THREADS,
                      histogram_pool_size = -1.,
                      min_data_in_leaf = 100,
                      min_sum_hessian_in_leaf = 10.,
                      lambda_l1 = 0.,
                      lambda_l2 = 0.,
                      min_gain_to_split = 0.,
                      feature_fraction = 1.,
                      feature_fraction_seed = 2,
                      bagging_fraction = 1.,
                      bagging_freq = 0,
                      bagging_seed = 3,
                      early_stopping_round = 0,
                      max_bin = 255,
                      data_random_seed = 1,
                      init_score = \"\",
                      is_sparse = true,
                      save_binary = false,
                      categorical_feature = Int[],
                      is_unbalance = false,
                      metric = [\"multi_logloss\"],
                      metric_freq = 1,
                      is_training_metric = false,
                      ndcg_at = Int[],
                      num_machines = 1,
                      local_listen_port = 12400,
                      time_out = 120,
                      machine_list_file = \"\",
                      num_class = 1,
                      device_type=\"cpu\"])

Return a LGBMMulticlass estimator.
"""
function LGBMMulticlass(; num_iterations = 10,
                        learning_rate = .1,
                        num_leaves = 127,
                        max_depth = -1,
                        tree_learner = "serial",
                        num_threads = Sys.CPU_THREADS,
                        histogram_pool_size = -1.,
                        min_data_in_leaf = 100,
                        min_sum_hessian_in_leaf = 10.,
                        lambda_l1 = 0.,
                        lambda_l2 = 0.,
                        min_gain_to_split = 0.,
                        feature_fraction = 1.,
                        feature_fraction_seed = 2,
                        bagging_fraction = 1.,
                        bagging_freq = 0,
                        bagging_seed = 3,
                        early_stopping_round = 0,
                        max_bin = 255,
                        data_random_seed = 1,
                        init_score = "",
                        is_sparse = true,
                        save_binary = false,
                        categorical_feature = Int[],
                        is_unbalance = false,
                        metric = ["multi_logloss"],
                        metric_freq = 1,
                        is_training_metric = false,
                        ndcg_at = Int[],
                        num_machines = 1,
                        local_listen_port = 12400,
                        time_out = 120,
                        machine_list_file = "",
                        num_class = 1,
                        device_type="cpu")

    @assert(in(tree_learner, ("serial", "feature", "data")),
            "Unknown tree_learner, got $tree_learner")
    foreach(m -> @assert(in(m, ("l1", "l2", "ndcg", "auc", "binary_logloss", "binary_error",
                                "multi_logloss", "multi_error")),
                         "Unknown metric, got $m"), metric)

    return LGBMMulticlass(Booster(), String[], "multiclass", num_iterations, learning_rate,
                          num_leaves, max_depth, tree_learner, num_threads, histogram_pool_size,
                          min_data_in_leaf, min_sum_hessian_in_leaf, lambda_l1, lambda_l2,
                          min_gain_to_split, feature_fraction, feature_fraction_seed,
                          bagging_fraction, bagging_freq, bagging_seed, early_stopping_round,
                          max_bin, data_random_seed, init_score,
                          is_sparse, save_binary, categorical_feature,
                          is_unbalance, metric, metric_freq,
                          is_training_metric, ndcg_at, num_machines, local_listen_port, time_out,
                          machine_list_file, num_class,device_type)
end
