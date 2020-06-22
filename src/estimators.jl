abstract type Estimator end
abstract type LGBMEstimator <: Estimator end

mutable struct LGBMRegression <: LGBMEstimator
    booster::Booster
    model::String
    application::String    
    boosting::String
    
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

    drop_rate::Float64
    max_drop::Int
    skip_drop:: Float64
    xgboost_dart_mode::Bool
    uniform_drop::Bool
    drop_seed::Int
    top_rate::Float64
    other_rate::Float64

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
    LGBMRegression(; [
        objective = "regression",
        boosting = "gbdt",
        num_iterations = 10,
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
        drop_rate = 0.1,
        max_drop = 50,
        skip_drop = 0.5,
        xgboost_dart_mode = false,
        uniform_drop = false,
        drop_seed = 4,
        top_rate = 0.2,
        other_rate = 0.1,
        metric = [\"l2\"],
        metric_freq = 1,
        is_training_metric = false,
        ndcg_at = Int[],
        num_machines = 1,
        local_listen_port = 12400,
        time_out = 120,
        machine_list_file = \"\",
        device_type=\"cpu\",
    ])

Return a LGBMRegression estimator.
"""
function LGBMRegression(;
    objective = "regression",
    boosting = "gbdt",
    num_iterations = 10,
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
    drop_rate = 0.1,
    max_drop = 50,
    skip_drop = 0.5,
    xgboost_dart_mode = false,
    uniform_drop = false,
    drop_seed = 4,
    top_rate = 0.2,
    other_rate = 0.1,
    metric = ["l2"],
    metric_freq = 1,
    is_training_metric = false,
    ndcg_at = Int[],
    num_machines = 1,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_file = "",
    device_type="cpu",
)

    return LGBMRegression(
        Booster(), "", objective, boosting, num_iterations, learning_rate, num_leaves,
        max_depth, tree_learner, num_threads, histogram_pool_size,
        min_data_in_leaf, min_sum_hessian_in_leaf, lambda_l1, lambda_l2,
        min_gain_to_split, feature_fraction, feature_fraction_seed,
        bagging_fraction, bagging_freq, bagging_seed, early_stopping_round,
        max_bin, data_random_seed, init_score,
        is_sparse, save_binary, categorical_feature,
        is_unbalance, drop_rate, max_drop, skip_drop, xgboost_dart_mode,
        uniform_drop, drop_seed, top_rate, other_rate,
        metric, metric_freq,
        is_training_metric, ndcg_at, num_machines, local_listen_port, time_out,
        machine_list_file, 1, device_type
    )
end


mutable struct LGBMClassification <: LGBMEstimator
    booster::Booster
    model::String
    application::String
    boosting :: String

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

    drop_rate::Float64
    max_drop::Int
    skip_drop:: Float64
    xgboost_dart_mode::Bool
    uniform_drop::Bool
    drop_seed::Int
    top_rate::Float64
    other_rate::Float64

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
    LGBMClassification(;[
        objective = "multiclass",
        boosting = "gbdt",
        num_iterations = 10,
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
        drop_rate = 0.1,
        max_drop = 50,
        skip_drop = 0.5,
        xgboost_dart_mode = false,
        uniform_drop = false,
        drop_seed = 4,
        top_rate = 0.2,
        other_rate = 0.1,
        metric = [\"multi_logloss\"],
        metric_freq = 1,
        is_training_metric = false,
        ndcg_at = Int[],
        num_machines = 1,
        local_listen_port = 12400,
        time_out = 120,
        machine_list_file = \"\",
        num_class = 1,
        device_type=\"cpu\",
    ])

Return a LGBMClassification estimator.
"""
function LGBMClassification(;
    objective = "multiclass",
    boosting = "gbdt",
    num_iterations = 10,
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
    drop_rate = 0.1,
    max_drop = 50,
    skip_drop = 0.5,
    xgboost_dart_mode = false,
    uniform_drop = false,
    drop_seed = 4,
    top_rate = 0.2,
    other_rate = 0.1,
    metric = ["None"],
    metric_freq = 1,
    is_training_metric = false,
    ndcg_at = Int[],
    num_machines = 1,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_file = "",
    num_class = 2,
    device_type="cpu",
)

    return LGBMClassification(
        Booster(), "", objective, boosting, num_iterations, learning_rate,
        num_leaves, max_depth, tree_learner, num_threads, histogram_pool_size,
        min_data_in_leaf, min_sum_hessian_in_leaf, lambda_l1, lambda_l2,
        min_gain_to_split, feature_fraction, feature_fraction_seed,
        bagging_fraction, bagging_freq, bagging_seed, early_stopping_round,
        max_bin, data_random_seed, init_score,
        is_sparse, save_binary, categorical_feature,
        is_unbalance, drop_rate, max_drop, skip_drop, xgboost_dart_mode,
        uniform_drop, drop_seed, top_rate, other_rate, metric, metric_freq,
        is_training_metric, ndcg_at, num_machines, local_listen_port, time_out,
        machine_list_file, num_class, device_type,
    )
end
