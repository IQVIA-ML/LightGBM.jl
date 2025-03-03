abstract type Estimator end
abstract type LGBMEstimator <: Estimator end

mutable struct LGBMRegression <: LGBMEstimator
    booster::Booster
    model::String
    objective::String
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
    max_delta_step::Float64
    lambda_l1::Float64
    lambda_l2::Float64
    min_gain_to_split::Float64
    feature_fraction::Float64
    feature_fraction_bynode::Float64
    feature_fraction_seed::Int
    bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    early_stopping_round::Int
    extra_trees::Bool
    extra_seed::Int

    max_bin::Int
    bin_construct_sample_cnt::Int
    data_random_seed::Int
    is_enable_sparse::Bool
    save_binary::Bool
    categorical_feature::Vector{Int}
    use_missing::Bool
    linear_tree::Bool
    feature_pre_filter::Bool

    is_unbalance::Bool
    boost_from_average::Bool
    alpha::Float64

    drop_rate::Float64
    max_drop::Int
    skip_drop:: Float64
    xgboost_dart_mode::Bool
    uniform_drop::Bool
    drop_seed::Int
    top_rate::Float64
    other_rate::Float64
    min_data_per_group::Int
    max_cat_threshold::Int
    cat_l2::Float64
    cat_smooth::Float64

    metric::Vector{String}
    metric_freq::Int
    is_provide_training_metric::Bool
    eval_at::Vector{Int}

    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_filename::String

    num_class::Int
    device_type::String
    gpu_use_dp::Bool
    gpu_platform_id::Int
    gpu_device_id::Int
    num_gpu::Int
    force_col_wise::Bool
    force_row_wise::Bool
end

"""
    LGBMRegression(; [
        objective = "regression",
        boosting = "gbdt",
        num_iterations = 100,
        learning_rate = .1,
        num_leaves = 31,
        max_depth = -1,
        tree_learner = \"serial\",
        num_threads = 0,
        histogram_pool_size = -1.,
        min_data_in_leaf = 20,
        min_sum_hessian_in_leaf = 1e-3,
        max_delta_step = 0.,
        lambda_l1 = 0.,
        lambda_l2 = 0.,
        min_gain_to_split = 0.,
        feature_fraction = 1.,
        feature_fraction_bynode = 1.,
        feature_fraction_seed = 2,
        bagging_fraction = 1.,
        bagging_freq = 0,
        bagging_seed = 3,
        early_stopping_round = 0,
        extra_trees = false
        extra_seed = 6,
        max_bin = 255,
        bin_construct_sample_cnt = 200000,
        data_random_seed = 1,
        is_enable_sparse = true,
        save_binary = false,
        categorical_feature = Int[],
        use_missing = true,
        linear_tree = false,
        feature_pre_filter = true,
        is_unbalance = false,
        boost_from_average = true,
        alpha = 0.9,
        drop_rate = 0.1,
        max_drop = 50,
        skip_drop = 0.5,
        xgboost_dart_mode = false,
        uniform_drop = false,
        drop_seed = 4,
        top_rate = 0.2,
        other_rate = 0.1,
        min_data_per_group = 100,
        max_cat_threshold = 32,
        cat_l2 = 10.0,
        cat_smooth = 10.0,
        metric = [""],
        metric_freq = 1,
        is_provide_training_metric = false,
        eval_at = Int[1, 2, 3, 4, 5],
        num_machines = 1,
        local_listen_port = 12400,
        time_out = 120,
        machine_list_filename = \"\",
        device_type=\"cpu\",
        gpu_use_dp = false,
        gpu_platform_id = -1,
        gpu_device_id = -1,
        num_gpu = 1,
        force_col_wise = false
        force_row_wise = false
    ])

Return a LGBMRegression estimator.
"""
function LGBMRegression(;
    objective = "regression",
    boosting = "gbdt",
    num_iterations = 100,
    learning_rate = .1,
    num_leaves = 31,
    max_depth = -1,
    tree_learner = "serial",
    num_threads = 0,
    histogram_pool_size = -1.,
    min_data_in_leaf = 20,
    min_sum_hessian_in_leaf = 1e-3,
    max_delta_step = 0.,
    lambda_l1 = 0.,
    lambda_l2 = 0.,
    min_gain_to_split = 0.,
    feature_fraction = 1.,
    feature_fraction_bynode = 1.,
    feature_fraction_seed = 2,
    bagging_fraction = 1.,
    bagging_freq = 0,
    bagging_seed = 3,
    early_stopping_round = 0,
    extra_trees = false,
    extra_seed = 6,
    max_bin = 255,
    bin_construct_sample_cnt = 200000,
    data_random_seed = 1,
    is_enable_sparse = true,
    save_binary = false,
    categorical_feature = Int[],
    use_missing = true,
    linear_tree = false,
    feature_pre_filter = true,
    is_unbalance = false,
    boost_from_average = true,
    alpha = 0.9,
    drop_rate = 0.1,
    max_drop = 50,
    skip_drop = 0.5,
    xgboost_dart_mode = false,
    uniform_drop = false,
    drop_seed = 4,
    top_rate = 0.2,
    other_rate = 0.1,
    min_data_per_group = 100,
    max_cat_threshold = 32,
    cat_l2 = 10.0,
    cat_smooth = 10.0,
    metric = [""],
    metric_freq = 1,
    is_provide_training_metric = false,
    eval_at = Int[1, 2, 3, 4, 5],
    num_machines = 1,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_filename = "",
    device_type="cpu",
    gpu_use_dp = false,
    gpu_platform_id = -1,
    gpu_device_id = -1,
    num_gpu = 1,
    force_col_wise = false,
    force_row_wise = false,
)

    return LGBMRegression(
        Booster(), "", objective, boosting, num_iterations, learning_rate, num_leaves,
        max_depth, tree_learner, num_threads, histogram_pool_size,
        min_data_in_leaf, min_sum_hessian_in_leaf, max_delta_step, lambda_l1, lambda_l2,
        min_gain_to_split, feature_fraction, feature_fraction_bynode, feature_fraction_seed,
        bagging_fraction, bagging_freq, bagging_seed, early_stopping_round, extra_trees,
        extra_seed, max_bin, bin_construct_sample_cnt, data_random_seed,
        is_enable_sparse, save_binary, categorical_feature, use_missing, linear_tree, feature_pre_filter,
        is_unbalance, boost_from_average, alpha, drop_rate, max_drop, skip_drop,
        xgboost_dart_mode,uniform_drop, drop_seed, top_rate, other_rate, min_data_per_group, max_cat_threshold,
        cat_l2, cat_smooth, metric, metric_freq, is_provide_training_metric, eval_at, num_machines, local_listen_port, time_out,
        machine_list_filename, 1, device_type, gpu_use_dp, gpu_platform_id, gpu_device_id, num_gpu,
        force_col_wise, force_row_wise,
    )
end


mutable struct LGBMClassification <: LGBMEstimator
    booster::Booster
    model::String
    objective::String
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
    max_delta_step::Float64
    lambda_l1::Float64
    lambda_l2::Float64
    min_gain_to_split::Float64
    feature_fraction::Float64
    feature_fraction_bynode::Float64
    feature_fraction_seed::Int
    bagging_fraction::Float64
    pos_bagging_fraction::Float64
    neg_bagging_fraction::Float64

    bagging_freq::Int
    bagging_seed::Int
    early_stopping_round::Int
    extra_trees::Bool
    extra_seed::Int

    max_bin::Int
    bin_construct_sample_cnt::Int
    data_random_seed::Int
    is_enable_sparse::Bool
    save_binary::Bool
    categorical_feature::Vector{Int}
    use_missing::Bool
    linear_tree::Bool
    feature_pre_filter::Bool

    is_unbalance::Bool
    boost_from_average::Bool
    scale_pos_weight::Float64
    sigmoid::Float64

    drop_rate::Float64
    max_drop::Int
    skip_drop:: Float64
    xgboost_dart_mode::Bool
    uniform_drop::Bool
    drop_seed::Int
    top_rate::Float64
    other_rate::Float64
    min_data_per_group::Int
    max_cat_threshold::Int
    cat_l2::Float64
    cat_smooth::Float64

    metric::Vector{String}
    metric_freq::Int
    is_provide_training_metric::Bool
    eval_at::Vector{Int}

    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_filename::String

    num_class::Int

    device_type::String
    gpu_use_dp::Bool
    gpu_platform_id::Int
    gpu_device_id::Int
    num_gpu::Int
    force_col_wise::Bool
    force_row_wise::Bool

end

"""
    LGBMClassification(;[
        objective = "multiclass",
        boosting = "gbdt",
        num_iterations = 100,
        learning_rate = .1,
        num_leaves = 31,
        max_depth = -1,
        tree_learner = \"serial\",
        num_threads = 0,
        histogram_pool_size = -1.,
        min_data_in_leaf = 20,
        min_sum_hessian_in_leaf = 1e-3,
        max_delta_step = 0.,
        lambda_l1 = 0.,
        lambda_l2 = 0.,
        min_gain_to_split = 0.,
        feature_fraction = 1.,
        feature_fraction_bynode = 1.,
        feature_fraction_seed = 2,
        bagging_fraction = 1.,
        pos_bagging_fraction = 1.,
        neg_bagging_fraction = 1.,
        bagging_freq = 0,
        bagging_seed = 3,
        early_stopping_round = 0,
        extra_trees = false,
        extra_seed = 6,
        max_bin = 255,
        bin_construct_sample_cnt = 200000,
        data_random_seed = 1,
        is_enable_sparse = true,
        save_binary = false,
        categorical_feature = Int[],
        use_missing = true,
        linear_tree = false,
        feature_pre_filter = true,
        is_unbalance = false,
        boost_from_average = true,
        scale_pos_weight = 1.0,
        sigmoid = 1.0,
        drop_rate = 0.1,
        max_drop = 50,
        skip_drop = 0.5,
        xgboost_dart_mode = false,
        uniform_drop = false,
        drop_seed = 4,
        top_rate = 0.2,
        other_rate = 0.1,
        min_data_per_group = 100,
        max_cat_threshold = 32,
        cat_l2 = 10.0,
        cat_smooth = 10.0,
        metric = [""],
        metric_freq = 1,
        is_provide_training_metric = false,
        eval_at = Int[1, 2, 3, 4, 5],
        num_machines = 1,
        local_listen_port = 12400,
        time_out = 120,
        machine_list_filename = \"\",
        num_class = 2,
        device_type=\"cpu\",
        gpu_use_dp = false,
        gpu_platform_id = -1,
        gpu_device_id = -1,
        num_gpu = 1,
        force_col_wise = false,
        force_row_wise = false,
    ])

Return a LGBMClassification estimator.
"""
function LGBMClassification(;
    objective = "multiclass",
    boosting = "gbdt",
    num_iterations = 100,
    learning_rate = .1,
    num_leaves = 31,
    max_depth = -1,
    tree_learner = "serial",
    num_threads = 0,
    histogram_pool_size = -1.,
    min_data_in_leaf = 20,
    min_sum_hessian_in_leaf = 1e-3,
    max_delta_step = 0.,
    lambda_l1 = 0.,
    lambda_l2 = 0.,
    min_gain_to_split = 0.,
    feature_fraction = 1.,
    feature_fraction_bynode = 1.,
    feature_fraction_seed = 2,
    bagging_fraction = 1.,
    pos_bagging_fraction = 1.,
    neg_bagging_fraction = 1.,
    bagging_freq = 0,
    bagging_seed = 3,
    early_stopping_round = 0,
    extra_trees = false,
    extra_seed = 6,
    max_bin = 255,
    bin_construct_sample_cnt = 200000,
    data_random_seed = 1,
    is_enable_sparse = true,
    save_binary = false,
    categorical_feature = Int[],
    use_missing = true,
    linear_tree = false,
    feature_pre_filter = true,
    is_unbalance = false,
    boost_from_average = true,
    scale_pos_weight = 1.0,
    sigmoid = 1.0,
    drop_rate = 0.1,
    max_drop = 50,
    skip_drop = 0.5,
    xgboost_dart_mode = false,
    uniform_drop = false,
    drop_seed = 4,
    top_rate = 0.2,
    other_rate = 0.1,
    min_data_per_group = 100,
    max_cat_threshold = 32,
    cat_l2 = 10.0,
    cat_smooth = 10.0,
    metric = [""],
    metric_freq = 1,
    is_provide_training_metric = false,
    eval_at = Int[1, 2, 3, 4, 5],
    num_machines = 1,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_filename = "",
    num_class = 2,
    device_type="cpu",
    gpu_use_dp = false,
    gpu_platform_id = -1,
    gpu_device_id = -1,
    num_gpu = 1,
    force_col_wise = false,
    force_row_wise = false,
)

    return LGBMClassification(
        Booster(), "", objective, boosting, num_iterations, learning_rate,
        num_leaves, max_depth, tree_learner, num_threads, histogram_pool_size,
        min_data_in_leaf, min_sum_hessian_in_leaf, max_delta_step, lambda_l1, lambda_l2,
        min_gain_to_split, feature_fraction, feature_fraction_bynode, feature_fraction_seed,
        bagging_fraction, pos_bagging_fraction, neg_bagging_fraction,bagging_freq,
        bagging_seed, early_stopping_round, extra_trees, extra_seed, max_bin, bin_construct_sample_cnt,
        data_random_seed, is_enable_sparse, save_binary,
        categorical_feature, use_missing, linear_tree, feature_pre_filter, is_unbalance, boost_from_average, scale_pos_weight, sigmoid,
        drop_rate, max_drop, skip_drop, xgboost_dart_mode,
        uniform_drop, drop_seed, top_rate, other_rate, min_data_per_group, max_cat_threshold, cat_l2, cat_smooth,
        metric, metric_freq, is_provide_training_metric, eval_at, num_machines, local_listen_port, time_out,
        machine_list_filename, num_class, device_type, gpu_use_dp, gpu_platform_id, gpu_device_id, num_gpu,
        force_col_wise, force_row_wise,
    )
end

mutable struct LGBMRanking <: LGBMEstimator
    booster::Booster
    model::String
    objective::String
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
    max_delta_step::Float64
    lambda_l1::Float64
    lambda_l2::Float64
    min_gain_to_split::Float64
    feature_fraction::Float64
    feature_fraction_bynode::Float64
    feature_fraction_seed::Int
    bagging_fraction::Float64
    pos_bagging_fraction::Float64
    neg_bagging_fraction::Float64

    bagging_freq::Int
    bagging_seed::Int
    early_stopping_round::Int
    extra_trees::Bool
    extra_seed::Int

    max_bin::Int
    bin_construct_sample_cnt::Int
    data_random_seed::Int
    is_enable_sparse::Bool
    save_binary::Bool
    categorical_feature::Vector{Int}
    use_missing::Bool
    linear_tree::Bool
    feature_pre_filter::Bool

    is_unbalance::Bool
    boost_from_average::Bool
    scale_pos_weight::Float64
    sigmoid::Float64

    drop_rate::Float64
    max_drop::Int
    skip_drop:: Float64
    xgboost_dart_mode::Bool
    uniform_drop::Bool
    drop_seed::Int
    top_rate::Float64
    other_rate::Float64
    min_data_per_group::Int
    max_cat_threshold::Int
    cat_l2::Float64
    cat_smooth::Float64

    metric::Vector{String}
    metric_freq::Int
    is_provide_training_metric::Bool
    eval_at::Vector{Int}

    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_filename::String

    num_class::Int

    device_type::String
    gpu_use_dp::Bool
    gpu_platform_id::Int
    gpu_device_id::Int
    num_gpu::Int
    force_col_wise::Bool
    force_row_wise::Bool

    lambdarank_truncation_level::Int
    lambdarank_norm::Bool
    label_gain::Vector{Int}
    objective_seed::Int
    group_column::String

end

"""
    LGBMRanking(;[
        objective = "lambdarank",
        boosting = "gbdt",
        num_iterations = 100,
        learning_rate = .1,
        num_leaves = 31,
        max_depth = -1,
        tree_learner = \"serial\",
        num_threads = 0,
        histogram_pool_size = -1.,
        min_data_in_leaf = 20,
        min_sum_hessian_in_leaf = 1e-3,
        max_delta_step = 0.,
        lambda_l1 = 0.,
        lambda_l2 = 0.,
        min_gain_to_split = 0.,
        feature_fraction = 1.,
        feature_fraction_bynode = 1.,
        feature_fraction_seed = 2,
        bagging_fraction = 1.,
        pos_bagging_fraction = 1.,
        neg_bagging_fraction = 1.,
        bagging_freq = 0,
        bagging_seed = 3,
        early_stopping_round = 0,
        extra_trees = false,
        extra_seed = 6,
        max_bin = 255,
        bin_construct_sample_cnt = 200000,
        data_random_seed = 1,
        is_enable_sparse = true,
        save_binary = false,
        categorical_feature = Int[],
        use_missing = true,
        linear_tree = false,
        feature_pre_filter = true,
        is_unbalance = false,
        boost_from_average = true,
        scale_pos_weight = 1.0,
        sigmoid = 1.0,
        drop_rate = 0.1,
        max_drop = 50,
        skip_drop = 0.5,
        xgboost_dart_mode = false,
        uniform_drop = false,
        drop_seed = 4,
        top_rate = 0.2,
        other_rate = 0.1,
        min_data_per_group = 100,
        max_cat_threshold = 32,
        cat_l2 = 10.0,
        cat_smooth = 10.0,
        metric = [""],
        metric_freq = 1,
        is_provide_training_metric = false,
        eval_at = Int[1, 2, 3, 4, 5],
        num_machines = 1,
        local_listen_port = 12400,
        time_out = 120,
        machine_list_filename = \"\",
        num_class = 1,
        device_type=\"cpu\",
        gpu_use_dp = false,
        gpu_platform_id = -1,
        gpu_device_id = -1,
        num_gpu = 1,
        force_col_wise = false,
        force_row_wise = false,
        lambdarank_truncation_level = 30,
        lambdarank_norm = true,
        label_gain = [2^i - 1 for i in 0:30],
        objective_seed = 5,
        group_column = ""
    ])

Return a LGBMRanking estimator.
"""
function LGBMRanking(;
    objective = "lambdarank",
    boosting = "gbdt",
    num_iterations = 100,
    learning_rate = .1,
    num_leaves = 31,
    max_depth = -1,
    tree_learner = "serial",
    num_threads = 0,
    histogram_pool_size = -1.,
    min_data_in_leaf = 20,
    min_sum_hessian_in_leaf = 1e-3,
    max_delta_step = 0.,
    lambda_l1 = 0.,
    lambda_l2 = 0.,
    min_gain_to_split = 0.,
    feature_fraction = 1.,
    feature_fraction_bynode = 1.,
    feature_fraction_seed = 2,
    bagging_fraction = 1.,
    pos_bagging_fraction = 1.,
    neg_bagging_fraction = 1.,
    bagging_freq = 0,
    bagging_seed = 3,
    early_stopping_round = 0,
    extra_trees = false,
    extra_seed = 6,
    max_bin = 255,
    bin_construct_sample_cnt = 200000,
    data_random_seed = 1,
    is_enable_sparse = true,
    save_binary = false,
    categorical_feature = Int[],
    use_missing = true,
    linear_tree = false,
    feature_pre_filter = true,
    is_unbalance = false,
    boost_from_average = true,
    scale_pos_weight = 1.0,
    sigmoid = 1.0,
    drop_rate = 0.1,
    max_drop = 50,
    skip_drop = 0.5,
    xgboost_dart_mode = false,
    uniform_drop = false,
    drop_seed = 4,
    top_rate = 0.2,
    other_rate = 0.1,
    min_data_per_group = 100,
    max_cat_threshold = 32,
    cat_l2 = 10.0,
    cat_smooth = 10.0,
    metric = ["ndcg"],
    metric_freq = 1,
    is_provide_training_metric = false,
    eval_at = Int[1, 2, 3, 4, 5],
    num_machines = 1,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_filename = "",
    num_class = 1,
    device_type="cpu",
    gpu_use_dp = false,
    gpu_platform_id = -1,
    gpu_device_id = -1,
    num_gpu = 1,
    force_col_wise = false,
    force_row_wise = false,
    lambdarank_truncation_level = 30,
    lambdarank_norm = true,
    label_gain = [2^i - 1 for i in 0:30],
    objective_seed = 5,
    group_column = "",
)

    return LGBMRanking(
        Booster(), "", objective, boosting, num_iterations, learning_rate,
        num_leaves, max_depth, tree_learner, num_threads, histogram_pool_size,
        min_data_in_leaf, min_sum_hessian_in_leaf, max_delta_step, lambda_l1, lambda_l2,
        min_gain_to_split, feature_fraction, feature_fraction_bynode, feature_fraction_seed,
        bagging_fraction, pos_bagging_fraction, neg_bagging_fraction,bagging_freq,
        bagging_seed, early_stopping_round, extra_trees, extra_seed, max_bin, bin_construct_sample_cnt,
        data_random_seed, is_enable_sparse, save_binary,
        categorical_feature, use_missing, linear_tree, feature_pre_filter, is_unbalance, boost_from_average, scale_pos_weight, sigmoid,
        drop_rate, max_drop, skip_drop, xgboost_dart_mode,
        uniform_drop, drop_seed, top_rate, other_rate, min_data_per_group, max_cat_threshold, cat_l2, cat_smooth,
        metric, metric_freq, is_provide_training_metric, eval_at, num_machines, local_listen_port, time_out,
        machine_list_filename, num_class, device_type, gpu_use_dp, gpu_platform_id, gpu_device_id, num_gpu,
        force_col_wise, force_row_wise, lambdarank_truncation_level, lambdarank_norm, label_gain, objective_seed, group_column,
    )
end
