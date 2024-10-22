abstract type Estimator end
abstract type LGBMEstimator <: Estimator end

mutable struct LGBMRegression <: LGBMEstimator
    booster::Booster
    model::String

    # Core parameters
    objective::String
    boosting::String
    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    tree_learner::String
    num_threads::Int
    device_type::String

    # Learning control parameters
    force_col_wise::Bool
    force_row_wise::Bool
    histogram_pool_size::Float64 
    max_depth::Int
    min_data_in_leaf::Int
    min_sum_hessian_in_leaf::Float64
    bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    feature_fraction::Float64
    feature_fraction_bynode::Float64
    feature_fraction_seed::Int
    extra_trees::Bool
    extra_seed::Int
    early_stopping_round::Int
    max_delta_step::Float64
    lambda_l1::Float64
    lambda_l2::Float64
    min_gain_to_split::Float64
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
    forcedsplits_filename::String
    refit_decay_rate::Float64
    
    # Dataset parameters
    linear_tree::Bool
    max_bin::Int
    bin_construct_sample_cnt::Int
    data_random_seed::Int
    is_enable_sparse::Bool
    use_missing::Bool
    feature_pre_filter::Bool
    two_round::Bool
    header::Bool
    label_column::String
    weight_column::String
    ignore_column::String
    categorical_feature::Vector{Int}
    forcedbins_filename::String
    precise_float_parser::Bool

    # Predict parameters
    start_iteration_predict::Int
    num_iteration_predict::Int
    predict_raw_score::Bool
    predict_leaf_index::Bool
    predict_contrib::Bool
    predict_disable_shape_check::Bool

    # Objective parameters
    num_class::Int 
    is_unbalance::Bool
    boost_from_average::Bool
    alpha::Float64

    # Metric parameters
    metric::Vector{String}
    metric_freq::Int
    is_provide_training_metric::Bool
    eval_at::Vector{Int}

    # Network parameters
    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_filename::String

    # GPU parameters
    gpu_platform_id::Int
    gpu_device_id::Int
    gpu_use_dp::Bool
    num_gpu::Int
end

"""
    LGBMRegression(; [
        objective = "regression",
        boosting = "gbdt",
        num_iterations = 100,
        learning_rate = .1,
        num_leaves = 31,
        tree_learner = \"serial\",
        num_threads = 0,
        device_type=\"cpu\",
        force_col_wise = false
        force_row_wise = false
        histogram_pool_size = -1.,
        max_depth = -1,
        min_data_in_leaf = 20,
        min_sum_hessian_in_leaf = 1e-3,
        bagging_fraction = 1.,
        bagging_freq = 0,
        bagging_seed = 3,
        feature_fraction = 1.,
        feature_fraction_bynode = 1.,
        feature_fraction_seed = 2,
        extra_trees = false
        extra_seed = 6,
        early_stopping_round = 0,
        max_delta_step = 0.,
        lambda_l1 = 0.,
        lambda_l2 = 0.,
        min_gain_to_split = 0.,
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
        cat_l2 = 10.,
        cat_smooth = 10.,
        forcedsplits_filename = "",
        refit_decay_rate = 0.9,
        linear_tree = false,
        max_bin = 255,
        bin_construct_sample_cnt = 200000,
        data_random_seed = 1,
        is_enable_sparse = true,
        use_missing = true,
        feature_pre_filter = true,
        two_round = false,
        header = false,
        label_column = "",
        weight_column = "",
        ignore_column = "",
        categorical_feature = Int[],
        forcedbins_filename = "",
        precise_float_parser = false,
        start_iteration_predict = 0,
        num_iteration_predict = -1,
        predict_raw_score = false,
        predict_leaf_index = false,
        predict_contrib = false,
        predict_disable_shape_check = false,
        is_unbalance = false,
        boost_from_average = true,
        alpha = 0.9,
        metric = [""],
        metric_freq = 1,
        is_provide_training_metric = false,
        eval_at = Int[1, 2, 3, 4, 5],
        num_machines = 1,
        local_listen_port = 12400,
        time_out = 120,
        machine_list_filename = \"\",
        gpu_platform_id = -1,
        gpu_device_id = -1,
        gpu_use_dp = false,
        num_gpu = 1,
    ])

Return a LGBMRegression estimator.
"""
function LGBMRegression(;
    objective = "regression",
    boosting = "gbdt",
    num_iterations = 100,
    learning_rate = .1,
    num_leaves = 31,
    tree_learner = "serial",
    num_threads = 0,
    device_type="cpu",
    force_col_wise = false,
    force_row_wise = false,
    histogram_pool_size = -1.,
    max_depth = -1,
    min_data_in_leaf = 20,
    min_sum_hessian_in_leaf = 1e-3,
    bagging_fraction = 1.,
    bagging_freq = 0,
    bagging_seed = 3,
    feature_fraction = 1.,
    feature_fraction_bynode = 1.,
    feature_fraction_seed = 2,
    extra_trees = false,
    extra_seed = 6,
    early_stopping_round = 0,
    max_delta_step = 0.,
    lambda_l1 = 0.,
    lambda_l2 = 0.,
    min_gain_to_split = 0.,
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
    cat_l2 = 10.,
    cat_smooth = 10.,
    forcedsplits_filename = "",
    refit_decay_rate = 0.9,
    linear_tree = false,
    max_bin = 255,
    bin_construct_sample_cnt = 200000,
    data_random_seed = 1,
    is_enable_sparse = true,
    use_missing = true,
    feature_pre_filter = true,
    two_round = false,
    header = false,
    label_column = "",
    weight_column = "",
    ignore_column = "",
    categorical_feature = Int[],
    forcedbins_filename = "",
    precise_float_parser = false,
    start_iteration_predict = 0,
    num_iteration_predict = -1,
    predict_raw_score = false,
    predict_leaf_index = false,
    predict_contrib = false,
    predict_disable_shape_check = false,
    is_unbalance = false,
    boost_from_average = true,
    alpha = 0.9,
    metric = [""],
    metric_freq = 1,
    is_provide_training_metric = false,
    eval_at = Int[1, 2, 3, 4, 5],
    num_machines = 1,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_filename = "",
    gpu_platform_id = -1,
    gpu_device_id = -1,
    gpu_use_dp = false,
    num_gpu = 1,
)

    return LGBMRegression(
        Booster(), "", objective, boosting, num_iterations, learning_rate, num_leaves,
        tree_learner, num_threads, device_type, force_col_wise, force_row_wise, histogram_pool_size,
        max_depth, min_data_in_leaf, min_sum_hessian_in_leaf, 
        bagging_fraction, bagging_freq, bagging_seed, feature_fraction, feature_fraction_bynode, feature_fraction_seed, extra_trees,
        extra_seed, early_stopping_round, max_delta_step, lambda_l1, lambda_l2,
        min_gain_to_split, drop_rate, max_drop, skip_drop,
        xgboost_dart_mode, uniform_drop, drop_seed, top_rate, other_rate, min_data_per_group, max_cat_threshold,
        cat_l2, cat_smooth, forcedsplits_filename, refit_decay_rate, linear_tree, max_bin, bin_construct_sample_cnt, data_random_seed,
        is_enable_sparse, use_missing, feature_pre_filter, two_round, header, label_column, weight_column, ignore_column, categorical_feature, forcedbins_filename,
        precise_float_parser, start_iteration_predict, num_iteration_predict, predict_raw_score, predict_leaf_index, predict_contrib, predict_disable_shape_check, 
        1, is_unbalance, boost_from_average, alpha, metric, metric_freq, is_provide_training_metric, eval_at, num_machines, local_listen_port, time_out,
        machine_list_filename, gpu_platform_id, gpu_device_id, gpu_use_dp, num_gpu,
    )
end


mutable struct LGBMClassification <: LGBMEstimator
    booster::Booster
    model::String

    # Core parameters
    objective::String
    boosting :: String
    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    tree_learner::String
    num_threads::Int
    device_type::String

    # Learning control parameters
    force_col_wise::Bool
    force_row_wise::Bool
    histogram_pool_size::Float64
    max_depth::Int
    min_data_in_leaf::Int
    min_sum_hessian_in_leaf::Float64
    bagging_fraction::Float64
    pos_bagging_fraction::Float64
    neg_bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    feature_fraction::Float64
    feature_fraction_bynode::Float64
    feature_fraction_seed::Int
    extra_trees::Bool
    extra_seed::Int
    early_stopping_round::Int
    max_delta_step::Float64
    lambda_l1::Float64
    lambda_l2::Float64
    min_gain_to_split::Float64
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
    forcedsplits_filename::String
    refit_decay_rate::Float64

    # Dataset parameters
    linear_tree::Bool
    max_bin::Int
    bin_construct_sample_cnt::Int
    data_random_seed::Int
    is_enable_sparse::Bool
    use_missing::Bool
    feature_pre_filter::Bool
    two_round::Bool
    header::Bool
    label_column::String
    weight_column::String
    ignore_column::String
    categorical_feature::Vector{Int}
    forcedbins_filename::String
    precise_float_parser::Bool

    # Predict parameters
    start_iteration_predict::Int
    num_iteration_predict::Int
    predict_raw_score::Bool
    predict_leaf_index::Bool
    predict_contrib::Bool
    predict_disable_shape_check::Bool
    pred_early_stop::Bool
    pred_early_stop_freq::Int
    pred_early_stop_margin::Float64

    # Objective parameters
    num_class::Int
    is_unbalance::Bool
    scale_pos_weight::Float64
    sigmoid::Float64
    boost_from_average::Bool

    # Metric parameters
    metric::Vector{String}
    metric_freq::Int
    is_provide_training_metric::Bool
    eval_at::Vector{Int}

    # Network parameters
    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_filename::String

    # GPU parameters
    gpu_platform_id::Int
    gpu_device_id::Int
    gpu_use_dp::Bool
    num_gpu::Int
end

"""
    LGBMClassification(;[
        objective = "multiclass",
        boosting = "gbdt",
        num_iterations = 100,
        learning_rate = .1,
        num_leaves = 31,
        tree_learner = \"serial\",
        num_threads = 0,
        device_type=\"cpu\",
        force_col_wise = false,
        force_row_wise = false,
        histogram_pool_size = -1.,
        max_depth = -1,
        min_data_in_leaf = 20,
        min_sum_hessian_in_leaf = 1e-3,
        bagging_fraction = 1.,
        pos_bagging_fraction = 1.,
        neg_bagging_fraction = 1.,
        bagging_freq = 0,
        bagging_seed = 3,
        feature_fraction = 1.,
        feature_fraction_bynode = 1.,
        feature_fraction_seed = 2,
        extra_trees = false,
        extra_seed = 6,
        early_stopping_round = 0,
        max_delta_step = 0.,
        lambda_l1 = 0.,
        lambda_l2 = 0.,
        min_gain_to_split = 0.,
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
        cat_l2 = 10.,
        cat_smooth = 10.,
        forcedsplits_filename = "",
        refit_decay_rate = 0.9,
        linear_tree = false,
        max_bin = 255,
        bin_construct_sample_cnt = 200000,
        data_random_seed = 1,
        is_enable_sparse = true,
        use_missing = true,
        feature_pre_filter = true,
        two_round = false,
        header = false,
        label_column = "",
        weight_column = "",
        ignore_column = "",
        categorical_feature = Int[],
        forcedbins_filename = "",
        precise_float_parser = false,
        start_iteration_predict = 0,
        num_iteration_predict = -1,
        predict_raw_score = false,
        predict_leaf_index = false,
        predict_contrib = false,
        predict_disable_shape_check = false,
        pred_early_stop = false,
        pred_early_stop_freq = 10,
        pred_early_stop_margin = 10.0,
        num_class = 2,
        is_unbalance = false,
        scale_pos_weight = 1.0,
        sigmoid = 1.0,
        boost_from_average = true,
        metric = [""],
        metric_freq = 1,
        is_provide_training_metric = false,
        eval_at = Int[1, 2, 3, 4, 5],
        num_machines = 1,
        local_listen_port = 12400,
        time_out = 120,
        machine_list_filename = \"\",
        gpu_platform_id = -1,
        gpu_device_id = -1,
        gpu_use_dp = false,
        num_gpu = 1,
    ])

Return a LGBMClassification estimator.
"""
function LGBMClassification(;
    objective = "multiclass",
    boosting = "gbdt",
    num_iterations = 100,
    learning_rate = .1,
    num_leaves = 31,
    tree_learner = "serial",
    num_threads = 0,
    device_type="cpu",
    force_col_wise = false,
    force_row_wise = false,
    histogram_pool_size = -1.,
    max_depth = -1,
    min_data_in_leaf = 20,
    min_sum_hessian_in_leaf = 1e-3,
    bagging_fraction = 1.,
    pos_bagging_fraction = 1.,
    neg_bagging_fraction = 1.,
    bagging_freq = 0,
    bagging_seed = 3,
    feature_fraction = 1.,
    feature_fraction_bynode = 1.,
    feature_fraction_seed = 2,
    extra_trees = false,
    extra_seed = 6,
    early_stopping_round = 0,
    max_delta_step = 0.,
    lambda_l1 = 0.,
    lambda_l2 = 0.,
    min_gain_to_split = 0.,
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
    cat_l2 = 10.,
    cat_smooth = 10.,
    forcedsplits_filename = "",
    refit_decay_rate = 0.9,
    linear_tree = false,
    max_bin = 255,
    bin_construct_sample_cnt = 200000,
    data_random_seed = 1,
    is_enable_sparse = true,
    use_missing = true,
    feature_pre_filter = true,
    two_round = false,
    header = false,
    label_column = "",
    weight_column = "",
    ignore_column = "",
    categorical_feature = Int[],
    forcedbins_filename = "",
    precise_float_parser = false,
    start_iteration_predict = 0,
    num_iteration_predict = -1,
    predict_raw_score = false,
    predict_leaf_index = false,
    predict_contrib = false,
    predict_disable_shape_check = false,
    pred_early_stop = false,
    pred_early_stop_freq = 10,
    pred_early_stop_margin = 10.0,
    num_class = 2,
    is_unbalance = false,
    scale_pos_weight = 1.0,
    sigmoid = 1.0,
    boost_from_average = true,
    metric = [""],
    metric_freq = 1,
    is_provide_training_metric = false,
    eval_at = Int[1, 2, 3, 4, 5],
    num_machines = 1,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_filename = "",
    gpu_platform_id = -1,
    gpu_device_id = -1,
    gpu_use_dp = false,
    num_gpu = 1,
)

    return LGBMClassification(
        Booster(), "", objective, boosting, num_iterations, learning_rate,
        num_leaves, tree_learner, num_threads, device_type, force_col_wise, force_row_wise, histogram_pool_size,
        max_depth, min_data_in_leaf, min_sum_hessian_in_leaf, 
        bagging_fraction, pos_bagging_fraction, neg_bagging_fraction,bagging_freq,
        bagging_seed, feature_fraction, feature_fraction_bynode, feature_fraction_seed, extra_trees, extra_seed, early_stopping_round, max_delta_step, lambda_l1, lambda_l2,
        min_gain_to_split, drop_rate, max_drop, skip_drop, xgboost_dart_mode,
        uniform_drop, drop_seed, top_rate, other_rate, min_data_per_group, max_cat_threshold, cat_l2, cat_smooth, forcedsplits_filename, refit_decay_rate, linear_tree, max_bin, bin_construct_sample_cnt,
        data_random_seed, is_enable_sparse, use_missing, feature_pre_filter, two_round, header, label_column, weight_column, ignore_column, categorical_feature, forcedbins_filename,
        precise_float_parser, start_iteration_predict, num_iteration_predict, predict_raw_score, predict_leaf_index, predict_contrib,
        predict_disable_shape_check, pred_early_stop, pred_early_stop_freq, pred_early_stop_margin,
        num_class, is_unbalance, scale_pos_weight, sigmoid, boost_from_average,
        metric, metric_freq, is_provide_training_metric, eval_at, num_machines, local_listen_port, time_out,
        machine_list_filename, gpu_platform_id, gpu_device_id, gpu_use_dp, num_gpu,
    )
end

mutable struct LGBMRanking <: LGBMEstimator
    booster::Booster
    model::String

    # Core parameters
    objective::String
    boosting :: String
    num_iterations::Int
    learning_rate::Float64
    num_leaves::Int
    tree_learner::String
    num_threads::Int
    device_type::String

    # Learning control parameters
    force_col_wise::Bool
    force_row_wise::Bool
    histogram_pool_size::Float64
    max_depth::Int
    min_data_in_leaf::Int
    min_sum_hessian_in_leaf::Float64
    bagging_fraction::Float64
    pos_bagging_fraction::Float64
    neg_bagging_fraction::Float64
    bagging_freq::Int
    bagging_seed::Int
    feature_fraction::Float64
    feature_fraction_bynode::Float64
    feature_fraction_seed::Int
    extra_trees::Bool
    extra_seed::Int
    early_stopping_round::Int
    max_delta_step::Float64
    lambda_l1::Float64
    lambda_l2::Float64
    min_gain_to_split::Float64
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
    forcedsplits_filename::String
    refit_decay_rate::Float64

    # Dataset parameters
    linear_tree::Bool
    max_bin::Int
    bin_construct_sample_cnt::Int
    data_random_seed::Int
    is_enable_sparse::Bool
    use_missing::Bool
    feature_pre_filter::Bool
    two_round::Bool
    header::Bool
    label_column::String
    weight_column::String
    group_column::String
    ignore_column::String
    categorical_feature::Vector{Int}
    forcedbins_filename::String
    precise_float_parser::Bool

    # Predict parameters
    start_iteration_predict::Int
    num_iteration_predict::Int
    predict_raw_score::Bool
    predict_leaf_index::Bool
    predict_contrib::Bool
    predict_disable_shape_check::Bool
    pred_early_stop::Bool
    pred_early_stop_freq::Int
    pred_early_stop_margin::Float64

    # Objective parameters
    objective_seed::Int
    num_class::Int
    is_unbalance::Bool
    scale_pos_weight::Float64
    sigmoid::Float64
    boost_from_average::Bool
    lambdarank_truncation_level::Int
    lambdarank_norm::Bool
    label_gain::Vector{Int}

    # Metric parameters
    metric::Vector{String}
    metric_freq::Int
    is_provide_training_metric::Bool
    eval_at::Vector{Int}

    # Network parameters
    num_machines::Int
    local_listen_port::Int
    time_out::Int
    machine_list_filename::String

    # GPU parameters
    gpu_platform_id::Int
    gpu_device_id::Int
    gpu_use_dp::Bool
    num_gpu::Int
end

"""
    LGBMRanking(;[
        objective = "lambdarank",
        boosting = "gbdt",
        num_iterations = 100,
        learning_rate = .1,
        num_leaves = 31,
        tree_learner = \"serial\",
        num_threads = 0,
        device_type=\"cpu\",
        force_col_wise = false,
        force_row_wise = false,
        histogram_pool_size = -1.,
        max_depth = -1,
        min_data_in_leaf = 20,
        min_sum_hessian_in_leaf = 1e-3,
        bagging_fraction = 1.,
        pos_bagging_fraction = 1.,
        neg_bagging_fraction = 1.,
        bagging_freq = 0,
        bagging_seed = 3,
        feature_fraction = 1.,
        feature_fraction_bynode = 1.,
        feature_fraction_seed = 2,
        extra_trees = false,
        extra_seed = 6,
        early_stopping_round = 0,
        max_delta_step = 0.,
        lambda_l1 = 0.,
        lambda_l2 = 0.,
        min_gain_to_split = 0.,
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
        cat_l2 = 10.,
        cat_smooth = 10.,
        forcedsplits_filename = "",
        refit_decay_rate = 0.9,
        linear_tree = false,
        max_bin = 255,
        bin_construct_sample_cnt = 200000,
        data_random_seed = 1,
        is_enable_sparse = true,
        use_missing = true,
        feature_pre_filter = true,
        two_round = false,
        header = false,
        label_column = "",
        weight_column = "",
        group_column = "",
        ignore_column = "",
        categorical_feature = Int[],
        forcedbins_filename = "",
        precise_float_parser = false,
        start_iteration_predict = 0,
        num_iteration_predict = -1,
        predict_raw_score = false,
        predict_leaf_index = false,
        predict_contrib = false,
        predict_disable_shape_check = false,
        pred_early_stop = false,
        pred_early_stop_freq = 10,
        pred_early_stop_margin = 10.0,
        objective_seed = 5,
        num_class = 1,
        is_unbalance = false,
        scale_pos_weight = 1.0,
        sigmoid = 1.0,
        boost_from_average = true,
        lambdarank_truncation_level = 30,
        lambdarank_norm = true,
        label_gain = [2^i - 1 for i in 0:30],
        metric = [""],
        metric_freq = 1,
        is_provide_training_metric = false,
        eval_at = Int[1, 2, 3, 4, 5],
        num_machines = 1,
        local_listen_port = 12400,
        time_out = 120,
        machine_list_filename = \"\",
        gpu_platform_id = -1,
        gpu_device_id = -1,
        gpu_use_dp = false,
        num_gpu = 1,
    ])

Return a LGBMRanking estimator.
"""
function LGBMRanking(;
    objective = "lambdarank",
    boosting = "gbdt",
    num_iterations = 100,
    learning_rate = .1,
    num_leaves = 31,
    tree_learner = "serial",
    num_threads = 0,
    device_type="cpu",
    force_col_wise = false,
    force_row_wise = false,
    histogram_pool_size = -1.,
    max_depth = -1,
    min_data_in_leaf = 20,
    min_sum_hessian_in_leaf = 1e-3,
    bagging_fraction = 1.,
    pos_bagging_fraction = 1.,
    neg_bagging_fraction = 1.,
    bagging_freq = 0,
    bagging_seed = 3,
    feature_fraction = 1.,
    feature_fraction_bynode = 1.,
    feature_fraction_seed = 2,
    extra_trees = false,
    extra_seed = 6,
    early_stopping_round = 0,
    max_delta_step = 0.,
    lambda_l1 = 0.,
    lambda_l2 = 0.,
    min_gain_to_split = 0.,
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
    cat_l2 = 10.,
    cat_smooth = 10.,
    forcedsplits_filename = "",
    refit_decay_rate = 0.9,
    linear_tree = false,
    max_bin = 255,
    bin_construct_sample_cnt = 200000,
    data_random_seed = 1,
    is_enable_sparse = true,
    use_missing = true,
    feature_pre_filter = true,
    two_round = false,
    header = false,
    label_column = "",
    weight_column = "",
    group_column = "",
    ignore_column = "",
    categorical_feature = Int[],
    forcedbins_filename = "",
    precise_float_parser = false,
    start_iteration_predict = 0,
    num_iteration_predict = -1,
    predict_raw_score = false,
    predict_leaf_index = false,
    predict_contrib = false,
    predict_disable_shape_check = false,
    pred_early_stop = false,
    pred_early_stop_freq = 10,
    pred_early_stop_margin = 10.0,
    objective_seed = 5,
    num_class = 1,
    is_unbalance = false,
    scale_pos_weight = 1.0,
    sigmoid = 1.0,
    boost_from_average = true,
    lambdarank_truncation_level = 30,
    lambdarank_norm = true,
    label_gain = [2^i - 1 for i in 0:30],
    metric = ["ndcg"],
    metric_freq = 1,
    is_provide_training_metric = false,
    eval_at = Int[1, 2, 3, 4, 5],
    num_machines = 1,
    local_listen_port = 12400,
    time_out = 120,
    machine_list_filename = "",
    gpu_platform_id = -1,
    gpu_device_id = -1,
    gpu_use_dp = false,
    num_gpu = 1,
)

    return LGBMRanking(
        Booster(), "", objective, boosting, num_iterations, learning_rate,
        num_leaves, tree_learner, num_threads, device_type, force_col_wise, force_row_wise, histogram_pool_size,
        max_depth, min_data_in_leaf, min_sum_hessian_in_leaf, 
        bagging_fraction, pos_bagging_fraction, neg_bagging_fraction, bagging_freq,
        bagging_seed, feature_fraction, feature_fraction_bynode, feature_fraction_seed, extra_trees, extra_seed, early_stopping_round, max_delta_step, lambda_l1, lambda_l2,
        min_gain_to_split, drop_rate, max_drop, skip_drop, xgboost_dart_mode,
        uniform_drop, drop_seed, top_rate, other_rate, min_data_per_group, max_cat_threshold, cat_l2, cat_smooth, forcedsplits_filename, refit_decay_rate, linear_tree, max_bin, bin_construct_sample_cnt,
        data_random_seed, is_enable_sparse, use_missing, feature_pre_filter, two_round, header, label_column, weight_column, group_column, ignore_column, categorical_feature, forcedbins_filename,
        precise_float_parser, start_iteration_predict, num_iteration_predict, predict_raw_score, predict_leaf_index, predict_contrib,
        predict_disable_shape_check, pred_early_stop, pred_early_stop_freq, pred_early_stop_margin,
        objective_seed, num_class, is_unbalance, scale_pos_weight, sigmoid, boost_from_average, lambdarank_truncation_level, lambdarank_norm, label_gain,
        metric, metric_freq, is_provide_training_metric, eval_at, num_machines, local_listen_port, time_out,
        machine_list_filename, gpu_platform_id, gpu_device_id, gpu_use_dp, num_gpu,
    )
end
