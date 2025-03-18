const LGBM_PARAMS_DOCS_LINK = "https://lightgbm.readthedocs.io/en/v3.3.5/Parameters.html"

"""
$(MLJModelInterface.doc_header(LGBMRegressor))

LightGBM, short for light gradient-boosting machine, is a
framework for gradient boosting based on decision tree algorithms and used for
classification, regression and other machine learning tasks, with a focus on
performance and scalability. This model in particular is used for various types of
regression tasks.

# Training data 

In MLJ or MLJBase, bind an instance `model` to data with 

  mach = machine(model, X, y) 

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of
  scitype `Continuous`; check the column scitypes with `schema(X)`; alternatively,
  `X` is any `AbstractMatrix` with `Continuous` elements; check the scitype with
  `scitype(X)`.
- y is a vector of targets whose items are of scitype `Continuous`. Check the
  scitype with `scitype(y)`.

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features
  `Xnew`, which should have the same scitype as `X` above.

# Hyper-parameters
See $LGBM_PARAMS_DOCS_LINK.

Currently, the following parameters and their defaults are supported:

- `boosting::String = "gbdt"`, 
- `num_iterations::Int = 100::(_ >= 0)`, 
- `learning_rate::Float64 = 0.1::(_ > 0.)`,
- `num_leaves::Int = 31::(1 < _ <= 131072)`, 
- `max_depth::Int = -1`, 
- `tree_learner::String = "serial"`,
- `histogram_pool_size::Float64 = -1.0`, 
- `min_data_in_leaf::Int = 20::(_ >= 0)`, 
- `min_sum_hessian_in_leaf::Float64 = 1e-3::(_ >= 0.0)`,
- `max_delta_step::Float64 = 0.0`, 
- `lambda_l1::Float64 = 0.0::(_ >= 0.0)`, 
- `lambda_l2::Float64 = 0.0::(_ >= 0.0)`,
- `min_gain_to_split::Float64 = 0.0::(_ >= 0.0)`, 
- `feature_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`,
- `feature_fraction_bynode::Float64 = 1.0::(0.0 < _ <= 1.0)`, 
- `feature_fraction_seed::Int = 2`, 
- `bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`,
- `pos_bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`, 
- `neg_bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`,
- `bagging_freq::Int = 0::(_ >= 0)`, 
- `bagging_seed::Int = 3`, 
- `early_stopping_round::Int = 0`, 
- `extra_trees::Bool = false`,
- `extra_seed::Int = 6`, 
- `max_bin::Int = 255::(_ > 1)`, 
- `bin_construct_sample_cnt = 200000::(_ > 0)`, 
- `drop_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)`,
- `max_drop::Int = 50`, 
- `skip_drop:: Float64 = 0.5::(0.0 <= _ <= 1)`, 
- `xgboost_dart_mode::Bool = false`, 
- `uniform_drop::Bool = false`, 
- `drop_seed::Int = 4`,
- `top_rate::Float64 = 0.2::(0.0 <= _ <= 1.0)`, 
- `other_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)`, 
- `min_data_per_group::Int = 100::(_ > 0)`,
- `max_cat_threshold::Int = 32::(_ > 0)`, 
- `cat_l2::Float64 = 10.0::(_ >= 0)`, 
- `cat_smooth::Float64 = 10.0::(_ >= 0)`, 
- `objective::String = "regression"`,
- `categorical_feature::Vector{Int} = Vector{Int}()`,
- `data_random_seed::Int = 1`, 
- `is_sparse::Bool = true`, 
- `is_unbalance::Bool = false`,
- `boost_from_average::Bool = true`, 
- `scale_pos_weight::Float64 = 1.0`, 
- `use_missing::Bool = true`, 
- `linear_tree::Bool = false`,
- `feature_pre_filter::Bool = true`, 
- `alpha::Float64 = 0.9::(_ > 0.0 )`, 
- `metric::Vector{String} = ["l2"]`, 
- `metric_freq::Int = 1::(_ > 0)`,
- `is_provide_training_metric::Bool = false`, 
- `eval_at::Vector{Int} = Vector{Int}([1, 2, 3, 4, 5])::(all(_ .> 0))`, 
- `num_machines::Int = 1::(_ > 0)`,
- `num_threads::Int  = 0::(_ >= 0)`, 
- `local_listen_port::Int = 12400::(_ > 0)`, 
- `time_out::Int = 120::(_ > 0)`, 
- `machine_list_file::String = ""`,
- `save_binary::Bool = false`, 
- `device_type::String = "cpu"`, 
- `gpu_use_dp::Bool = false`, 
- `gpu_platform_id::Int = -1`, 
- `gpu_device_id::Int = -1`,
- `num_gpu::Int = 1`, 
- `force_col_wise::Bool = false`, 
- `force_row_wise::Bool = false`, 
- `truncate_booster::Bool = true`.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `fitresult`: Fitted model information, contains a `LGBMRegression` object, an
  empty vector, and the regressor with all its parameters

# Report

The fields of `report(mach)` are:

- `training_metrics`: A dictionary containing all training metrics.
- `importance`: A `namedtuple` containing:
    - `gain`: The total gain of each split used by the model
    - `split`: The number of times each feature is used by the model.

# Examples

```julia

using DataFrames
using MLJ

# load the model
LGBMRegressor = @load LGBMRegressor pkg=LightGBM 

X, y = @load_boston # a table and a vector 
X = DataFrame(X)
train, test = partition(collect(eachindex(y)), 0.70, shuffle=true)

first(X, 3)
lgb = LGBMRegressor() # initialise a model with default params
mach = machine(lgb, X[train, :], y[train]) |> fit!

predict(mach, X[test, :])

# access feature importances
model_report = report(mach)
gain_importance = model_report.importance.gain
split_importance = model_report.importance.split
```

See also
[LightGBM.jl](https://github.com/IQVIA-ML/LightGBM.jl) and
the unwrapped model type
[`LightGBM.LGBMRegression`](@ref)
"""
LGBMRegressor


"""
$(MLJModelInterface.doc_header(LGBMClassifier))

`LightGBM, short for light gradient-boosting machine, is a
framework for gradient boosting based on decision tree algorithms and used for
classification and other machine learning tasks, with a focus on
performance and scalability. This model in particular is used for various types of
classification tasks.

# Training data In MLJ or MLJBase, bind an instance `model` to data with 

  mach = machine(model, X, y) 

Here:

- `X` is any table of input features (eg, a `DataFrame`) whose columns are of
  scitype `Continuous`; check the column scitypes with `schema(X)`; alternatively,
  `X` is any `AbstractMatrix` with `Continuous` elements; check the scitype with
  `scitype(X)`.
- y is a vector of targets whose items are of scitype `Continuous`. Check the
  scitype with scitype(y).

Train the machine using `fit!(mach, rows=...)`.

# Operations

- `predict(mach, Xnew)`: return predictions of the target given new features
  `Xnew`, which should have the same scitype as `X` above.

# Hyper-parameters
See $LGBM_PARAMS_DOCS_LINK.

Currently, the following parameters and their defaults are supported:

- `boosting::String = "gbdt"`, 
- `num_iterations::Int = 100::(_ >= 0)`, 
- `learning_rate::Float64 = 0.1::(_ > 0.)`, 
- `num_leaves::Int = 31::(1 < _ <= 131072)`, 
- `max_depth::Int = -1`, 
- `tree_learner::String = "serial"`,
- `histogram_pool_size::Float64 = -1.0`, 
- `min_data_in_leaf::Int = 20::(_ >= 0)`, 
- `min_sum_hessian_in_leaf::Float64 = 1e-3::(_ >= 0.0)`,
- `max_delta_step::Float64 = 0.0`, 
- `lambda_l1::Float64 = 0.0::(_ >= 0.0)`, 
- `lambda_l2::Float64 = 0.0::(_ >= 0.0)`,
- `min_gain_to_split::Float64 = 0.0::(_ >= 0.0)`, 
- `feature_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`,
- `feature_fraction_bynode::Float64 = 1.0::(0.0 < _ <= 1.0)`, 
- `feature_fraction_seed::Int = 2`, 
- `bagging_fraction::Float64 = 1.0::(0.0 < _ <= 1.0)`,
- `bagging_freq::Int = 0::(_ >= 0)`, 
- `bagging_seed::Int = 3`, 
- `early_stopping_round::Int = 0`, 
- `extra_trees::Bool = false`,
- `extra_seed::Int = 6`, 
- `max_bin::Int = 255::(_ > 1)`, 
- `bin_construct_sample_cnt = 200000::(_ > 0)`, 
- `drop_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)`,
- `max_drop::Int = 50`,
- `skip_drop:: Float64 = 0.5::(0.0 <= _ <= 1)`, 
- `xgboost_dart_mode::Bool = false`, 
- `uniform_drop::Bool = false`, 
- `drop_seed::Int = 4`,
- `top_rate::Float64 = 0.2::(0.0 <= _ <= 1.0)`, 
- `other_rate::Float64 = 0.1::(0.0 <= _ <= 1.0)`, 
- `min_data_per_group::Int = 100::(_ > 0)`,
- `max_cat_threshold::Int = 32::(_ > 0)`, 
- `cat_l2::Float64 = 10.0::(_ >= 0)`, 
- `cat_smooth::Float64 = 10.0::(_ >= 0)`, 
- `objective::String = "multiclass"`,
- `categorical_feature::Vector{Int} = Vector{Int}()`, 
- `data_random_seed::Int = 1`, 
- `is_sparse::Bool = true`, 
- `is_unbalance::Bool = false`,
- `boost_from_average::Bool = true`,
- `use_missing::Bool = true`, 
- `linear_tree::Bool = false`, 
- `feature_pre_filter::Bool = true`,
- `metric::Vector{String} = ["none"]`, 
- `metric_freq::Int = 1::(_ > 0)`, 
- `is_provide_training_metric::Bool = false`,
- `eval_at::Vector{Int} = Vector{Int}([1, 2, 3, 4, 5])::(all(_ .> 0))`, 
- `num_machines::Int = 1::(_ > 0)`, 
- `num_threads::Int  = 0::(_ >= 0)`,
- `local_listen_port::Int = 12400::(_ > 0)`, 
- `time_out::Int = 120::(_ > 0)`, 
- `machine_list_file::String = ""`, 
- `save_binary::Bool = false`,
- `device_type::String = "cpu"`, 
- `gpu_use_dp::Bool = false`, 
- `gpu_platform_id::Int = -1`, 
- `gpu_device_id::Int = -1`, 
- `num_gpu::Int = 1`,
- `force_col_wise::Bool = false`, 
- `force_row_wise::Bool = false`, 
- `truncate_booster::Bool = true`.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `fitresult`: Fitted model information, contains a `LGBMClassification` object, a
  `CategoricalArray` of the input class names, and the classifier with all its
  parameters

# Report

The fields of `report(mach)` are:

- `training_metrics`: A dictionary containing all training metrics.
- `importance`: A `namedtuple` containing:
    - `gain`: The total gain of each split used by the model
    - `split`: The number of times each feature is used by the model.


# Examples

```julia

using DataFrames
using MLJ

# load the model
LGBMClassifier = @load LGBMClassifier pkg=LightGBM 

X, y = @load_iris 
X = DataFrame(X)
train, test = partition(collect(eachindex(y)), 0.70, shuffle=true)

first(X, 3)
lgb = LGBMClassifier() # initialise a model with default params
mach = machine(lgb, X[train, :], y[train]) |> fit!

predict(mach, X[test, :])

# access feature importances
model_report = report(mach)
gain_importance = model_report.importance.gain
split_importance = model_report.importance.split
```

See also
[LightGBM.jl](https://github.com/IQVIA-ML/LightGBM.jl) and
the unwrapped model type
[`LightGBM.LGBMClassification`](@ref)

"""
LGBMClassifier