# TODO: Change verbosity on LightGBM's side after a booster is created.

const C_API_PREDICT_NORMAL = 0
const C_API_PREDICT_RAW_SCORE = 1
const C_API_PREDICT_LEAF_INDEX = 2
const C_API_PREDICT_CONTRIB = 3


function get_predict_type(
    predict_raw_score::Bool, predict_leaf_index::Bool, predict_contrib::Bool,
    estimator::LGBMEstimator
)
    predict_type = C_API_PREDICT_NORMAL

    if predict_raw_score && !predict_leaf_index && !predict_contrib
        predict_type = C_API_PREDICT_RAW_SCORE
    elseif predict_leaf_index && !predict_raw_score && !predict_contrib
        predict_type = C_API_PREDICT_LEAF_INDEX
    elseif predict_contrib && !predict_raw_score && !predict_leaf_index
        predict_type = C_API_PREDICT_CONTRIB
    elseif estimator.predict_raw_score && !estimator.predict_leaf_index && !estimator.predict_contrib
        predict_type = C_API_PREDICT_RAW_SCORE
    elseif estimator.predict_leaf_index && !estimator.predict_raw_score && !estimator.predict_contrib
        predict_type = C_API_PREDICT_LEAF_INDEX
    elseif estimator.predict_contrib && !estimator.predict_raw_score && !estimator.predict_leaf_index
        predict_type = C_API_PREDICT_CONTRIB
    end

    return predict_type
end

"""
    predict(estimator, X; [num_iterations = -1, verbosity = 1,
    is_row_major = false, num_threads = -1, predict_raw_score = false, predict_leaf_index = false, predict_contrib = false])

Return a **MATRIX** with the labels that the `estimator` predicts for features data `X`.
Use `dropdims` if a vector is required.

# Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `X::Matrix{T<:Real}`: the features data.
* `predict_type::Integer`: keyword argument that controls the prediction type. `0` for normal
    scores with transform (if needed), `1` for raw scores, `2` for leaf indices, `3` for SHAP contributions.
* `num_iterations::Integer`: keyword argument that sets the number of iterations of the model to
    use in the prediction. `< 0` for all iterations.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
* `is_row_major::Bool`: keyword argument that indicates whether or not `X` is row-major. `true`
    indicates that it is row-major, `false` indicates that it is column-major (Julia's default).
* `num_threads::Integer`: keyword argument specifying the number of threads to use
    for prediction. Default is `-1` which reuses `num_threads` of the estimator.
* `predict_raw_score::Bool`: keyword argument that indicates whether or not to predict raw scores. Setting to true overrides the estimator's setting.
* `predict_leaf_index::Bool`: keyword argument that indicates whether or not to predict leaf indices. Setting to true overrides the estimator's setting.
* `predict_contrib::Bool`: keyword argument that indicates whether or not to predict SHAP contributions. Setting to true overrides the estimator's setting.

"""
function predict(
    estimator::LGBMEstimator, X::AbstractMatrix{TX};
    start_iteration::Integer = 0, num_iterations::Integer = -1, verbosity::Integer = 1,
    is_row_major::Bool = false, 
    num_threads::Integer = -1,
    predict_raw_score::Bool = false, predict_leaf_index::Bool = false, predict_contrib::Bool = false
)::Matrix{Float64} where TX <:Real

    tryload!(estimator)

    predict_type = get_predict_type(predict_raw_score, predict_leaf_index, predict_contrib, estimator)

    log_debug(verbosity, "Started predicting\n")

    if num_threads != -1
        parameter = "num_threads=$(num_threads)"
    else
        parameter = "num_threads=$(estimator.num_threads)"
    end

    if !(estimator isa LGBMRegression) && estimator.pred_early_stop
        parameter *= " pred_early_stop=true pred_early_stop_freq=$(estimator.pred_early_stop_freq) pred_early_stop_margin=$(estimator.pred_early_stop_margin)"
    end

    prediction = LGBM_BoosterPredictForMat(
        estimator.booster, X, predict_type, start_iteration, num_iterations, is_row_major, parameter
    )

    # This works the same one way or another because when n=1, (regression) reshaping is basically no-op
    # except for adding the extra dim
    prediction = transpose(reshape(prediction, estimator isa LGBMRegression ? 1 : estimator.num_class, :))

    return prediction

end


function predict_classes(
    estimator::LGBMClassification, X::AbstractMatrix{TX};
    num_iterations::Integer = -1, verbosity::Integer = 1,
    is_row_major::Bool = false, binary_threshold::Float64 = 0.5,
    num_threads::Integer = -1,
) where TX <:Real

    # pass through, get probabilities
    predicted_probabilities = predict(
        estimator, X; num_iterations=num_iterations,
        verbosity=verbosity, is_row_major=is_row_major, num_threads=num_threads,
    )

    # binary case
    if estimator.num_class == 1
        return Int.(predicted_probabilities .> binary_threshold)
    end

    return getindex.(argmax(predicted_probabilities, dims=2), 2) .- 1

end
