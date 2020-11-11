# TODO: Change verbosity on LightGBM's side after a booster is created.
"""
    predict(estimator, X; [predict_type = 0, num_iterations = -1, verbosity = 1,
    is_row_major = false])

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

One can obtain some form of feature importances by averaging SHAP contributions across predictions, i.e.
`mean(LightGBM.predict(estimator, X; predict_type=3); dims=1)`
"""
function predict(
    estimator::LGBMEstimator, X::AbstractMatrix{TX}; predict_type::Integer = 0,
    start_iteration::Integer = 0, num_iterations::Integer = -1, verbosity::Integer = 1,
    is_row_major::Bool = false,
)::Matrix{Float64} where TX <:Real

    tryload!(estimator)

    log_debug(verbosity, "Started predicting\n")

    prediction = LGBM_BoosterPredictForMat(
        estimator.booster, X, predict_type, start_iteration, num_iterations, is_row_major
    )

    # This works the same one way or another because when n=1, (regression) reshaping is basically no-op
    # except for adding the extra dim
    prediction = transpose(reshape(prediction, estimator.num_class, :))

    return prediction

end


function predict_classes(
    estimator::LGBMClassification, X::AbstractMatrix{TX}; predict_type::Integer = 0,
    num_iterations::Integer = -1, verbosity::Integer = 1,
    is_row_major::Bool = false, binary_threshold::Float64 = 0.5
) where TX <:Real

    # pass through, get probabilities
    predicted_probabilities = predict(
        estimator, X; predict_type=predict_type, num_iterations=num_iterations,
        verbosity=verbosity, is_row_major=is_row_major,
    )

    # binary case
    if estimator.num_class == 1
        return Int.(predicted_probabilities .> binary_threshold)
    end

    return getindex.(argmax(predicted_probabilities, dims=2), 2) .- 1

end
