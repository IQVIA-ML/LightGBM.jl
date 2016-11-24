# Predict using the C-API of LightGBM.
# TODO: request option to change verbosity on LightGBM's side after a booster is created.
function api_predict{TX<:Real}(estimator::LGBMEstimator, X::Matrix{TX};
                               predict_type::Integer = 0,
                               n_trees::Integer = -1, verbosity::Integer = 1)
    @assert(estimator.booster.handle != C_NULL, "Estimator does not contain a fitted model.")

    log_debug(verbosity, "Started predicting\n")
    prediction = LGBM_BoosterPredictForMat(estimator.booster, X, predict_type, n_trees)

    return prediction
end
