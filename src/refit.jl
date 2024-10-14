function refit(estimator::LGBMEstimator, featuresdata::AbstractMatrix{TX}, label::Vector{Ty};
    refit_decay_rate::Float64=0.9)::LGBMEstimator where {TX<:Real,Ty<:Real}
    """
    Refit the existing Booster by new data.

    # Arguments

    * `featuredata::AbstractMatrix{TX}`: Features data for refit.
    * `label::Vector{Ty}`: Label for refit.
    * `refit_decay_rate::Float64`: optional (default=0.9) Decay rate of refit, 
    will use `leaf_output = decay_rate * old_leaf_output + (1.0 - decay_rate) * new_leaf_output` to refit trees.

    Returns
    result : Booster
        Refitted Booster.
    """

    # get leaf predictions for the provided estimator on new data
    leaf_preds = predict(estimator, featuresdata; predict_leaf_index = true)
    # Convert to Int32 which is expected by the C API and should be for leaf indices
    # Reshape the leaf predictions as their number of rows is number of data points/rows * num_trees/iterations
    reshaped_leaf_predictions = reshape(convert(Matrix{Int32}, leaf_preds), (size(featuresdata, 1), estimator.num_iterations))

    # check if the model is linear
    if LGBM_BoosterGetLinear(estimator.booster) == 1
       estimator.linear_tree = true
    end

    ds_parameters = stringifyparams(estimator)
    train_dataset = LGBM_DatasetCreateFromMat(featuresdata, ds_parameters)
    LGBM_DatasetSetField(train_dataset, "label", label)
    estimator.refit_decay_rate = refit_decay_rate
    new_estimator = deepcopy(estimator)
    new_params = stringifyparams(new_estimator)
    new_estimator.booster = LGBM_BoosterCreate(train_dataset, new_params)

    # Merge the new booster with the existing booster
    LGBM_BoosterMerge(new_estimator.booster, estimator.booster)
    # Refit the merged booster with the new leaf predictions
    LGBM_BoosterRefit(new_estimator.booster, reshaped_leaf_predictions)

    return new_estimator
end