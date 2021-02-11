
const INDEXPARAMS = [:categorical_feature]

const MAXIMIZE_METRICS = ["auc", "ndcg", "average_precision"]

# LOGGING Funcs
function log_fatal(verbosity::Integer, msg...)
    warn(msg...)
end

function log_warning(verbosity::Integer, msg...)
    verbosity >= 0 && warn(msg...)
end

function log_info(verbosity::Integer, msg...)
    verbosity >= 1 && print(msg...)
end

function log_debug(verbosity::Integer, msg...)
    verbosity >= 2 && print(msg...)
end


"""
    savemodel(estimator, filename; [num_iteration = -1])

Save the fitted model in `estimator` as `filename`.

# Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `filename::String`: the name of the file to save the model in.
* `num_iteration::Integer`: keyword argument that sets the number of iterations of the model that
    should be saved. `< 0` for all iterations.
* `start_iteration` : : Start index of the iteration that should be saved.
* `feature_importance_type` : Type of feature importance,
    can be C_API_FEATURE_IMPORTANCE_SPLIT or C_API_FEATURE_IMPORTANCE_GAIN
"""
function savemodel(
    estimator::LGBMEstimator,
    filename::String;
    num_iteration::Integer=-1,
    start_iteration::Integer=0,
    feature_importance_type::Integer=0
)
    @assert(estimator.booster.handle != C_NULL, "Estimator does not contain a fitted model.")
    LGBM_BoosterSaveModel(estimator.booster, start_iteration, num_iteration, feature_importance_type, filename)
    return nothing
end

"""
    loadmodel!(estimator, filename)

Load the fitted model `filename` into `estimator`. Note that this only loads the fitted model—not
the parameters or data of the estimator whose model was saved as `filename`.

# Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `filename::String`: the name of the file that contains the model.
"""
function loadmodel!(estimator::LGBMEstimator, filename::String)
    estimator.booster = LGBM_BoosterCreateFromModelfile(filename)
    estimator.num_class = LGBM_BoosterGetNumClasses(estimator.booster)
    return nothing
end


function tryload!(estimator::LGBMEstimator)

    if estimator.booster.handle == C_NULL
        # first check for a serialised model
        if length(estimator.model) == 0
            throw(ErrorException("Estimator does not contain a fitted model."))
        end
        # load it
        estimator.booster = LGBM_BoosterLoadModelFromString(estimator.model)
    end

    return nothing
end


function get_iter_number(estimator::LGBMEstimator)

    if estimator.booster.handle == C_NULL
        # We cannot call LGBM_BoosterGetCurrentIteration without an initialised booster
        throw(ErrorException("Estimator does not contain any form of booster"))
    end

    return LGBM_BoosterGetCurrentIteration(estimator.booster)

end


function feature_importance_wrapper(estimator::LGBMEstimator, importance_type::Integer, num_iteration::Integer)
    # reason why main func isn't marked as a mutator because it isnt an "important" mutation
    tryload!(estimator)
    return LGBM_BoosterFeatureImportance(estimator.booster, num_iteration, importance_type)

end

"""
    gain_importance(estimator, num_iteration)
    gain_importance(estimator)

    Returns the importance of a fitted booster in terms of information gain across
    all boostings, or up to `num_iteration` boostings
"""
gain_importance(estimator::LGBMEstimator, num_iteration::Integer) = feature_importance_wrapper(estimator, 1, num_iteration)
gain_importance(estimator::LGBMEstimator) = feature_importance_wrapper(estimator, 1, 0)


"""
    split_importance(estimator, num_iteration)
    split_importance(estimator)

    Returns the importance of a fitted booster in terms of number of times feature was
    used in a split across all boostings, or up to `num_iteration` boostings
"""
split_importance(estimator::LGBMEstimator, num_iteration::Integer) = feature_importance_wrapper(estimator, 0, num_iteration)
split_importance(estimator::LGBMEstimator) = feature_importance_wrapper(estimator, 0, 0)
