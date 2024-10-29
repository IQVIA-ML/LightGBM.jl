module TestParameters

using Test
using LightGBM


NSAMPLES_TRAIN = 2000
NSAMPLES_TEST = 500
NFEATURES = 20

X_train = randn(NSAMPLES_TRAIN, NFEATURES)
y_train_binary = rand(0:1, NSAMPLES_TRAIN)
y_train_regression = rand(NSAMPLES_TRAIN)
X_test = randn(NSAMPLES_TEST, NFEATURES)


norm(x) = sqrt(sum(x .^ 2))

@testset "parameters -- boosting" begin

    # The scheme is to set up an old style gbdt and a new DART and show that the learned
    # models are "diffferent", i.e. the parameter passed through and made a difference
    # Check for both classification and regression
    # Why do we use num_iterations 12? well, GOSS doesn't "work" until after 1/learning_rate
    # and default learning rate is 0.1, so we need to make sure models are going for at least
    # enough time to be different.

    classifier_gdbt = LightGBM.LGBMClassification(boosting = "gbdt", num_iterations=12)
    classifier_dart = LightGBM.LGBMClassification(boosting = "dart", num_iterations=12)
    classifier_goss = LightGBM.LGBMClassification(boosting = "goss", num_iterations=12)
    regressor_gbdt = LightGBM.LGBMRegression(boosting = "gbdt", num_iterations=12)
    regressor_dart = LightGBM.LGBMRegression(boosting = "dart", num_iterations=12)
    regressor_goss = LightGBM.LGBMRegression(boosting = "goss", num_iterations=12)

    LightGBM.fit!(classifier_dart, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(classifier_gdbt, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(classifier_goss, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(regressor_gbdt, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(regressor_dart, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(regressor_goss, X_train, y_train_regression, verbosity = -1)

    p_gbdt = LightGBM.predict(classifier_gdbt, X_test, verbosity = -1)
    p_dart = LightGBM.predict(classifier_dart, X_test, verbosity = -1)
    p_goss = LightGBM.predict(classifier_goss, X_test, verbosity = -1)
    r_gbdt = LightGBM.predict(regressor_gbdt, X_test, verbosity = -1)
    r_dart = LightGBM.predict(regressor_dart, X_test, verbosity = -1)
    r_goss = LightGBM.predict(regressor_goss, X_test, verbosity = -1)

    # checking the models are sufficiently different by making sure
    # that the distance between the vectors is relatively large
    @test norm(p_dart .- p_gbdt) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(p_goss .- p_gbdt) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(p_dart .- p_goss) >= sqrt(NSAMPLES_TEST) * 1e-5
    # Check also for regression
    @test norm(r_dart .- r_gbdt) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(r_goss .- r_gbdt) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(r_dart .- r_goss) >= sqrt(NSAMPLES_TEST) * 1e-5

end


@testset "parameters -- dart classifcation" begin

    # the scheme is to set up a DART classifier and show that changing each one of
    # the parameters individually results in a difference to the model

    dart_default = LightGBM.LGBMClassification(boosting = "dart")
    dart_drop_rate = LightGBM.LGBMClassification(boosting = "dart", drop_rate = 0.9)
    dart_max_drop = LightGBM.LGBMClassification(boosting = "dart", max_drop = 1)
    dart_skip_drop = LightGBM.LGBMClassification(boosting = "dart", skip_drop = 1.0)
    dart_xdm = LightGBM.LGBMClassification(boosting = "dart", xgboost_dart_mode = true)
    dart_uniform_drop = LightGBM.LGBMClassification(boosting = "dart", uniform_drop = true)
    dart_drop_seed = LightGBM.LGBMClassification(boosting = "dart", drop_seed = 20)

    LightGBM.fit!(dart_default, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_drop_rate, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_max_drop, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_skip_drop, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_xdm, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_uniform_drop, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_drop_seed, X_train, y_train_binary, verbosity = -1)

    p_dart_default = LightGBM.predict(dart_default, X_test, verbosity = -1)
    p_dart_drop_rate = LightGBM.predict(dart_drop_rate, X_test, verbosity = -1)
    p_dart_max_drop = LightGBM.predict(dart_max_drop, X_test, verbosity = -1)
    p_dart_skip_drop = LightGBM.predict(dart_skip_drop, X_test, verbosity = -1)
    p_dart_xdm = LightGBM.predict(dart_xdm, X_test, verbosity = -1)
    p_dart_uniform_drop = LightGBM.predict(dart_uniform_drop, X_test, verbosity = -1)
    p_dart_drop_seed = LightGBM.predict(dart_drop_seed, X_test, verbosity = -1)

    @test norm(p_dart_default .- p_dart_drop_rate) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(p_dart_default .- p_dart_max_drop) >= sqrt(NSAMPLES_TEST) * 1e-10
    @test norm(p_dart_default .- p_dart_skip_drop) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(p_dart_default .- p_dart_xdm) >= sqrt(NSAMPLES_TEST) * 1e-10
    @test norm(p_dart_default .- p_dart_uniform_drop) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(p_dart_default .- p_dart_drop_seed) >= sqrt(NSAMPLES_TEST) * 1e-5

end


@testset "parameters -- dart regression" begin

    # the scheme is to set up a DART regressor and show that changing each one of
    # the parameters individually results in a difference to the model

    dart_default = LightGBM.LGBMRegression(boosting = "dart")
    dart_drop_rate = LightGBM.LGBMRegression(boosting = "dart", drop_rate = 0.9)
    dart_max_drop = LightGBM.LGBMRegression(boosting = "dart", max_drop = 1)
    dart_skip_drop = LightGBM.LGBMRegression(boosting = "dart", skip_drop = 1.0)
    dart_xdm = LightGBM.LGBMRegression(boosting = "dart", xgboost_dart_mode = true)
    dart_uniform_drop = LightGBM.LGBMRegression(boosting = "dart", uniform_drop = true)
    dart_drop_seed = LightGBM.LGBMRegression(boosting = "dart", drop_seed = 20)

    LightGBM.fit!(dart_default, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_drop_rate, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_max_drop, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_skip_drop, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_xdm, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_uniform_drop, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_drop_seed, X_train, y_train_regression, verbosity = -1)

    r_dart_default = LightGBM.predict(dart_default, X_test, verbosity = -1)
    r_dart_drop_rate = LightGBM.predict(dart_drop_rate, X_test, verbosity = -1)
    r_dart_max_drop = LightGBM.predict(dart_max_drop, X_test, verbosity = -1)
    r_dart_skip_drop = LightGBM.predict(dart_skip_drop, X_test, verbosity = -1)
    r_dart_xdm = LightGBM.predict(dart_xdm, X_test, verbosity = -1)
    r_dart_uniform_drop = LightGBM.predict(dart_uniform_drop, X_test, verbosity = -1)
    r_dart_drop_seed = LightGBM.predict(dart_drop_seed, X_test, verbosity = -1)

    @test norm(r_dart_default .- r_dart_drop_rate) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(r_dart_default .- r_dart_max_drop) >= sqrt(NSAMPLES_TEST) * 1e-10
    @test norm(r_dart_default .- r_dart_skip_drop) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(r_dart_default .- r_dart_xdm) >= sqrt(NSAMPLES_TEST) * 1e-10
    @test norm(r_dart_default .- r_dart_uniform_drop) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(r_dart_default .- r_dart_drop_seed) >= sqrt(NSAMPLES_TEST) * 1e-5

end

@testset "parameters -- goss classifcation" begin

    # the scheme is to set up a goss classifier and show that changing each one of
    # the parameters individually results in a difference to the model

    goss_default = LightGBM.LGBMClassification(boosting = "goss", num_iterations=12)
    goss_top_rate = LightGBM.LGBMClassification(boosting = "goss", num_iterations=12, top_rate = 0.01)
    goss_other_rate = LightGBM.LGBMClassification(boosting = "goss", num_iterations=12, other_rate = 0.01)


    LightGBM.fit!(goss_default, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(goss_top_rate, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(goss_other_rate, X_train, y_train_binary, verbosity = -1)


    p_goss_default = LightGBM.predict(goss_default, X_test, verbosity = -1)
    p_goss_top_rate = LightGBM.predict(goss_top_rate, X_test, verbosity = -1)
    p_goss_other_rate = LightGBM.predict(goss_other_rate, X_test, verbosity = -1)


    @test norm(p_goss_default .- p_goss_top_rate) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(p_goss_default .- p_goss_other_rate) >= sqrt(NSAMPLES_TEST) * 1e-5

end


@testset "parameters -- goss regression" begin

    # the scheme is to set up a goss regressor and show that changing each one of
    # the parameters individually results in a difference to the model

    goss_default = LightGBM.LGBMRegression(boosting = "goss", num_iterations=12)
    goss_top_rate = LightGBM.LGBMRegression(boosting = "goss", num_iterations=12, top_rate = 0.01)
    goss_other_rate = LightGBM.LGBMRegression(boosting = "goss", num_iterations=12, other_rate = 0.01)


    LightGBM.fit!(goss_default, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(goss_top_rate, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(goss_other_rate, X_train, y_train_regression, verbosity = -1)


    r_goss_default = LightGBM.predict(goss_default, X_test, verbosity = -1)
    r_goss_top_rate = LightGBM.predict(goss_top_rate, X_test, verbosity = -1)
    r_goss_other_rate = LightGBM.predict(goss_other_rate, X_test, verbosity = -1)


    @test norm(r_goss_default .- r_goss_top_rate) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(r_goss_default .- r_goss_other_rate) >= sqrt(NSAMPLES_TEST) * 1e-5

end


@testset "parameters -- prediction" begin
    # Generate random data
    X_train = randn(1000, 20)
    y_train = rand([0, 1], 1000)

    # Define combinations of parameters
    combinations = [
        # predict_raw_score
        (true, false, false),
        # predict_leaf_index
        (false, true, false),
        # predict_contrib
        (false, false, true),
        # predict_normal
        (false, false, false),
        # predict_normal
        (true, true, true)
    ]

    # Function to create and fit estimators for each combination
    function fit_estimators(combinations, model_type, objective)
        estimators = []
        for (predict_raw_score, predict_leaf_index, predict_contrib) in combinations
            estimator = model_type(
                objective = objective,
                start_iteration_predict = 0,
                num_iteration_predict = -1,
                predict_raw_score = predict_raw_score,
                predict_leaf_index = predict_leaf_index,
                predict_contrib = predict_contrib,
                predict_disable_shape_check = false,
                num_iterations = 100,
            )
            if model_type == LightGBM.LGBMClassification
                estimator.num_class = 1
            end
            LightGBM.fit!(estimator, X_train, y_train, verbosity = -1)
            push!(estimators, estimator)
        end
        return estimators
    end

    # Function to generate predictions for each estimator
    function generate_predictions(models)
        predictions = []
        for model in models
            prediction = LightGBM.predict(model, X_train, verbosity = -1)
            push!(predictions, prediction)
        end
        return predictions
    end

    # Fit classifiers and regressors and generate predictions
    classifiers = fit_estimators(combinations, LightGBM.LGBMClassification, "binary")
    regressors = fit_estimators(combinations, LightGBM.LGBMRegression, "regression")
    regressors_poisson = fit_estimators(combinations, LightGBM.LGBMRegression, "poisson")
    classifier_predictions = generate_predictions(classifiers)
    regressor_predictions = generate_predictions(regressors)
    regressor_poisson_predictions = generate_predictions(regressors_poisson)

    # Test prediction outputs for different parameters for classifier
    @testset "Classifier predict parameters" begin
        # 4 and 5 should be the same as they are both predict_normal
        @test classifier_predictions[4] == classifier_predictions[5]
        # 1, 2, 3 should be different as they are different predict types
        # 1 is predict_raw_score, 2 is predict_leaf_index, 3 is predict_contrib
        @test classifier_predictions[1] != classifier_predictions[2]
        @test classifier_predictions[1] != classifier_predictions[3]
        @test classifier_predictions[2] != classifier_predictions[3]
        # 1, 2, 3 should not be equal to 4 (or 5)
        @test classifier_predictions[1] != classifier_predictions[4]
        @test classifier_predictions[2] != classifier_predictions[4]
        @test classifier_predictions[3] != classifier_predictions[4]
    end

    # Test prediction outputs for different parameters for regressor
    @testset "Regressor predict parameters with regression objective" begin
        # 4 and 5 should be the same as they are both predict_normal
        @test regressor_predictions[4] == regressor_predictions[5]
        # 1, 2, 3 should be different as they are different predict types
        # 1 is predict_raw_score, 2 is predict_leaf_index, 3 is predict_contrib
        @test regressor_predictions[1] != regressor_predictions[2]
        @test regressor_predictions[1] != regressor_predictions[3]
        @test regressor_predictions[2] != regressor_predictions[3]
        # 2 and 3 should not be equal to 4 (or 5)
        @test regressor_predictions[2] != regressor_predictions[4]
        @test regressor_predictions[3] != regressor_predictions[4]
        # 1 should be the same as 4 (or 5) as for "regression" objective
        # there is no transformation so predict_raw_score is the same as predict_normal
        @test regressor_predictions[1] == regressor_predictions[4]
    end

    # Test prediction outputs for different parameters for regressor with poisson objective
    @testset "Regressor predict parameters with poisson objective" begin
        # 4 and 5 should be the same as they are both predict_normal
        @test regressor_poisson_predictions[4] == regressor_poisson_predictions[5]
        # 1, 2, 3 should be different as they are different predict types
        # 1 is predict_raw_score, 2 is predict_leaf_index, 3 is predict_contrib
        @test regressor_poisson_predictions[1] != regressor_poisson_predictions[2]
        @test regressor_poisson_predictions[1] != regressor_poisson_predictions[3]
        @test regressor_poisson_predictions[2] != regressor_poisson_predictions[3]
        # 1, 2 and 3 should not be equal to 4 (or 5)
        @test regressor_poisson_predictions[1] != regressor_poisson_predictions[4]
        @test regressor_poisson_predictions[2] != regressor_poisson_predictions[4]
        @test regressor_poisson_predictions[3] != regressor_poisson_predictions[4]
    end

    @testset "Predict parameters precedence" begin
    # Create a simple estimator with default predict params set to false
        estimator = LightGBM.LGBMClassification(
            objective = "binary",
            predict_raw_score = false,
            predict_leaf_index = false,
            predict_contrib = false,
            num_class = 1
        )
        estimator_raw = LightGBM.LGBMClassification(
            objective = "binary",
            predict_raw_score = true,
            num_class = 1
        )
        LightGBM.fit!(estimator, X_train, y_train, verbosity = -1)
        LightGBM.fit!(estimator_raw, X_train, y_train, verbosity = -1)
    
        # Generate normal predictions (with default predict params)
        prediction_default = LightGBM.predict(estimator, X_train, verbosity = -1)
        prediction_default_raw = LightGBM.predict(estimator_raw, X_train, verbosity = -1)
    
        # Generate predictions with predict_raw_score set to true
        prediction_raw = LightGBM.predict(estimator, X_train; verbosity = -1, predict_raw_score = true)
        prediction_leaf = LightGBM.predict(estimator, X_train; verbosity = -1, predict_leaf_index = true)
        prediction_contrib = LightGBM.predict(estimator, X_train; verbosity = -1, predict_contrib = true)
        prediction_default_raw = LightGBM.predict(estimator_raw, X_train; verbosity = -1)
    
        # Test that the default predictions are different (params passed to predict function have precedence)
        @test prediction_default != prediction_raw
        @test prediction_default != prediction_leaf
        @test prediction_default != prediction_contrib
        # Test that the default predictions with estimator predict_raw_score but no additional predict param 
        # are the same as the predictions with predict_raw_score set to true
        @test prediction_default_raw == prediction_raw
    end
end


@testset "parameters -- prediction with early stopping" begin
    # Generate random data
    X_train = randn(1000, 20)
    y_train_classifier = rand([0, 1], 1000)

    # Define combinations of parameters for early stopping
    # (pred_early_stop, pred_early_stop_freq, pred_early_stop_margin)
    combinations = [
        # No early stopping case (full predictions)
        (false, 0, 0.0),
        # Early stopping with very low margin case which means very fast predictions but they will be less accurate
        # Such predictions and predicted probabilities will be different from full predictions and predicted probabilities
        (true, 10, 0.1),
        # High margin case (predictions with a 0.5 binary threshold will be the same as full predictions but predicted probabilities will differ)
        (true, 10, 5.5),
        # High frequency case (both predictions and predicted probabilities will be the same as full predictions given early stop frequency is 100 and 100 iterations)
        (true, 100, 0.2),
    ]

    # Function to create and fit estimators for each combination
    function fit_estimators(combinations, model_type, objective)
        estimators = []
        for (pred_early_stop, pred_early_stop_freq, pred_early_stop_margin) in combinations
            estimator = model_type(
                objective = objective,
                num_class = 1,
                pred_early_stop = pred_early_stop,
                pred_early_stop_freq = pred_early_stop_freq,
                pred_early_stop_margin = pred_early_stop_margin,
                num_iterations = 100,
            )
            LightGBM.fit!(estimator, X_train, y_train_classifier, verbosity = -1)
            push!(estimators, estimator)
        end
        return estimators
    end

    # Function to generate predictions and predicted probabilities for each estimator
    function generate_predictions(models)
        predictions = []
        predicted_probabilities = []
        for model in models
            prediction = LightGBM.predict_classes(model, X_train, binary_threshold=0.5, verbosity = -1)
            predicted_probability = LightGBM.predict(model, X_train, verbosity = -1)
            push!(predictions, prediction)
            push!(predicted_probabilities, predicted_probability)
        end
        return predictions, predicted_probabilities
    end

    # Fit classifiers and generate predictions
    classifiers = fit_estimators(combinations, LightGBM.LGBMClassification, "binary")
    classifier_predictions, classifier_predicted_probabilities = generate_predictions(classifiers)

    # Test prediction outputs for different parameters for classifier
    @testset "Classifier predict parameters with early stopping" begin
        # Predictions and predicted probabilities with early stopping will be different from full predictions and predicted probabilities
        # for very low pred_early_stop_margin as the speed of predictions will be prioritized over accuracy
        @test all(classifier_predictions[1] != classifier_predictions[2])
        @test all(classifier_predicted_probabilities[1] != classifier_predicted_probabilities[2])
        # High margin case should give the same predictions with a threshold of 0.5 but the predicted probabilities values will differ from full predicted probabilities
        @test all(classifier_predictions[1] == classifier_predictions[3])
        @test all(classifier_predicted_probabilities[1] != classifier_predicted_probabilities[3])
        # High frequency case given 100 iterations and 100 frequency check should give the same predictions and predicted probabilities as full predictions and predicted probabilities
        @test classifier_predictions[1] == classifier_predictions[4]
        @test classifier_predicted_probabilities[1] == classifier_predicted_probabilities[4]
        # Ensure all predicted probabilities are within 0 and 1
        @test all(0.0 .<= prob <= 1.0 for prob in vcat(classifier_predicted_probabilities...))
    end
end


@testset "parameters -- monotone constraints" begin

    # Generate random data
    X_train = randn(1000, 20)
    y_train = rand([0, 1], 1000)
    X_test = randn(500, 20)

    # Define the parameters for monotone constraints
    # The constraints are defined for each feature in the dataset
    # The pattern below is for 20 features where the first feature has a positive monotone constraint,
    # the second feature has a negative monotone constraint, and the third feature has no constraint.
    monotone_constraints = [1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]
    monotone_constraints_method = "basic"
    monotone_penalty = 0.1

    # Create and fit the estimator with monotone constraints
    estimator_with_constraints = LightGBM.LGBMClassification(
        objective = "binary",
        monotone_constraints = monotone_constraints,
        monotone_constraints_method = monotone_constraints_method,
        monotone_penalty = monotone_penalty,
        num_class = 1
    )
    LightGBM.fit!(estimator_with_constraints, X_train, y_train, verbosity = -1)

    # Create and fit the estimator without monotone constraints
    estimator_without_constraints = LightGBM.LGBMClassification(objective = "binary", num_class = 1)
    LightGBM.fit!(estimator_without_constraints, X_train, y_train, verbosity = -1)

    # Generate predictions
    prediction_with_constraints = LightGBM.predict(estimator_with_constraints, X_test, verbosity = -1)
    prediction_without_constraints = LightGBM.predict(estimator_without_constraints, X_test, verbosity = -1)

    # Test that the predictions are different
    @test prediction_with_constraints != prediction_without_constraints

end

@testset "parameters -- refit with refit decay rate" begin
    # Create sample data, labels and estimator
    featuresdata = randn(1000, 20)
    labels = rand([0, 1], 1000)
    estimator = LightGBM.LGBMClassification(objective = "binary", num_class = 1)
    LightGBM.fit!(estimator, featuresdata, labels, verbosity = -1)
    
    # Refit with a default refit_decay_rate
    new_booster = LightGBM.refit(estimator, featuresdata, labels)
    # Refit with a custom refit_decay_rate
    new_booster_custom = LightGBM.refit(estimator, featuresdata, labels, refit_decay_rate = 0.5)
    
    # Verify the returned booster
    @test new_booster != nothing
    @test new_booster != estimator
    @test new_booster_custom != new_booster

end


end # end module
