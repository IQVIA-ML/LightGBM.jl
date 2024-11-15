module TestUtils

using Test
using LightGBM

X_train = randn(1000, 20)
y_train_binary = rand(0:1, 1000)
y_train_regression = rand(1000)


@testset "loadmodel predicts same as original model -- regression" begin
    # Arrange
    estimator = LightGBM.LGBMRegression(objective = "regression", verbosity = -1)
    LightGBM.fit!(estimator, X_train, y_train_regression; verbosity=-1)
    expected_prediction = predict(estimator, X_train)
    # Save the fitted model.
    model_filename = joinpath(@__DIR__, "fixture.model")
    savemodel(estimator, model_filename)
  
    # Act
    estimator_from_file = LGBMRegression(verbosity = -1)
    loadmodel!(estimator_from_file, model_filename)
    actual_prediction = predict(estimator_from_file, X_train)

    # Assert
    @test expected_prediction == actual_prediction

    # Teardown
    rm(model_filename)

end


@testset "loadmodel predicts same as original model -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(objective = "binary", num_class = 1, verbosity = -1)
    LightGBM.fit!(estimator, X_train, y_train_binary; verbosity=-1)
    expected_prediction = predict(estimator, X_train)
    # Save the fitted model.
    model_filename = joinpath(@__DIR__, "fixture.model")
    savemodel(estimator, model_filename)

    # Act
    estimator_from_file = LGBMClassification(verbosity = -1)
    loadmodel!(estimator_from_file, model_filename)
    actual_prediction = predict(estimator_from_file, X_train)

    # Assert
    @test expected_prediction == actual_prediction

    # Teardown
    rm(model_filename)
end

@testset "loadmodel predicts same as original model with custom pararms -- regression" begin
    # Arrange
    estimator = estimator = LGBMRegression(
        objective = "regression",
        num_iterations = 100,
        learning_rate = .2,
        early_stopping_round = 3,
        feature_fraction = .5,
        bagging_fraction = .6,
        bagging_freq = 2,
        num_leaves = 100,
        metric = ["auc", "binary_logloss"],
        verbosity = -1
    )
    LightGBM.fit!(estimator, X_train, y_train_regression; verbosity=-1)
    expected_prediction = predict(estimator, X_train)
    # Save the fitted model.
    model_filename = joinpath(@__DIR__, "fixture.model")
    savemodel(estimator, model_filename)
  
    # Act
    estimator_from_file = LGBMRegression(verbosity = -1)
    loadmodel!(estimator_from_file, model_filename)
    actual_prediction = predict(estimator_from_file, X_train)

    # Assert
    @test expected_prediction == actual_prediction

    # Teardown
    rm(model_filename)

end


@testset "loadmodel predicts same as original model with custom params -- binary" begin
    # Arrange
    estimator = estimator = LGBMClassification(
        objective = "binary",
        num_iterations = 100,
        learning_rate = .2,
        early_stopping_round = 3,
        feature_fraction = .5,
        bagging_fraction = .6,
        bagging_freq = 2,
        num_leaves = 100,
        num_class = 1,
        metric = ["auc", "binary_logloss"],
        verbosity = -1
    )
    LightGBM.fit!(estimator, X_train, y_train_binary; verbosity=-1)
    expected_prediction = predict(estimator, X_train)
    # Save the fitted model.
    model_filename = joinpath(@__DIR__, "fixture.model")
    savemodel(estimator, model_filename)

    # Act
    estimator_from_file = LGBMClassification(verbosity = -1)
    loadmodel!(estimator_from_file, model_filename)
    actual_prediction = predict(estimator_from_file, X_train)

    # Assert
    @test expected_prediction == actual_prediction

    # Teardown
    rm(model_filename)
end

end # Module
