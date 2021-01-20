using Test
using LightGBM

X_train = randn(1000, 20)
y_train_binary = rand(0:1, 1000)
y_train_regression = rand(1000)

@testset "stringifyparams -- convert to zero-based" begin
    indices = [1, 3, 5, 7, 9]
    classifier = LightGBM.LGBMClassification(categorical_feature = indices)
    ds_parameters = LightGBM.stringifyparams(classifier, LightGBM.DATASETPARAMS)

    expected = "categorical_feature=0,2,4,6,8"
    @test occursin(expected, ds_parameters)
end



@testset "loadmodel predicts same as original model -- regression" begin
    # Arrange
    estimator = LightGBM.LGBMRegression(objective = "regression")
    LightGBM.fit!(estimator, X_train, y_train_regression)
    expected_prediction = predict(estimator, X_train)
    # Save the fitted model.
    model_filename = joinpath(@__DIR__, "fixture.model")
    savemodel(estimator, model_filename)
  
    # Act
    estimator_from_file = LGBMRegression()
    loadmodel!(estimator_from_file, model_filename)
    actual_prediction = predict(estimator_from_file, X_train)

    # Assert
    @test expected_prediction == actual_prediction

    # Teardown
    rm(model_filename)

end


@testset "loadmodel predicts same as original model -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(objective = "binary", num_class = 1)
    LightGBM.fit!(estimator, X_train, y_train_binary)
    expected_prediction = predict(estimator, X_train)
    # Save the fitted model.
    model_filename = joinpath(@__DIR__, "fixture.model")
    savemodel(estimator, model_filename)

    # Act
    estimator_from_file = LGBMClassification()
    loadmodel!(estimator_from_file, model_filename)
    actual_prediction = predict(estimator_from_file, X_train)

    # Assert
    @test expected_prediction == actual_prediction

    # Teardown
    rm(model_filename)
end
