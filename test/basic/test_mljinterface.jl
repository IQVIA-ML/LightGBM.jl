module TestUtils


using MLJBase
using Test

import LightGBM


@testset "mlj_to_kwargs removes non-LightGBM parameters from classifier" begin
    # Arrange
    fixture = LightGBM.MLJInterface.LGBMClassifier(verbosity = -1)

    # Act
    output = LightGBM.MLJInterface.mlj_to_kwargs(fixture, [0,1])

    # Assert - both truncate_booster and feature_importance should be removed
    @test :truncate_booster ∉ keys(output)
    @test :feature_importance ∉ keys(output)
end

@testset "mlj_to_kwargs removes non-LightGBM parameters from regressor" begin
    # Arrange
    fixture = LightGBM.MLJInterface.LGBMRegressor(verbosity = -1)

    # Act
    output = LightGBM.MLJInterface.mlj_to_kwargs(fixture)

    # Assert - both truncate_booster and feature_importance should be removed
    @test :truncate_booster ∉ keys(output)
    @test :feature_importance ∉ keys(output)
end

@testset "mlj_to_kwargs adds classifier num_class" begin
    # Arrange
    fixture = LightGBM.MLJInterface.LGBMClassifier(verbosity = -1)

    # Act
    output = LightGBM.MLJInterface.mlj_to_kwargs(fixture, [0,1])

    # Assert
    @test :num_class in keys(output)
end


@testset "reports_feature_importances trait returns true for both models" begin
    @test MLJModelInterface.reports_feature_importances(LightGBM.MLJInterface.LGBMClassifier) == true
    @test MLJModelInterface.reports_feature_importances(LightGBM.MLJInterface.LGBMRegressor) == true
end

@testset "feature_importance hyperparameter validation" begin
    # Valid values should work
    model_gain = LightGBM.MLJInterface.LGBMClassifier(feature_importance=:gain)
    @test model_gain.feature_importance == :gain
    
    model_split = LightGBM.MLJInterface.LGBMRegressor(feature_importance=:split)
    @test model_split.feature_importance == :split
    
    # Invalid values should raise error
    @test_throws ArgumentError LightGBM.MLJInterface.LGBMClassifier(feature_importance=:invalid)
    @test_throws ArgumentError LightGBM.MLJInterface.LGBMRegressor(feature_importance=:invalid)
end


end # Module
