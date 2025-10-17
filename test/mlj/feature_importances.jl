module TestFeatureImportances

using MLJBase
using Test
using CategoricalArrays

import LightGBM

# Helper function to validate feature importances structure
function validate_importances(importances, expected_length)
    @test importances isa Vector{<:Pair}
    @test length(importances) == expected_length
    @test all(pair -> pair.first isa Symbol && pair.second isa Real, importances)
    @test all(pair -> pair.second >= 0.0, importances)
end

@testset "feature_importances function tests" begin
    # Setup test data
    Nsamples = 1000
    X = rand(Nsamples, 5)
    y = sqrt.(sum(X .^ 2, dims=2)) # make the targets the L2 norm of the vectors
    
    # Split data
    train, test = MLJBase.partition(MLJBase.eachindex(y), 0.7)
    X_train = MLJBase.selectrows(X, train)
    y_train = y[train]
    X_test = MLJBase.selectrows(X, test)
    y_test = y[test]
    
    @testset "LGBMRegressor feature importances" begin
        # Test with gain importance (default)
        model_gain = LightGBM.MLJInterface.LGBMRegressor(
            num_iterations=50, 
            verbosity=-1, 
            feature_importance=:gain
        )
        
        fitresult, cache, report = MLJBase.fit(model_gain, 0, X_train, y_train)
        importances_gain = feature_importances(model_gain, fitresult, report)
        
        validate_importances(importances_gain, 5)
        
        # Test with split importance
        model_split = LightGBM.MLJInterface.LGBMRegressor(
            num_iterations=50, 
            verbosity=-1, 
            feature_importance=:split
        )
        
        fitresult_split, _, report_split = MLJBase.fit(model_split, 0, X_train, y_train)
        importances_split = feature_importances(model_split, fitresult_split, report_split)
        
        validate_importances(importances_split, 5)
        
        # Gain and split importances should be different
        @test importances_gain != importances_split
        
        # Feature names should be the same
        gain_features = [pair.first for pair in importances_gain]
        split_features = [pair.first for pair in importances_split]
        @test gain_features == split_features
    end
    
    @testset "LGBMClassifier feature importances" begin
        # Create classification data with categorical arrays
        y_class = categorical([rand() > 0.5 ? "A" : "B" for _ in 1:Nsamples])
        y_class_train = y_class[train]
        
        # Test with gain importance (default)
        model_gain = LightGBM.MLJInterface.LGBMClassifier(
            num_iterations=50, 
            verbosity=-1, 
            feature_importance=:gain
        )
        
        fitresult, cache, report = MLJBase.fit(model_gain, 0, X_train, y_class_train)
        importances_gain = feature_importances(model_gain, fitresult, report)
        
        validate_importances(importances_gain, 5)
        
        # Test with split importance
        model_split = LightGBM.MLJInterface.LGBMClassifier(
            num_iterations=50, 
            verbosity=-1, 
            feature_importance=:split
        )
        
        fitresult_split, _, report_split = MLJBase.fit(model_split, 0, X_train, y_class_train)
        importances_split = feature_importances(model_split, fitresult_split, report_split)
        
        validate_importances(importances_split, 5)
        
        # Gain and split importances should be different
        @test importances_gain != importances_split
    end
    
    @testset "feature_importances error handling" begin
        # Test invalid feature_importance value
        model = LightGBM.MLJInterface.LGBMRegressor(num_iterations=10, verbosity=-1)
        # Manually set invalid importance type to test error handling
        model.feature_importance = :invalid
        
        fitresult, cache, report = MLJBase.fit(model, 0, X_train, y_train)
        
        @test_throws ErrorException feature_importances(model, fitresult, report)
    end
    
    @testset "get_feature_names with default names" begin
        # Test that default feature names are returned when not explicitly set
        model = LightGBM.MLJInterface.LGBMRegressor(num_iterations=10, verbosity=-1)
        fitresult, cache, report = MLJBase.fit(model, 0, X_train, y_train)
        
        fitted_model, _, _ = fitresult
        feature_names = LightGBM.MLJInterface.get_feature_names(fitted_model)
        
        # LightGBM provides default names like "Column_0", "Column_1", etc.
        @test length(feature_names) == 5
        @test all(name -> name isa String, feature_names)
        # When not explicitly set, LightGBM uses "Column_{i}" format
        expected_names = ["Column_$i" for i in 0:4]
        @test feature_names == expected_names
        
        # Also test via feature_importances
        importances = feature_importances(model, fitresult, report)
        feature_names_from_importances = [pair.first for pair in importances]
        expected_symbols = [Symbol("Column_$i") for i in 0:4]
        @test feature_names_from_importances == expected_symbols
    end
    
    @testset "get_feature_names with custom names" begin
        # Test that custom feature names are preserved through the booster
        X_test = rand(30, 4)
        
        dataset_custom = LightGBM.LGBM_DatasetCreateFromMat(X_test, "")
        custom_names = ["alpha", "beta", "gamma", "delta"]
        LightGBM.LGBM_DatasetSetFeatureNames(dataset_custom, custom_names)
        booster_custom = LightGBM.LGBM_BoosterCreate(dataset_custom, "")
        
        # Test via FFI directly
        names_custom = LightGBM.LGBM_BoosterGetFeatureNames(booster_custom)
        @test !isnothing(names_custom)
        @test !isempty(names_custom)
        @test names_custom == custom_names
        @test length(names_custom) == 4
        @test all(name -> name isa String, names_custom)
        
        # Test via get_feature_names helper
        estimator_custom = LightGBM.LGBMRegression()
        estimator_custom.booster = booster_custom
        @test LightGBM.MLJInterface.get_feature_names(estimator_custom) == custom_names
    end
    
    @testset "get_feature_names error on uninitialized booster" begin
        # Test that calling get_feature_names on an unfitted model throws an error
        estimator = LightGBM.LGBMRegression()
        
        # Verify booster is uninitialized
        @test estimator.booster.handle == C_NULL
        
        # Should throw an ErrorException with informative message
        @test_throws ErrorException LightGBM.MLJInterface.get_feature_names(estimator)
    end
end

end # module
true

