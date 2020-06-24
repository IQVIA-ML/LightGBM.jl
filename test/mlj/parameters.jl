module TestMLJParameters

using MLJBase
using Test

import LightGBM


parameters = Dict( 
    :boosting => "goss",
    :drop_rate => 0.6,
    :max_drop => 100,
    :skip_drop => 0.9,
    :xgboost_dart_mode => true,
    :uniform_drop => true,
    :drop_seed => 10,
    :top_rate => 0.7,
    :other_rate => 0.2,
)

# The tests for regressor and classifier is to check if
# the parameters are present or not after we fit them

@testset "MLJ parameters -- regressor" begin

    X, y = @MLJBase.load_boston;
    
    # Passed the dictionary of parameters to the regressor model 
    model = LightGBM.MLJInterface.LGBMRegressor(;parameters...)

    # Extract the estimator
    fit_result, _, _ = MLJBase.fit(model, 0, X, y)
    estimator, _, _ = fit_result

    for (k, v) in parameters
        @test getproperty(estimator, k) == v
    end
    
end


@testset "MLJ parameters -- classifier" begin

    X, y = @MLJBase.load_iris;
    model = LightGBM.MLJInterface.LGBMClassifier(;parameters...)
    fit_result, _, _ = MLJBase.fit(model, 0, X, y)
    estimator, _, _ = fit_result

    for (k, v) in parameters
        @test getproperty(estimator, k) == v
    end
    
end

end #end module