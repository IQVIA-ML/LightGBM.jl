module TestResamplingAPI

# Test the MLJ resampling API: reformat → selectrows → fit(LGBMFrontEndData, y, w) → predict.
# This path is used by MLJ when you evaluate(mach, resampling=CV(...)) or similar.

using MLJBase
using StatisticalMeasures
using Test

import LightGBM

# reformat/selectrows are in MLJModelInterface (dependency of MLJBase); import for dispatch
using MLJModelInterface: reformat, selectrows

@testset "reformat" begin
    model = LightGBM.MLJInterface.LGBMRegressor(num_iterations=50, verbosity=-1)
    X = rand(200, 4)
    y = vec(sqrt.(sum(X .^ 2, dims=2)))
    w = rand(200)

    data_only = reformat(model, X)
    @test data_only isa Tuple
    @test length(data_only) == 1
    @test data_only[1] isa LightGBM.MLJInterface.LGBMFrontEndData

    data_with_y = reformat(model, X, y)
    @test length(data_with_y) == 2
    # Same X => same logical front-end data (matrix, feature_names, params). Dataset handle may differ.
    a, b = data_only[1], data_with_y[1]
    @test isequal(a.matrix, b.matrix) && a.feature_names == b.feature_names && a.dataset_params == b.dataset_params
    @test data_with_y[2] === y

    data_with_y_w = reformat(model, X, y, w)
    @test length(data_with_y_w) == 3
    a, b = data_only[1], data_with_y_w[1]
    @test isequal(a.matrix, b.matrix) && a.feature_names == b.feature_names && a.dataset_params == b.dataset_params
    @test data_with_y_w[2] === y
    @test data_with_y_w[3] === w
end

@testset "selectrows and fit(LGBMFrontEndData) regression" begin
    model = LightGBM.MLJInterface.LGBMRegressor(num_iterations=50, verbosity=-1)
    n = 400
    X = rand(n, 5)
    y = vec(sqrt.(sum(X .^ 2, dims=2)))
    train_idx, test_idx = MLJBase.partition(MLJBase.eachindex(y), 0.7)

    # Reformat once (as MLJ would)
    (data,) = reformat(model, X)
    data_train = selectrows(data, train_idx)
    data_test  = selectrows(data, test_idx)

    # Fit using the new path: fit(model, verbosity, data::LGBMFrontEndData, y, w)
    fitresult, cache, report = MLJBase.fit(model, 0, data_train, y[train_idx])
    @test fitresult isa Tuple
    @test cache.num_boostings_done == [50]

    # Predict with LGBMFrontEndData (data_test)
    yhat = MLJBase.predict(model, fitresult, data_test)
    @test length(yhat) == length(test_idx)
    @test eltype(yhat) <: Real
    rmse_val = rms(yhat, y[test_idx])
    @test rmse_val < 0.2  # sanity check
end

@testset "selectrows and fit(LGBMFrontEndData) with weights regression" begin
    model = LightGBM.MLJInterface.LGBMRegressor(num_iterations=50, verbosity=-1)
    n = 400
    X = rand(n, 5)
    y = vec(sqrt.(sum(X .^ 2, dims=2)))
    w = rand(n) .^ 2
    train_idx, test_idx = MLJBase.partition(MLJBase.eachindex(y), 0.7)

    (data,) = reformat(model, X)
    data_train = selectrows(data, train_idx)
    data_test  = selectrows(data, test_idx)

    fitresult, cache, report = MLJBase.fit(model, 0, data_train, y[train_idx], w[train_idx])
    @test fitresult isa Tuple
    yhat = MLJBase.predict(model, fitresult, data_test)
    @test length(yhat) == length(test_idx)
end

@testset "evaluate with CV (full resampling stack)" begin
    model = LightGBM.MLJInterface.LGBMRegressor(num_iterations=30, verbosity=-1)
    X = rand(300, 4)
    y = vec(sqrt.(sum(X .^ 2, dims=2)))
    mach = machine(model, X, y; scitype_check_level=0)
    # 2-fold CV: uses reformat → selectrows → fit(data, y) under the hood
    result = evaluate!(mach; resampling=CV(nfolds=2), measure=rms)
    @test result isa MLJBase.PerformanceEvaluation
    @test length(result.per_fold) >= 1
    # per_fold and measurement can be scalar or vector depending on MLJBase version
    all_finite(x) = all(isfinite, x isa AbstractVector ? x : [x])
    @test all_finite(result.measurement)
end

end
true
