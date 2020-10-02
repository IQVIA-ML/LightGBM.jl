module TestRegressionLGBM


using MLJBase
using Test
using Random: seed!

import LightGBM

## Regression -- shamelessly adapted from the other tests

Nsamples = 3000
seed!(0)
calc_rmse(p, t) = sqrt(sum((p - t) .^ 2) / length(p))

model = LightGBM.MLJInterface.LGBMRegressor(num_iterations=100)

X       = rand(Nsamples, 5)
y       = sqrt.(sum(X .^ 2, dims=2)) # make the targets the L2 norm of the vectors
weights = rand(Nsamples)


# fit once, without weights
train, test              = MLJBase.partition(MLJBase.eachindex(y), 0.6)
fitresult, cache, report = MLJBase.fit(model, 0, MLJBase.selectrows(X, train), y[train];)
yhat                     = MLJBase.mode.(MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test)))

# fit again with weights
fitresult, cache, report = MLJBase.fit(model, 0, MLJBase.selectrows(X, train), y[train], weights[train])
yhat_with_weights        = MLJBase.mode.(MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test)))

rmse                     = calc_rmse(y[test], yhat)
rmse_weights             = calc_rmse(y[test], yhat_with_weights)

@test rmse < 0.05
@test !isapprox(rmse, rmse_weights, rtol=0.1) # check that they differ by at least around 10% with/without weights

@test yhat_with_weights != yhat

# Cache contains iterations counts history
@test cache isa NamedTuple
@test cache.num_boostings_done == [100]

@test isa(report, NamedTuple)

expected_return_type = Tuple{
    LightGBM.LGBMRegression,
    Vector{Any}, # blep
    LightGBM.MLJInterface.LGBMRegressor,
}

@test isa(fitresult, expected_return_type)


# Provided by Anthony Blaom as a simple integration test
X, y = @load_boston;
model = LightGBM.MLJInterface.LGBMRegressor()
yhat = fit!(machine(model, X, y); verbosity=0) |> predict;
scitype(yhat) == AbstractVector{Continuous}
@test rms(yhat, y) < 6  # or do proper out-of-sample test


end # module
true
