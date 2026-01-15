module TestBinaryLGBM


using MLJBase
using StatisticalMeasures
using Test

import CategoricalArrays
import LightGBM

## CLASSIFIER -- shamelessly copied from MLJModels/test/XGBoost.jl

model = LightGBM.MLJInterface.LGBMClassifier(objective="binary", num_iterations=100, verbosity = -1)

# test binary case:
N = 2
Nsamples = 3000

X       = (x1=rand(Nsamples), x2=rand(Nsamples), x3=rand(Nsamples))
ycat    = string.(mod.(round.(Int, X.x1 * 10), N)) |> MLJBase.categorical
weights = rand(Nsamples) .^ 2

y = MLJBase.identity.(ycat) # make plain Vector with categ. elements (actually not sure what this is doing)

# fit once, without weights
train, test              = MLJBase.partition(MLJBase.eachindex(y), 0.6)
fitresult, cache, report = MLJBase.fit(model, 0, MLJBase.selectrows(X, train), y[train];)
yhat                     = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))
yhatprob                 = MLJBase.pdf(yhat, MLJBase.levels(y))
yhatpred                 = MLJBase.mode.(yhat)

# fit again with weights
fitresult, cache, report = MLJBase.fit(model, 0, MLJBase.selectrows(X, train), y[train], weights[train])
yhat_with_weights        = MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test))
yhat_with_weights_prob   = MLJBase.pdf(yhat_with_weights, MLJBase.levels(y))

misclassification_rate   = sum(yhatpred .!= y[test])/length(test)

@test misclassification_rate < 0.05

# All we can really say about fitting with/without weights for this example is that the solutions shouldn't be identical
@test yhat_with_weights_prob != yhatprob

# Cache contains iterations counts history
@test cache isa NamedTuple
@test cache.num_boostings_done == [100]

@test isa(report, NamedTuple)

expected_return_type = Tuple{
    LightGBM.LGBMClassification,
    CategoricalArrays.CategoricalArray,
    LightGBM.MLJInterface.LGBMClassifier,
    Union{Nothing, Tuple},
}

@test isa(fitresult, expected_return_type)

# here we test the bit where we try to fit a nonbinary using a binary estimator -- we expect this to throw
crazyN = 5 # testing nonbinary
ycrazy = string.(mod.(round.(Int, X.x1 * 10), crazyN)) |> MLJBase.categorical
@test_throws ArgumentError MLJBase.fit(model, 0, X, ycrazy)

end # module
true
