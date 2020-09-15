module TestBinaryLGBM


using MLJBase
using Test
using Random: seed!

import CategoricalArrays
import LightGBM

## CLASSIFIER -- shamelessly copied from MLJModels/test/XGBoost.jl

model = LightGBM.MLJInterface.LGBMClassifier(objective="binary", num_iterations=100)

# test binary case:
N = 2
seed!(0)

X       = (x1=rand(1000), x2=rand(1000), x3=rand(1000))
ycat    = string.(mod.(round.(Int, X.x1 * 10), N)) |> MLJBase.categorical
weights = Float64.(MLJBase.int(ycat)) # just use the 1's/2's directly as multipliers

y = MLJBase.identity.(ycat) # make plain Vector with categ. elements (actually not sure what this is doing)

# fit once, without weights
train, test              = MLJBase.partition(MLJBase.eachindex(y), 0.6)
fitresult, cache, report = MLJBase.fit(model, 0, MLJBase.selectrows(X, train), y[train];)
yhat                     = MLJBase.mode.(MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test)))

# fit again with weights
fitresult, cache, report = MLJBase.fit(model, 0, MLJBase.selectrows(X, train), y[train], weights[train])
yhat_with_weights        = MLJBase.mode.(MLJBase.predict(model, fitresult, MLJBase.selectrows(X, test)))

misclassification_rate   = sum(yhat .!= y[test])/length(test)

# Well, although XGBoost gets misclassification below 0.01, LightGBM can't do it with the default settings ...
# It gets to exactly 0.01...
@test misclassification_rate < 0.015

# All we can really say about fitting with/without weights for this example is that the solutions shouldn't be identical
@test yhat_with_weights != yhat

# Cache contains iterations counts history
@test cache isa NamedTuple
@test cache.num_boostings_done == [100]

@test isa(report, NamedTuple)

expected_return_type = Tuple{
    LightGBM.LGBMClassification,
    CategoricalArrays.CategoricalArray,
    LightGBM.MLJInterface.LGBMClassifier,
}

@test isa(fitresult, expected_return_type)

# here we test the bit where we try to fit a nonbinary using a binary estimator -- we expect this to throw
crazyN = 5 # testing nonbinary
ycrazy = string.(mod.(round.(Int, X.x1 * 10), crazyN)) |> MLJBase.categorical
@test_throws ArgumentError MLJBase.fit(model, 0, X, ycrazy)

end # module
true
