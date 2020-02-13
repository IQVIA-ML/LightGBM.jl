module TestMulticlassLGBM


using MLJBase
using Test
using Random: seed!

import LightGBM

## CLASSIFIER -- shamelessly adapted from binary_classification.jl

# test multiclass case:
N = 5
Nsamples = 3000
seed!(0)

model = LightGBM.mlj_interface.LGBMClassifier(num_iterations=100, num_class=N)

X       = (x1=rand(Nsamples), x2=rand(Nsamples), x3=rand(Nsamples))
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

@test misclassification_rate < 0.015

# All we can really say about fitting with/without weights for this example is that the solutions shouldn't be identical
@test yhat_with_weights != yhat

# It's what it should be for now
@test cache==nothing

@test isa(report, Tuple)

expected_return_type = Tuple{
    LightGBM.LGBMMulticlass,
    MLJBase.CategoricalArrays.CategoricalArray,
}

@test isa(fitresult, expected_return_type)

# here we test the bit where we try to fit a single class problem -- we expect this to throw
model = LightGBM.mlj_interface.LGBMClassifier(num_iterations=100, num_class=1)
@test_throws ErrorException MLJBase.fit(model, 0, X, y)

end # module
true
