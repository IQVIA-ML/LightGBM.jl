module TestMulticlassLGBM


using MLJBase
using Test

import CategoricalArrays
import LightGBM

## CLASSIFIER -- shamelessly adapted from binary_classification.jl

# test multiclass case:
N = 5
Nsamples = 3000

model = LightGBM.MLJInterface.LGBMClassifier(num_iterations=100)

X       = (x1=rand(Nsamples), x2=rand(Nsamples), x3=rand(Nsamples))
ycat    = string.(mod.(round.(Int, X.x1 * 10), N)) |> MLJBase.categorical
weights = Float64.(MLJBase.int(ycat)) # just use the 1's/2's directly as multipliers

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
yhat_with_weights_pred   = MLJBase.mode.(yhat_with_weights)

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
}

@test isa(fitresult, expected_return_type)

# Provided by Anthony Blaom as a simple integration test
X, y = @load_iris;
model = LightGBM.MLJInterface.LGBMClassifier()
yhat = fit!(machine(model, X, y); verbosity=0) |> predict;
@test scitype(mode.(yhat)) <: AbstractVector{Multiclass{3}}
@test mean(cross_entropy(yhat, y)) < 0.6  # or do proper out-of-sample test


end # module
true
