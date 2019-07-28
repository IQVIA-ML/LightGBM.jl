# Tests that the weighting scheme works for multiclass to balance skewed data

# Get margin for each sample in votes, according to its true label in labels
function margin( votes::Array{T1,2}, labels::Vector{T2} ) where {T1 <: Real, T2 <: Real}
    numClasses = size(votes,1)
    numSamples = size(votes,2)

    votes = copy(votes)

    mg = zeros(eltype(votes), numSamples)
    for i=1:numSamples
        local v = votes[Int(labels[i]) + 1, i]  # copy score of true label

        # compute margin
        votes[Int(labels[i]) + 1, i] = -Inf
        mg[i] = v - maximum(votes[:,i])
    end

    return mg
end

# Create a skewed 3-class problem
N1 = 100
N2 = 100000
N3 = 1000000
N = N1 + N2 + N3

numClasses = 3
numFeats = 5

X = rand((N, numFeats))
y = vcat(zeros(N1), ones(N2), 2*ones(N3))

# one feature is exactly the label
X[:,1] = y

estimator = LightGBM.LGBMMulticlass(num_iterations = 10,
                                    learning_rate = .5,
                                    feature_fraction = 0.5,
                                    bagging_fraction = 1.0,
                                    bagging_freq = 1,
                                    num_leaves = 5,
                                    metric = ["multi_logloss"],
                                    num_class = numClasses,
                                    min_data_in_leaf=1,
                                    min_sum_hessian_in_leaf=1);

# Compute required weights
classWeights = [N / (numClasses * sum(y .== n)) for n in 0:(numClasses - 1)]
weights = [classWeights[Int(n+1)] for n in y]

LightGBM.fit(estimator, X, y; weights=weights);

pred = LightGBM.predict(estimator, X; predict_type=1)
pred = reshape(pred, ( numClasses, size(X,1)))

# Average margins for each prediction should be similar if weighting works
margins = margin(pred, y)
margins = [mean(margins[y .== label]) for label in 0:(numClasses-1)]

meanMargin = mean(margins)
marginDeviations = abs.(margins - meanMargin)
maxMarginDeviation = maximum(marginDeviations)

@test maxMarginDeviation / meanMargin < 0.1
