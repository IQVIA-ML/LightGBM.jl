# Tests that the weighting scheme works for binary_classification

module TestGroupQuery

using Test
using LightGBM


#=
The below code is a direct translation of the following Python code:
import pandas as pd
import lightgbm as lgb
# Create X_train DataFrame
X_train = pd.DataFrame({
    'var1' : [0.3, 0.1, 0.5, 0.3, 0.7, 0.2, 0.1, 0.4],
    'var2' : [0.6, 0.4, 0.8, 0.6, 1.0, 0.5, 0.4, 0.7],
    'var3' : [0.9, 0.7, 1.1, 0.9, 1.3, 0.8, 0.7, 1.0],
})
# Create X_test DataFrame
X_test = pd.DataFrame({
    'var1' : [0.6, 0.2],
    'var2' : [0.9, 0.5],
    'var3' : [1.2, 0.8],
})
# Create y_train and y_test arrays
y_train = [0, 0, 0, 0, 1, 0, 1, 1]
y_test = [0, 1]
# Create group_train and group_test arrays
group_train = [2, 2, 4]
group_test = [1, 1]
# Create a ranker with the specified parameters
ranker = lgb.LGBMRanker(
    objective = 'lambdarank',
    metric = 'ndcg',
    ndcg_eval_at = [1, 3, 5, 10],
    learning_rate = 0.1,
    num_leaves = 31,
    min_data_in_leaf = 1,
    verbose =  -1,
)
# Train the model
ranker.fit(X_train, y_train, group=group_train)
# Predict the relevance scores for the test set
y_pred = ranker.predict(X_test)
# >>> y_pred
# array([ 0.37139225, -0.15637136])
=#
@testset "Group Query information for training data" begin


    # Create X_train Matrix
    X_train = [
        0.3 0.6 0.9;
        0.1 0.4 0.7;
        0.5 0.8 1.1;
        0.3 0.6 0.9;
        0.7 1.0 1.3;
        0.2 0.5 0.8;
        0.1 0.4 0.7;
        0.4 0.7 1.0;
    ]

    # Create X_test Matrix
    X_test = [
        0.6 0.9 1.2;
        0.2 0.5 0.8;
    ]

    # Create y_train and y_test arrays
    y_train = [0, 0, 0, 0, 1, 0, 1, 1]
    y_test = [0, 1]

    # Create group_train and group_test arrays
    group_train = [2, 2, 4]
    group_test = [1, 1]

    # Create ranker model
    ranker = LightGBM.LGBMRanking(
        num_class = 1,
        objective = "lambdarank",
        metric = ["ndcg"],
        eval_at = [1, 3, 5, 10],
        learning_rate = 0.1,
        num_leaves = 31,
        min_data_in_leaf = 1,
    )

    # Fit the model
    LightGBM.fit!(ranker, X_train, Vector(y_train), group = group_train)

    # Predict the relevance scores for the test set
    y_pred = LightGBM.predict(ranker, X_test)

    # Test the predicted scores are as expected
    @test y_pred ≈ [0.3713922520492622; -0.15637136430479565] atol=1e-6

end


end # module
