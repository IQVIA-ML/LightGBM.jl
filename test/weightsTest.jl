# Tests that the weighting scheme works for binary_classification

module TestWeights

using Test
using DelimitedFiles
using LightGBM

@testset "weightsTest.jl" begin
    try

        LGBM_PATH = ENV["LIGHTGBM_EXAMPLES_PATH"]
        LGBM_PATH = if isabspath(LGBM_PATH) LGBM_PATH else abspath(joinpath(pwd(), "..", LGBM_PATH)) end

        binary_test = readdlm(joinpath(LGBM_PATH, "examples", "binary_classification", "binary.test"), '\t')
        binary_train = readdlm(joinpath(LGBM_PATH, "examples", "binary_classification", "binary.train"), '\t')
        X_train = binary_train[:, 2:end]
        y_train = binary_train[:, 1]
        X_test = binary_test[:, 2:end]
        y_test = binary_test[:, 1]

        binary_test_weight = readdlm(joinpath(LGBM_PATH, "examples", "binary_classification", "binary.test.weight"), '\t')[:,1]
        binary_train_weight = readdlm(joinpath(LGBM_PATH, "examples", "binary_classification", "binary.train.weight"), '\t')[:,1]

        # Test binary estimator.
        estimator = LightGBM.LGBMClassification(
            objective = "binary",
            num_class = 1,
            num_iterations = 20,
            learning_rate = .1,
            early_stopping_round = 1,
            feature_fraction = .8,
            bagging_fraction = .9,
            bagging_freq = 1,
            num_leaves = 1000,
            metric = ["auc", "binary_logloss"],
            is_training_metric = true,
            max_bin = 255,
            min_sum_hessian_in_leaf = 0.,
            min_data_in_leaf = 1,
        )

        # Test fitting.
        LightGBM.fit!(estimator, X_train, y_train, verbosity = -1, weights=binary_train_weight[:,1])
        LightGBM.fit!(estimator, X_train, y_train, (X_test, y_test), verbosity = -1, weights=binary_train_weight[:,1])

        @test true
    catch
        @test false
    end
end   


end # module
