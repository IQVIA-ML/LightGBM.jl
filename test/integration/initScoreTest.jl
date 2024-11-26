# Tests that init_score works for multiclass
# reference URL
# https://github.com/Microsoft/LightGBM/issues/1778
# https://stackoverflow.com/questions/57275029/using-the-score-from-first-lightgbm-as-init-score-to-second-lightgbm-gives-diffe

module TestInitScore

using Test
using DelimitedFiles
using LightGBM

@testset "initScoreTest.jl" begin
    try

        LGBM_PATH = ENV["LIGHTGBM_EXAMPLES_PATH"]
        LGBM_PATH = if isabspath(LGBM_PATH) LGBM_PATH else abspath(joinpath(pwd(), "..", LGBM_PATH)) end

        # Test regression estimator.
        regression_test = readdlm(joinpath(LGBM_PATH, "examples", "regression", "regression.test"), '\t')
        regression_train = readdlm(joinpath(LGBM_PATH, "examples", "regression", "regression.train"), '\t')
        X_train = regression_train[:, 2:end]
        y_train = regression_train[:, 1]
        X_test = regression_test[:, 2:end]
        y_test = regression_test[:, 1]

        regression_test_init = readdlm(joinpath(LGBM_PATH, "examples", "regression", "regression.test.init"), '\t')[:,1]
        regression_train_init = readdlm(joinpath(LGBM_PATH, "examples", "regression", "regression.train.init"), '\t')[:,1]

        estimator = LightGBM.LGBMRegression(
            num_iterations = 100,
            learning_rate = .05,
            feature_fraction = .9,
            bagging_fraction = .8,
            bagging_freq = 5,
            num_leaves = 31,
            metric = ["l2"],
            metric_freq = 1,
            is_provide_training_metric = true,
            max_bin = 255,
            min_sum_hessian_in_leaf = 5.,
            min_data_in_leaf = 100,
            max_depth = -1,
            verbosity = -1,
        )

        LightGBM.fit!(estimator, X_train, y_train, verbosity = -1, init_score=regression_train_init);
        LightGBM.fit!(estimator, X_train, y_train, (X_test, y_test), verbosity = -1, init_score=regression_train_init);
        @test true
    catch
        @test false
    end
end


end # module
