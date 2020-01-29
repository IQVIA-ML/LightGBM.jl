module TestBasic

using Test
using DelimitedFiles
using LightGBM

LGBM_PATH = ENV["LIGHTGBM_EXAMPLES_PATH"]
LGBM_PATH = if isabspath(LGBM_PATH) LGBM_PATH else abspath(joinpath(pwd(), "..", LGBM_PATH)) end

@testset "basic_tests.jl  -- binary" begin

    # Use binary example for generic tests.
    binary_test = readdlm(joinpath(LGBM_PATH, "examples", "binary_classification", "binary.test"), '\t')
    binary_train = readdlm(joinpath(LGBM_PATH, "examples", "binary_classification", "binary.train"), '\t')
    X_train = binary_train[:, 2:end]
    y_train = binary_train[:, 1]
    X_test = binary_test[:, 2:end]
    y_test = binary_test[:, 1]

    # Test wrapper functions.
    train_ds = LightGBM.LGBM_DatasetCreateFromMat(X_train, "objective=binary")
    @test LightGBM.LGBM_DatasetGetNumData(train_ds) == 7000
    @test LightGBM.LGBM_DatasetGetNumFeature(train_ds) == 28
    @test LightGBM.LGBM_DatasetSetField(train_ds, "label", y_train) == nothing
    @test LightGBM.LGBM_DatasetGetField(train_ds, "label") == y_train
    bst = LightGBM.LGBM_BoosterCreate(train_ds, "lambda_l1=10. metric=auc, verbosity=-1")

    test_ds = LightGBM.LGBM_DatasetCreateFromMat(X_test, "objective=binary", train_ds);
    @test LightGBM.LGBM_DatasetSetField(test_ds, "label", y_test) == nothing
    @test LightGBM.LGBM_BoosterAddValidData(bst, test_ds) == nothing
    @test LightGBM.LGBM_BoosterUpdateOneIter(bst) == 0
    @test LightGBM.LGBM_BoosterGetEvalCounts(bst) == 1
    @test LightGBM.LGBM_BoosterGetEvalNames(bst)[1] == "auc"

    # Test binary estimator.
    estimator = LightGBM.LGBMBinary(
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
    LightGBM.fit(estimator, X_train, y_train, verbosity = -1);
    LightGBM.fit(estimator, X_train, y_train, (X_test, y_test), verbosity = -1)

    # Test setting feature names
    jl_feature_names = ["testname_$i" for i in 1:28]
    LightGBM.LGBM_DatasetSetFeatureNames(estimator.booster.datasets[1], jl_feature_names)
    lgbm_feature_names = LightGBM.LGBM_DatasetGetFeatureNames(estimator.booster.datasets[1])
    @test jl_feature_names == lgbm_feature_names

    # Test prediction, and loading and saving models.
    test_filename = tempname()
        LightGBM.savemodel(estimator, test_filename);
    try
        pre = LightGBM.predict(estimator, X_train, verbosity = -1)
        LightGBM.loadmodel(estimator, test_filename);
        post = LightGBM.predict(estimator, X_train, verbosity = -1)
        @test isapprox(pre, post)
    finally
        rm(test_filename)
    end

    # Test cross-validation.
    splits = (collect(1:3500), collect(3501:7000))
    LightGBM.cv(estimator, X_train, y_train, splits; verbosity = -1)

    # Test exhaustive search.
    params = [Dict(:num_iterations => num_iterations,
                   :num_leaves => num_leaves) for
                   num_iterations in (1, 2),
                   num_leaves in (5, 10)]
    LightGBM.search_cv(estimator, X_train, y_train, splits, params; verbosity = -1)

end

@testset "basic_tests.jl -- regression" begin

    # Test regression estimator.
    regression_test = readdlm(joinpath(LGBM_PATH, "examples", "regression", "regression.test"), '\t')
    regression_train = readdlm(joinpath(LGBM_PATH, "examples", "regression", "regression.train"), '\t')
    X_train = regression_train[:, 2:end]
    y_train = regression_train[:, 1]
    X_test = regression_test[:, 2:end]
    y_test = regression_test[:, 1]

    estimator = LightGBM.LGBMRegression(
        num_iterations = 100,
        learning_rate = .05,
        feature_fraction = .9,
        bagging_fraction = .8,
        bagging_freq = 5,
        num_leaves = 31,
        metric = ["l2"],
        metric_freq = 1,
        is_training_metric = true,
        max_bin = 255,
        min_sum_hessian_in_leaf = 5.,
        min_data_in_leaf = 100,
        max_depth = -1,
    )

    scores = LightGBM.fit(estimator, X_train, y_train, (X_test, y_test), verbosity = -1)
    @test scores["test_1"]["l2"][end] < .5

end


@testset "basic_tests.jl -- multiclass" begin

    # Test multiclass estimator.
    multiclass_test = readdlm(joinpath(LGBM_PATH, "examples", "multiclass_classification", "multiclass.test"), '\t')
    multiclass_train = readdlm(joinpath(LGBM_PATH, "examples", "multiclass_classification", "multiclass.train"), '\t')
    X_train = Matrix(multiclass_train[:, 2:end])
    y_train = Array(multiclass_train[:, 1])
    X_test = Matrix(multiclass_test[:, 2:end])
    y_test = Array(multiclass_test[:, 1])

    estimator = LightGBM.LGBMMulticlass(
        num_iterations = 100,
        learning_rate = .05,
        feature_fraction = .9,
        bagging_fraction = .8,
        bagging_freq = 5,
        num_leaves = 31,
        metric = ["multi_logloss"],
        metric_freq = 1,
        is_training_metric = true,
        max_bin = 255,
        min_sum_hessian_in_leaf = 5.,
        min_data_in_leaf = 100,
        num_class = 5,
        early_stopping_round = 10,
    )

    scores = LightGBM.fit(estimator, X_train, y_train, (X_test, y_test), verbosity = -1)
    @test scores["test_1"]["multi_logloss"][end] < 1.4

    # Test row major multiclass
    X_train = Matrix(multiclass_train[:, 2:end]')
    X_test = Matrix(multiclass_test[:, 2:end]')

    estimator = LightGBM.LGBMMulticlass(
        num_iterations = 100,
        learning_rate = .05,
        feature_fraction = .9,
        bagging_fraction = .8,
        bagging_freq = 5,
        num_leaves = 31,
        metric = ["multi_logloss"],
        metric_freq = 1,
        is_training_metric = true,
        max_bin = 255,
        min_sum_hessian_in_leaf = 5.,
        min_data_in_leaf = 100,
        num_class = 5,
        early_stopping_round = 10,
    )

    scores = LightGBM.fit(estimator, X_train, y_train, (X_test, y_test), verbosity = -1,
                          is_row_major = true)
    @test scores["test_1"]["multi_logloss"][end] < 1.4

end

end # module
