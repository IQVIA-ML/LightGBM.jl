module TestParameters

using Test
using DelimitedFiles
using LightGBM


NSAMPLES_TRAIN = 2000
NSAMPLES_TEST = 500
NFEATURES = 20

X_train = randn(NSAMPLES_TRAIN, NFEATURES)
y_train_binary = rand(0:1, NSAMPLES_TRAIN)
y_train_regression = rand(NSAMPLES_TRAIN)
X_test = randn(NSAMPLES_TEST, NFEATURES)


norm(x) = sqrt(sum(x .^ 2))

@testset "parameters -- boosting" begin

    # The scheme is to set up an old style gbdt and a new DART and show that the learned
    # models are "diffferent", i.e. the parameter passed through and made a difference
    # Check for both classification and regression

    classifier_gdbt = LightGBM.LGBMClassification(boosting = "gbdt")
    classifier_dart = LightGBM.LGBMClassification(boosting = "dart")
    classifier_goss = LightGBM.LGBMClassification(boosting = "goss")
    regressor_gbdt = LightGBM.LGBMRegression(boosting = "gbdt")
    regressor_dart = LightGBM.LGBMRegression(boosting = "dart")
    regressor_goss = LightGBM.LGBMRegression(boosting = "goss")

    LightGBM.fit!(classifier_dart, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(classifier_gdbt, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(classifier_goss, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(regressor_gbdt, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(regressor_dart, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(regressor_goss, X_train, y_train_regression, verbosity = -1)

    p_gbdt = LightGBM.predict(classifier_gdbt, X_test, verbosity = -1)
    p_dart = LightGBM.predict(classifier_dart, X_test, verbosity = -1)
    p_goss = LightGBM.predict(classifier_goss, X_test, verbosity = -1)
    r_gbdt = LightGBM.predict(regressor_gbdt, X_test, verbosity = -1)
    r_dart = LightGBM.predict(regressor_dart, X_test, verbosity = -1)
    r_goss = LightGBM.predict(regressor_goss, X_test, verbosity = -1)

    # checking the models are sufficiently different by making sure
    # that the distance between the vectors is relatively large
    @test norm(p_dart .- p_gbdt) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test_broken norm(p_goss .- p_gbdt) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(p_dart .- p_goss) >= sqrt(NSAMPLES_TEST) * 1e-5
    # Check also for regression
    @test norm(r_dart .- r_gbdt) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test_broken norm(r_goss .- r_gbdt) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(r_dart .- r_goss) >= sqrt(NSAMPLES_TEST) * 1e-5

end


@testset "parameters -- dart classifcation" begin

    # the scheme is to set up a DART classifier and show that changing each one of
    # the parameters individually results in a difference to the model

    dart_default = LightGBM.LGBMClassification(boosting = "dart")
    dart_drop_rate = LightGBM.LGBMClassification(boosting = "dart", drop_rate = 0.9)
    dart_max_drop = LightGBM.LGBMClassification(boosting = "dart", max_drop = 1)
    dart_skip_drop = LightGBM.LGBMClassification(boosting = "dart", skip_drop = 1.0)
    dart_xdm = LightGBM.LGBMClassification(boosting = "dart", xgboost_dart_mode = true)
    dart_uniform_drop = LightGBM.LGBMClassification(boosting = "dart", uniform_drop = true)
    dart_drop_seed = LightGBM.LGBMClassification(boosting = "dart", drop_seed = 20)

    LightGBM.fit!(dart_default, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_drop_rate, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_max_drop, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_skip_drop, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_xdm, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_uniform_drop, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(dart_drop_seed, X_train, y_train_binary, verbosity = -1)

    p_dart_default = LightGBM.predict(dart_default, X_test, verbosity = -1)
    p_dart_drop_rate = LightGBM.predict(dart_drop_rate, X_test, verbosity = -1)
    p_dart_max_drop = LightGBM.predict(dart_max_drop, X_test, verbosity = -1)
    p_dart_skip_drop = LightGBM.predict(dart_skip_drop, X_test, verbosity = -1)
    p_dart_xdm = LightGBM.predict(dart_xdm, X_test, verbosity = -1)
    p_dart_uniform_drop = LightGBM.predict(dart_uniform_drop, X_test, verbosity = -1)
    p_dart_drop_seed = LightGBM.predict(dart_drop_seed, X_test, verbosity = -1)

    @test norm(p_dart_default .- p_dart_drop_rate) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(p_dart_default .- p_dart_max_drop) >= sqrt(NSAMPLES_TEST) * 1e-10
    @test norm(p_dart_default .- p_dart_skip_drop) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(p_dart_default .- p_dart_xdm) >= sqrt(NSAMPLES_TEST) * 1e-10
    @test_broken norm(p_dart_default .- p_dart_uniform_drop) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(p_dart_default .- p_dart_drop_seed) >= sqrt(NSAMPLES_TEST) * 1e-5

end


@testset "parameters -- dart regression" begin

    # the scheme is to set up a DART regressor and show that changing each one of
    # the parameters individually results in a difference to the model

    dart_default = LightGBM.LGBMRegression(boosting = "dart")
    dart_drop_rate = LightGBM.LGBMRegression(boosting = "dart", drop_rate = 0.9)
    dart_max_drop = LightGBM.LGBMRegression(boosting = "dart", max_drop = 1)
    dart_skip_drop = LightGBM.LGBMRegression(boosting = "dart", skip_drop = 1.0)
    dart_xdm = LightGBM.LGBMRegression(boosting = "dart", xgboost_dart_mode = true)
    dart_uniform_drop = LightGBM.LGBMRegression(boosting = "dart", uniform_drop = true)
    dart_drop_seed = LightGBM.LGBMRegression(boosting = "dart", drop_seed = 20)

    LightGBM.fit!(dart_default, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_drop_rate, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_max_drop, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_skip_drop, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_xdm, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_uniform_drop, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(dart_drop_seed, X_train, y_train_regression, verbosity = -1)

    r_dart_default = LightGBM.predict(dart_default, X_test, verbosity = -1)
    r_dart_drop_rate = LightGBM.predict(dart_drop_rate, X_test, verbosity = -1)
    r_dart_max_drop = LightGBM.predict(dart_max_drop, X_test, verbosity = -1)
    r_dart_skip_drop = LightGBM.predict(dart_skip_drop, X_test, verbosity = -1)
    r_dart_xdm = LightGBM.predict(dart_xdm, X_test, verbosity = -1)
    r_dart_uniform_drop = LightGBM.predict(dart_uniform_drop, X_test, verbosity = -1)
    r_dart_drop_seed = LightGBM.predict(dart_drop_seed, X_test, verbosity = -1)

    @test norm(r_dart_default .- r_dart_drop_rate) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(r_dart_default .- r_dart_max_drop) >= sqrt(NSAMPLES_TEST) * 1e-10
    @test norm(r_dart_default .- r_dart_skip_drop) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(r_dart_default .- r_dart_xdm) >= sqrt(NSAMPLES_TEST) * 1e-10
    @test_broken norm(r_dart_default .- r_dart_uniform_drop) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test norm(r_dart_default .- r_dart_drop_seed) >= sqrt(NSAMPLES_TEST) * 1e-5

end

@testset "parameters -- goss classifcation" begin

    # the scheme is to set up a goss classifier and show that changing each one of
    # the parameters individually results in a difference to the model

    goss_default = LightGBM.LGBMClassification(boosting = "goss")
    goss_top_rate = LightGBM.LGBMClassification(boosting = "goss", top_rate = 0.9)
    goss_other_rate = LightGBM.LGBMClassification(boosting = "goss", other_rate = 0.8)


    LightGBM.fit!(goss_default, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(goss_top_rate, X_train, y_train_binary, verbosity = -1)
    LightGBM.fit!(goss_other_rate, X_train, y_train_binary, verbosity = -1)


    p_goss_default = LightGBM.predict(goss_default, X_test, verbosity = -1)
    p_goss_top_rate = LightGBM.predict(goss_top_rate, X_test, verbosity = -1)
    p_goss_other_rate = LightGBM.predict(goss_other_rate, X_test, verbosity = -1)


    @test_broken norm(p_goss_default .- p_goss_top_rate) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test_broken norm(p_goss_default .- p_goss_other_rate) >= sqrt(NSAMPLES_TEST) * 1e-5

end


@testset "parameters -- goss regression" begin

    # the scheme is to set up a goss regressor and show that changing each one of
    # the parameters individually results in a difference to the model

    goss_default = LightGBM.LGBMRegression(boosting = "goss")
    goss_top_rate = LightGBM.LGBMRegression(boosting = "goss", top_rate = 0.9)
    goss_other_rate = LightGBM.LGBMRegression(boosting = "goss", other_rate = 0.8)


    LightGBM.fit!(goss_default, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(goss_top_rate, X_train, y_train_regression, verbosity = -1)
    LightGBM.fit!(goss_other_rate, X_train, y_train_regression, verbosity = -1)


    r_goss_default = LightGBM.predict(goss_default, X_test, verbosity = -1)
    r_goss_top_rate = LightGBM.predict(goss_top_rate, X_test, verbosity = -1)
    r_goss_other_rate = LightGBM.predict(goss_other_rate, X_test, verbosity = -1)


    @test_broken norm(r_goss_default .- r_goss_top_rate) >= sqrt(NSAMPLES_TEST) * 1e-5
    @test_broken norm(r_goss_default .- r_goss_other_rate) >= sqrt(NSAMPLES_TEST) * 1e-5

end


end # end module
