module TestParameters

using Test
using DelimitedFiles
using LightGBM
using LinearAlgebra


LGBM_PATH = ENV["LIGHTGBM_EXAMPLES_PATH"]
LGBM_PATH = if isabspath(LGBM_PATH) LGBM_PATH else abspath(joinpath(pwd(), "..", LGBM_PATH)) end

binary_test = readdlm(joinpath(LGBM_PATH, "examples", "binary_classification", "binary.test"), '\t')
binary_train = readdlm(joinpath(LGBM_PATH, "examples", "binary_classification", "binary.train"), '\t')
X_train = binary_train[:, 2:end]
y_train = binary_train[:, 1]
X_test = binary_test[:, 2:end]
y_test = binary_test[:, 1]

@testset "parameters -- boosting" begin

   estimator_gbdt = LightGBM.LGBMClassification(boosting = "gbdt")
   estimator_dart = LightGBM.LGBMClassification(boosting = "dart")
   LightGBM.fit!(estimator_dart, X_train, y_train, verbosity = -1)
   LightGBM.fit!(estimator_gbdt, X_train, y_train, verbosity = -1)
   p_gbdt = LightGBM.predict(estimator_gbdt, X_test, verbosity = -1)
   p_dart = LightGBM.predict(estimator_dart, X_test, verbosity = -1)

   # checking the models are sufficiently different by making sure
   #that the distance between the vectors is relatively large 
   @test norm(p_dart .- p_gbdt) >= sqrt(length(y_test)) * 1e-5 

end

@testset "parameters -- dart -- classifcation" begin

    dart_default = LightGBM.LGBMClassification(boosting = "dart")
    dart_drop_rate = LightGBM.LGBMClassification(boosting = "dart", drop_rate = 0.9)
    dart_max_drop = LightGBM.LGBMClassification(boosting = "dart", max_drop = 1)
    dart_skip_drop = LightGBM.LGBMClassification(boosting = "dart", skip_drop = 1.0)
    dart_xdm = LightGBM.LGBMClassification(boosting = "dart", xgboost_dart_mode = true)
    dart_uniform_drop = LightGBM.LGBMClassification(boosting = "dart", uniform_drop = true)
    dart_drop_seed = LightGBM.LGBMClassification(boosting = "dart", drop_seed = 20)

    LightGBM.fit!(dart_default, X_train, y_train, verbosity = -1)
    LightGBM.fit!(dart_drop_rate, X_train, y_train, verbosity = -1)
    LightGBM.fit!(dart_max_drop, X_train, y_train, verbosity = -1)
    LightGBM.fit!(dart_skip_drop, X_train, y_train, verbosity = -1)
    LightGBM.fit!(dart_xdm, X_train, y_train, verbosity = -1)
    LightGBM.fit!(dart_uniform_drop, X_train, y_train, verbosity = -1)
    LightGBM.fit!(dart_drop_seed, X_train, y_train, verbosity = -1)
    
    p_dart_default = LightGBM.predict(dart_default, X_test, verbosity = -1)
    p_dart_drop_rate = LightGBM.predict(dart_drop_rate, X_test, verbosity = -1)
    p_dart_max_drop = LightGBM.predict(dart_max_drop, X_test, verbosity = -1)
    p_dart_skip_drop = LightGBM.predict(dart_skip_drop, X_test, verbosity = -1)
    p_dart_xdm = LightGBM.predict(dart_xdm, X_test, verbosity = -1)
    p_dart_uniform_drop = LightGBM.predict(dart_uniform_drop, X_test, verbosity = -1)
    p_dart_drop_seed = LightGBM.predict(dart_drop_seed, X_test, verbosity = -1)

    @test norm(p_dart_default .- p_dart_drop_rate) >= sqrt(length(y_test)) * 1e-5
    @test norm(p_dart_default .- p_dart_max_drop) >= sqrt(length(y_test)) * 1e-10
    @test norm(p_dart_default .- p_dart_skip_drop) >= sqrt(length(y_test)) * 1e-5
    @test norm(p_dart_default .- p_dart_xdm) >= sqrt(length(y_test)) * 1e-10
    @test_broken norm(p_dart_default .- p_dart_uniform_drop) >= sqrt(length(y_test)) * 1e-5
    @test norm(p_dart_default .- p_dart_drop_seed) >= sqrt(length(y_test)) * 1e-5

end

end #end module