module TestFFIBooster

using LightGBM
using Test
using Random


verbosity = "verbose=-1"


@testset "LGBM_BoosterCreate" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)

    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)

    @test booster.handle != C_NULL

end


@testset "LGBM_BoosterCreateFromModelfile" begin

    booster = LightGBM.LGBM_BoosterCreateFromModelfile(joinpath(@__DIR__, "data", "test_tree"))
    @test booster.handle != C_NULL

end


@testset "LGBM_BoosterLoadModelFromString" begin

    load_str = read(joinpath(@__DIR__, "data", "test_tree"), String)
    booster = LightGBM.LGBM_BoosterLoadModelFromString(load_str)
    @test booster.handle != C_NULL

end


@testset "LGBM_BoosterFree" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)

    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)

    # These tests exposed a double-free
    @test LightGBM.LGBM_BoosterFree(booster) == nothing
    @test booster.handle == C_NULL

end


@testset "LGBM_BoosterMerge" begin

    booster1 = LightGBM.LGBM_BoosterCreateFromModelfile(joinpath(@__DIR__, "data", "test_tree"))
    booster2 = LightGBM.LGBM_BoosterCreateFromModelfile(joinpath(@__DIR__, "data", "test_tree"))

    @test booster1.handle != booster2.handle

    # record the pointer values so we can verify they dont get mutated as part of the ops
    handle1 = UInt64(booster1.handle)
    handle2 = UInt64(booster2.handle)

    @test LightGBM.LGBM_BoosterMerge(booster1, booster2) == nothing

    @test booster1.handle != C_NULL
    @test booster2.handle != C_NULL

    @test UInt64(booster1.handle) == handle1
    @test UInt64(booster2.handle) == handle2

    @test booster1.handle != booster2.handle

end


@testset "LGBM_BoosterAddValidData" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    v_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat .+ 1., verbosity)

    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)

    @test LightGBM.LGBM_BoosterAddValidData(booster, v_dataset) == nothing
    @test v_dataset.handle in getfield.(booster.datasets, :handle)
    @test length(booster.datasets) == 2

end


@testset "LGBM_BoosterResetTrainingData" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    e_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat .+ 1., verbosity)
    v_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat .+ 2., verbosity)

    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)

    @test LightGBM.LGBM_BoosterResetTrainingData(booster, e_dataset) == nothing
    @test length(booster.datasets) == 1
    @test first(booster.datasets).handle == e_dataset.handle

    # rewrite it but add another dataset
    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)
    LightGBM.LGBM_BoosterAddValidData(booster, v_dataset)

    @test LightGBM.LGBM_BoosterResetTrainingData(booster, e_dataset) == nothing
    @test length(booster.datasets) == 2
    @test first(booster.datasets).handle == e_dataset.handle
    @test last(booster.datasets).handle == v_dataset.handle

end


@testset "LGBM_BoosterGetNumClasses" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    v_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat .+ 1., verbosity)

    for n in 2:20

        booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=multiclass num_class=$(n) $verbosity")

        @test LightGBM.LGBM_BoosterGetNumClasses(booster) == n

    end

end


@testset "LGBM_BoosterUpdateOneIter" begin

    mymat = [2. 1.; 4. 3.; 5. 6.; 3. 4.; 6. 5.; 2. 1.]
    labels = [1.f0, 0.f0, 1.f0, 1.f0, 0.f0, 0.f0]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)
    # default params won't allow this to learn anything from this useless data set (i.e. splitting completes)
    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)

    finished = LightGBM.LGBM_BoosterUpdateOneIter(booster)
    @test finished == 1
    finished = LightGBM.LGBM_BoosterUpdateOneIter(booster)
    @test finished == 1 # just checking nothing silly happens if you try to continue

    # Feed the tree nuts data so it has a hard time learning
    mymat = randn(1000, 2)
    labels = randn(1000)
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)

    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)
    finished = LightGBM.LGBM_BoosterUpdateOneIter(booster)
    @test finished == 0

end


@testset "LGBM_BoosterUpdateOneIterCustom" begin

    numdata = 1000
    mymat = randn(numdata, 2)
    labels = dropdims(sum(mymat .^2; dims=2); dims=2)
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)
    # default params won't allow this to learn anything from this useless data set (i.e. splitting completes)
    booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=none $verbosity")

    finished = LightGBM.LGBM_BoosterUpdateOneIterCustom(booster, randn(numdata), ones(numdata))
    pred1 = LightGBM.LGBM_BoosterGetPredict(booster, 0)
    # check both types of float work
    finished = LightGBM.LGBM_BoosterUpdateOneIterCustom(booster, Float32.(randn(numdata)), Float32.(ones(numdata)))
    pred2 = LightGBM.LGBM_BoosterGetPredict(booster, 0)
    @test !isapprox(pred1, pred2; rtol=1e-5) # show that the gradients caused an update

    finished = LightGBM.LGBM_BoosterUpdateOneIterCustom(booster, zeros(numdata), ones(numdata))
    pred3 = LightGBM.LGBM_BoosterGetPredict(booster, 0)
    @test isapprox(pred2, pred3; rtol=1e-16) # show that the gradients did not cause an update

    @test_throws DimensionMismatch LightGBM.LGBM_BoosterUpdateOneIterCustom(booster, zeros(1), zeros(1))

    existing_booster = LightGBM.LGBM_BoosterCreateFromModelfile(joinpath(@__DIR__, "data", "test_tree"))

    # can't exactly match the size if there is no size (no training data) to match
    @test_throws ErrorException LightGBM.LGBM_BoosterUpdateOneIterCustom(existing_booster, zeros(1), zeros(1))

    # handle multiclass too
    num_class = 3
    mymat = randn(numdata, 2)
    labels = rand((1:num_class) .- 1, numdata)

    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)
    booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=none num_class=$(num_class) $verbosity")

    finished = LightGBM.LGBM_BoosterUpdateOneIterCustom(booster, randn(numdata*num_class), rand(numdata*num_class))
    pred1 = LightGBM.LGBM_BoosterGetPredict(booster, 0)
    finished = LightGBM.LGBM_BoosterUpdateOneIterCustom(booster, randn(numdata*num_class), rand(numdata*num_class))
    pred2 = LightGBM.LGBM_BoosterGetPredict(booster, 0)

    @test !isapprox(pred1, pred2; rtol=1e-5) # show that the gradients caused an update

    # check the naive silly thing does in fact not get accepted
    @test_throws DimensionMismatch LightGBM.LGBM_BoosterUpdateOneIterCustom(booster, Float32.(randn(numdata)), Float32.(rand(numdata)))

end


@testset "LGBM_BoosterRollbackOneIter" begin

    # Arrange
    mymat = randn(10000, 2)
    labels = randn(10000)
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)
    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)

    # update learning 20 times
    for _ in [1:20;]
        finished = LightGBM.LGBM_BoosterUpdateOneIter(booster)
    end

    # Act and Assert
    for n in reverse([1:20;])
        @test LightGBM.LGBM_BoosterGetCurrentIteration(booster) == n

        model_string = LightGBM.LGBM_BoosterSaveModelToString(booster)
        # pull out the line in the string on tree sizes (only one line should be return, so take the first element)
        tree_sizes_in_string = match(r"\ntree_sizes=(.*)\n", model_string)[1]
        # tree sizes are delimited by space, so we can count the number of trees easily
        @test length(split(tree_sizes_in_string, " ")) == n

        model_loaded_from_string = LightGBM.LGBM_BoosterLoadModelFromString(model_string)
        @test LightGBM.LGBM_BoosterGetCurrentIteration(model_loaded_from_string) == n

        LightGBM.LGBM_BoosterRollbackOneIter(booster)

    end

end


@testset "LGBM_BoosterGetCurrentIteration" begin

    mymat = [2. 1.; 4. 3.; 5. 6.; 3. 4.; 6. 5.; 2. 1.]
    labels = [1.f0, 0.f0, 1.f0, 1.f0, 0.f0, 0.f0]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)
    # default params won't allow this to learn anything from this useless data set (i.e. splitting completes)
    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)

    # check its finished
    @test LightGBM.LGBM_BoosterUpdateOneIter(booster) == 1

    for i in 1:10

        # Iteration number shouldn't increment cause we're not adding any more trees
        LightGBM.LGBM_BoosterUpdateOneIter(booster)
        @test LightGBM.LGBM_BoosterGetCurrentIteration(booster) == 1

    end

    # do again with nuts data
    mymat = randn(1000, 2)
    labels = randn(1000)
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)

    # deliberately constrain the trees to ensure it can't converge by 10 gradient boosts by small chance
    booster = LightGBM.LGBM_BoosterCreate(dataset, "max_depth=1 num_leaf=2 $verbosity")

    for i in 1:10

        LightGBM.LGBM_BoosterUpdateOneIter(booster)
        @test LightGBM.LGBM_BoosterGetCurrentIteration(booster) == i

    end

end


@testset "LGBM_BoosterGetEvalCounts" begin

    # Unsure how to test
    @test_broken false

end


@testset "LGBM_BoosterGetEvalNames" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    booster = LightGBM.LGBM_BoosterCreate(dataset, "metric=binary_logloss $(verbosity)")
    @test LightGBM.LGBM_BoosterGetEvalNames(booster) |> first == "binary_logloss"

end


@testset "LGBM_BoosterGetFeatureNames" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    randfeatures = [randstring(2000), randstring(3000)]
    LightGBM.LGBM_DatasetSetFeatureNames(dataset, randfeatures)
    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)
    @test LightGBM.LGBM_BoosterGetFeatureNames(booster) == randfeatures

end


@testset "LGBM_BoosterGetNumFeature" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)
    @test LightGBM.LGBM_BoosterGetNumFeature(booster) == 2

end


@testset "LGBM_BoosterGetEval" begin

    # Needs implementing
    @test_broken false

end


@testset "LGBM_BoosterGetNumPredict" begin

    # Needs implementing
    @test_broken false

end


@testset "LGBM_BoosterGetPredict" begin
    # Regression test
    numdata = 1000
    mymat = randn(numdata, 2)
    labels = dropdims(sum(mymat .^2; dims=2); dims=2)
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)
    booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=regression $verbosity")
    for _ in 1:10
        LightGBM.LGBM_BoosterUpdateOneIter(booster)
    end
    preds_regression = LightGBM.LGBM_BoosterGetPredict(booster, 0)
    # predictions should equal to 1 * numdata for regression
    @test length(preds_regression) == numdata

    # Classification test
    labels = rand([0, 1], numdata)
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)
    booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=binary num_class=1 $verbosity")
    for _ in 1:10
        LightGBM.LGBM_BoosterUpdateOneIter(booster)
    end
    preds_classification = LightGBM.LGBM_BoosterGetPredict(booster, 0)
    # predictions should equal to num_class * numdata (1 * numdata for binary classification)
    @test length(preds_classification) == numdata

    # Multi-class test
    num_classes = 3
    labels = rand(0:num_classes-1, numdata)
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)
    booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=multiclass num_class=$num_classes $verbosity")
    for _ in 1:10
        LightGBM.LGBM_BoosterUpdateOneIter(booster)
    end
    preds_multiclass = LightGBM.LGBM_BoosterGetPredict(booster, 0)
    # predictions should equal to num_class * numdata (3 * numdata for this example multi-class classification)
    @test length(preds_multiclass) == num_classes * numdata

end


@testset "LGBM_BoosterPredictForFile" begin
    # Setup test data from file
    data = [
        "feature1,feature2,label",
        "1.0,2.0,0",
        "2.0,3.0,1",
        "3.0,4.0,0",
        "4.0,5.0,1"
    ]
    data_filename = "dummy_data.csv"
    result_filename = "dummy_predictions.csv"
    open(data_filename, "w") do f
        for line in data
            println(f, line)
        end
    end
   # Setup booster
    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)

    # Call the LGBM_BoosterPredictForFile function
    LightGBM.LGBM_BoosterPredictForFile(
        booster, 
        data_filename, 
        true,  # data_has_header
        0,  # predict_type (0 for raw score)
        0,  # start_iteration
        0,  # num_iteration
        "", # no parameters
        result_filename, 
    )

    # Check if the result file is created
    @test isfile(result_filename)

    # Clean up
    rm(data_filename, force=true)
    rm(result_filename, force=true)

end


@testset "LGBM_BoosterCalcNumPredict" begin
    # Setup train data and booster
    mymat = rand(100, 10)  # 100 rows and 10 features
    labels = rand([0, 1], 100)
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)
    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)
    num_rows = size(mymat, 1)
    num_features = size(mymat, 2)

    # Train the booster with a known number of trees
    num_iterations = 10
    for i in 1:num_iterations
        LightGBM.LGBM_BoosterUpdateOneIter(booster)
    end

    # Verify the number of iterations completed
    num_trees = LightGBM.LGBM_BoosterGetCurrentIteration(booster)

    # Test for normal prediction (predict_type = 0, start_iteration = 0, num_iteration = num_iterations)
    # For normal prediction, the number of predictions should be equal to the number of rows
    # multiplied by the number of classes in case of multi-class classification or 1 for binary classification and regression
    preds_normal = LightGBM.LGBM_BoosterCalcNumPredict(booster, num_rows, 0, 0, num_iterations)
    @test preds_normal == num_rows

    # Test for raw score (predict_type = 1, start_iteration = 0, num_iteration = num_iterations)
    # Similar to normal prediction, the number of predictions should be equal to the number of rows or multiplied by the number of classes
    preds_raw = LightGBM.LGBM_BoosterCalcNumPredict(booster, num_rows, 1, 0, num_iterations)
    @test preds_raw == num_rows

    # Test for predict leaf index (predict_type = 2, start_iteration = 0, num_iteration = num_iterations)
    # The number of predictions should be equal to the number of rows multiplied by the number of trees
    # As it returns the leaf indices for each tree in the model for each data point
    preds_leaf = LightGBM.LGBM_BoosterCalcNumPredict(booster, num_rows, 2, 0, num_iterations)
    @test preds_leaf == num_rows * num_trees

    # Test for predict contribution (predict_type = 3, start_iteration = 0, num_iteration = num_iterations)
    # This returns the SHAP values (feature contributions) for each feature in the dataset and an additional bias term/base value
    preds_contrib = LightGBM.LGBM_BoosterCalcNumPredict(booster, num_rows, 3, 0, num_iterations)
    @test preds_contrib == num_rows * (num_features + 1)

end


@testset "LGBM_BoosterPredictForMat" begin
    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)

    preds = LightGBM.LGBM_BoosterPredictForMat(booster, mymat, 0, 0, -1) 
    @test all(iszero, preds)

    preds_threaded = LightGBM.LGBM_BoosterPredictForMat(booster, mymat, 0, 0, -1, false, "num_threads=2") 
    @test all(iszero, preds_threaded)
end


@testset "LGBM_BoosterRefit" begin
    # Sample dataset
    X_train = randn(1000, 20)
    y_train = rand([0, 1], 1000)
    
    # Create an estimator with predict_leaf_index set to true to obtain leaf index predictions
    estimator = LightGBM.LGBMClassification(objective = "binary", num_class = 1, predict_leaf_index = true, num_iterations = 20, verbosity = -1)
    
    # Fit the estimator with the training data
    LightGBM.fit!(estimator, X_train, y_train, verbosity = -1)
    
    # Get the leaf predictions using the training data
    leaf_predictions = LightGBM.predict(estimator, X_train)

    # Convert to Int32 which is expected by the C API and should be for leaf indices
    # Reshape the leaf predictions as their number of rows is number of data points/rows * num_trees/iterations
    reshaped_leaf_predictions = reshape(convert(Matrix{Int32}, leaf_predictions), (size(X_train, 1), estimator.num_iterations))

    # Refit the model using the reshaped leaf predictions
    result = LightGBM.LGBM_BoosterRefit(estimator.booster, reshaped_leaf_predictions)
    
    # Test that LGBM_BoosterRefit returns nothing (meaning successful)
    @test result == nothing
 
end


@testset "LGBM_BoosterSaveModel" begin

    booster = LightGBM.LGBM_BoosterCreateFromModelfile(joinpath(@__DIR__, "data", "test_tree"))
    model_path = joinpath(@__DIR__, "data", "test_model.txt")
    if isfile(model_path)
        rm(model_path, force = true)
    end
    LightGBM.LGBM_BoosterSaveModel(booster, 0, 0, 0, model_path )
    @test isfile(model_path)
    rm(model_path; force = true)

end


@testset "LGBM_BoosterSaveModelToString" begin

    booster = LightGBM.LGBM_BoosterCreateFromModelfile(joinpath(@__DIR__, "data", "test_tree"))
    string_repr = LightGBM.LGBM_BoosterSaveModelToString(booster)
    # so it turns out that the string save and file save aren't necesarily the same so..
    # check a bunch of expected substrings, etc
    @test occursin(r"version=v[3-9]", string_repr)
    @test occursin("num_leaves=1", string_repr)
    @test occursin("end of trees", string_repr)
    @test occursin("feature_importances:", string_repr)
    @test occursin("parameters:", string_repr)
    @test occursin("[convert_model: gbdt_prediction.cpp]", string_repr)
    @test occursin("Tree=0", string_repr)
    @test occursin("end of parameters", string_repr)

    # this is an additional test to check the presence and correctness of split and gain after saving model to string
    # due to LGBM v3.0.0 changing the LGBM_BoosterSaveModelToString API that now includes an additional parameter
    # for either split or gain parameters
    # this test is to show that the correct split and gain can be computed regardless of whether 0(split) or 1(gain)
    # `feature_importance_type` parameter is used to save model

    # load a model from file
    booster = LightGBM.LGBM_BoosterCreateFromModelfile(joinpath(@__DIR__, "data", "gain_test_booster"))
    # compute both its gain and split importance
    split_importance = LightGBM.LGBM_BoosterFeatureImportance(booster, 0, 0)
    gain_importance = LightGBM.LGBM_BoosterFeatureImportance(booster, 0, 1)
    # save the model twice, once with each of those gain and split parameters
    split_model_save = LightGBM.LGBM_BoosterSaveModelToString(booster, 0, 0, 0)
    gain_model_save = LightGBM.LGBM_BoosterSaveModelToString(booster, 0, 0, 1)
    # load each of the saved models to a new model
    split_load_model = LightGBM.LGBM_BoosterLoadModelFromString(split_model_save)
    gain_load_model = LightGBM.LGBM_BoosterLoadModelFromString(gain_model_save)
    # for EACH model, compute gain and split importances
    split_importance_from_split_loaded_model = LightGBM.LGBM_BoosterFeatureImportance(split_load_model, 0, 0)
    gain_importance_from_split_loaded_model= LightGBM.LGBM_BoosterFeatureImportance(split_load_model, 0, 1)
    gain_importance_from_gain_loaded_model = LightGBM.LGBM_BoosterFeatureImportance(gain_load_model, 0, 1)
    split_importance_from_gain_loaded_model = LightGBM.LGBM_BoosterFeatureImportance(gain_load_model, 0, 0)
    # and show that both values are the same as the originals for both models
    @test split_importance == split_importance_from_split_loaded_model
    @test split_importance == split_importance_from_gain_loaded_model
    @test gain_importance == gain_importance_from_split_loaded_model
    @test gain_importance == gain_importance_from_gain_loaded_model

end


@testset "LGBM_BoosterFeatureImportance" begin

    booster_path = joinpath(@__DIR__, "data", "gain_test_booster")
    booster = LightGBM.LGBM_BoosterCreateFromModelfile(booster_path)

    split_importance = LightGBM.LGBM_BoosterFeatureImportance(booster, 0, 0)
    gain_importance = LightGBM.LGBM_BoosterFeatureImportance(booster, 0, 1)

    split_sub_importance = LightGBM.LGBM_BoosterFeatureImportance(booster, 2, 0)
    gain_sub_importance = LightGBM.LGBM_BoosterFeatureImportance(booster, 2, 1)

    # this is copy/paste from python output without tweaks to display nums, it should be fine though
    # splits are actually ints
    expected_gain = [89.73980999, 65.49232054, 112.80447054, 107.81817985, 124.81229973]
    expected_split = [17, 11, 19, 20, 23]

    @test isapprox(gain_importance, expected_gain, atol=1e-4)
    @test split_importance == expected_split

    expected_sub_gain = [69.14388967, 52.85783052, 75.43093014, 56.17689991, 96.83372998]
    expected_sub_split = [12,  8, 13, 10, 17]

    @test isapprox(gain_sub_importance, expected_sub_gain, atol=1e-4)
    @test split_sub_importance == expected_sub_split

end


@testset "LGBM_BoosterGetLinear" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    # non-linear example
    booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=binary")
    @test LightGBM.LGBM_BoosterGetLinear(booster) == 0
    # linear example
    linear_booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=binary linear_tree=true")
    @test LightGBM.LGBM_BoosterGetLinear(linear_booster) == 1

end


@testset "LGBM_BoosterNumModelPerIteration" begin


    mymat = [1. 2.; 3. 4.; 5. 6.]
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)

    booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=binary $verbosity")

    @test LightGBM.LGBM_BoosterNumModelPerIteration(booster) == 1

    booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=regression $verbosity")

    @test LightGBM.LGBM_BoosterNumModelPerIteration(booster) == 1

    for n in 2:20

        booster = LightGBM.LGBM_BoosterCreate(dataset, "objective=multiclass num_class=$(n) $verbosity")

        @test LightGBM.LGBM_BoosterNumModelPerIteration(booster) == n

    end

end

end # module
