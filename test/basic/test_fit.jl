module TestFit

using Test
using Dates
using LightGBM

# test fixtures
train_matrix = rand(10000,70) # create random dataset
train_labels = rand([0, 1], 10000)
train_dataset = LightGBM.LGBM_DatasetCreateFromMat(train_matrix, "")

test_matrix = rand(5000,70) # create random dataset
test_labels = rand([0, 1], 5000)
test_dataset = LightGBM.LGBM_DatasetCreateFromMat(test_matrix, "", train_dataset)

test2_matrix = rand(2500,70) # create second random dataset
test2_labels = rand([0, 1], 2500)
test2_dataset = LightGBM.LGBM_DatasetCreateFromMat(test2_matrix, "", train_dataset)

@testset "test fit! with dataset -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(objective = "binary", num_class = 1)
    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)
    LightGBM.LGBM_DatasetSetField(test_dataset, "label", test_labels)
    
    # Act
    LightGBM.fit!(estimator, train_dataset, test_dataset, verbosity = -1);
    p_binary = LightGBM.predict(estimator, train_matrix, verbosity = -1)
    binary_classes = LightGBM.predict_classes(estimator, train_matrix, verbosity = -1)

    # Assert
    # check also that binary outputs and classes are invariant in prediction output shapes, i.e. they're matrices
    @test length(size(p_binary)) == 2
    @test length(size(binary_classes)) == 2
    
end

@testset "test fit! with dataset without testset -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(objective = "binary", num_class = 1)
    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)

    # Act
    LightGBM.fit!(estimator, train_dataset, verbosity = -1);
    p_binary = LightGBM.predict(estimator, train_matrix, verbosity = -1)
    binary_classes = LightGBM.predict_classes(estimator, train_matrix, verbosity = -1)

    # Assert
    # check also that binary outputs and classes are invariant in prediction output shapes, i.e. they're matrices
    @test length(size(p_binary)) == 2
    @test length(size(binary_classes)) == 2
    
end

@testset "test train! single test set -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(
        objective = "binary", 
        num_class = 1,
        is_training_metric = true, 
        metric = ["auc"],
    )

    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)
    LightGBM.LGBM_DatasetSetField(test_dataset, "label", test_labels)

    bst_parameters = LightGBM.stringifyparams(estimator, LightGBM.BOOSTERPARAMS) * " verbosity=-1"
    estimator.booster = LightGBM.LGBM_BoosterCreate(train_dataset, bst_parameters)
    LightGBM.LGBM_BoosterAddValidData(estimator.booster, test_dataset)

    # Act
    results = LightGBM.train!(estimator, 10, ["test_set_with_whatever_label"], -1, DateTime(2020, 1, 11, 11, 00));

    # Assert
    @test length(results["metrics"]["training"]["auc"]) > 0
    @test length(results["metrics"]["test_set_with_whatever_label"]["auc"]) > 0
    
end


@testset "test train! multiple test set multiple metrics -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(
        objective = "binary", 
        num_class = 1,
        is_training_metric = true, 
        metric = ["auc", "l2"],
    )

    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)
    LightGBM.LGBM_DatasetSetField(test_dataset, "label", test_labels)
    LightGBM.LGBM_DatasetSetField(test2_dataset, "label", test2_labels)

    bst_parameters = LightGBM.stringifyparams(estimator, LightGBM.BOOSTERPARAMS) * " verbosity=-1"
    estimator.booster = LightGBM.LGBM_BoosterCreate(train_dataset, bst_parameters)
    LightGBM.LGBM_BoosterAddValidData(estimator.booster, test_dataset)
    LightGBM.LGBM_BoosterAddValidData(estimator.booster, test2_dataset)

    # Act
    results = LightGBM.train!(estimator, 10, ["test_1", "test_2"], -1, DateTime(2020, 1, 11, 11, 00));

    # Assert
    @test length(results["metrics"]["training"]["auc"]) > 0
    @test length(results["metrics"]["training"]["l2"]) > 0
    @test length(results["metrics"]["test_1"]["auc"]) > 0
    @test length(results["metrics"]["test_1"]["l2"]) > 0
    @test length(results["metrics"]["test_2"]["auc"]) > 0
    @test length(results["metrics"]["test_2"]["l2"]) > 0
    
end



@testset "test train! multiple test set multiple metrics -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(
        objective = "binary", 
        num_class = 1,
        is_training_metric = true, 
        metric = ["auc", "l2"],
    )

    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)
    LightGBM.LGBM_DatasetSetField(test_dataset, "label", test_labels)
    LightGBM.LGBM_DatasetSetField(test2_dataset, "label", test2_labels)

    bst_parameters = LightGBM.stringifyparams(estimator, LightGBM.BOOSTERPARAMS) * " verbosity=-1"
    estimator.booster = LightGBM.LGBM_BoosterCreate(train_dataset, bst_parameters)
    LightGBM.LGBM_BoosterAddValidData(estimator.booster, test_dataset)
    LightGBM.LGBM_BoosterAddValidData(estimator.booster, test2_dataset)

    # Act
    results = LightGBM.train!(estimator, 10, ["test_1", "test_2"], -1, DateTime(2020, 1, 11, 11, 00));

    # Assert
    @test length(results["metrics"]["training"]["auc"]) > 0
    @test length(results["metrics"]["training"]["l2"]) > 0
    @test length(results["metrics"]["test_1"]["auc"]) > 0
    @test length(results["metrics"]["test_1"]["l2"]) > 0
    @test length(results["metrics"]["test_2"]["auc"]) > 0
    @test length(results["metrics"]["test_2"]["l2"]) > 0
    
end


@testset "test store_scores! adds results" begin
    # Arrange
    results_fixture = Dict("metrics" => Dict{String,Dict{String,Vector{Float64}}}())

    # Act
    LightGBM.store_scores!(results_fixture, "whatever_dataset","some_metric",0.5)
    LightGBM.store_scores!(results_fixture, "whatever_dataset","some_metric",0.55)

    LightGBM.store_scores!(results_fixture, "whatever_dataset","some_extra_metric",0.6)
    LightGBM.store_scores!(results_fixture, "whatever_dataset","some_extra_metric",0.65)

    LightGBM.store_scores!(results_fixture, "whatever_second_dataset","some_metric",0.7)
    LightGBM.store_scores!(results_fixture, "whatever_second_dataset","some_metric",0.75)
    
    # Assert
    @test results_fixture["metrics"]["whatever_dataset"]["some_metric"] == [0.5, 0.55]
    @test results_fixture["metrics"]["whatever_dataset"]["some_extra_metric"] == [0.6, 0.65]
    @test results_fixture["metrics"]["whatever_second_dataset"]["some_metric"] == [0.7, 0.75]
        
end



@testset "test eval_metrics! early stop disabled" begin
    # Arrange
    results_fixture = Dict(
        "best_iter" => 0,
        "metrics" => Dict{String,Dict{String,Vector{Float64}}}(),
    )
    estimator = LightGBM.LGBMClassification(
        num_iterations = 10, objective = "binary", num_class = 1, 
        is_training_metric = false, metric = ["auc"], 
        early_stopping_round = 0 # default value, but stating explicitly to test!
    )
    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)
    LightGBM.LGBM_DatasetSetField(test_dataset, "label", test_labels)

    bst_parameters = LightGBM.stringifyparams(estimator, LightGBM.BOOSTERPARAMS) * " verbosity=-1"
    estimator.booster = LightGBM.LGBM_BoosterCreate(train_dataset, bst_parameters)
    LightGBM.LGBM_BoosterAddValidData(estimator.booster, test_dataset)

    bigger_is_better = Dict{String,Float64}("auc" => 1.0)
    best_scores = Dict{String,Dict{String,Real}}("auc" => Dict("test_bla" => 1))
    best_iterations = Dict{String,Dict{String,Real}}("auc" => Dict("test_bla" => 1))

    # Act and assert each round returns false (despite iter 1 have best score always)
    for iter in 1:10
        LightGBM.LGBM_BoosterUpdateOneIter(estimator.booster)
        output = LightGBM.eval_metrics!(
            results_fixture, estimator, ["test_bla"], iter, -1,
            bigger_is_better, best_scores, best_iterations, ["auc"]
        )

        @test output == false

    end
end

#=
This test assumes:
* early_stopping_round set to 5
* total iterations = 10
* best iter is ALWAYS round 1 of metric score 1 (auc)
Criteria: early_stopping should kick in on round 6
=#
@testset "test eval_metrics! stops correctly" begin
    # Arrange
    results_fixture = Dict(
        "best_iter" => 0,
        "metrics" => Dict{String,Dict{String,Vector{Float64}}}(),
    )
    estimator = LightGBM.LGBMClassification(
        num_iterations = 10, objective = "binary", num_class = 1, 
        is_training_metric = false, metric = ["auc"], 
        early_stopping_round = 5
    )
    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)
    LightGBM.LGBM_DatasetSetField(test_dataset, "label", test_labels)

    bst_parameters = LightGBM.stringifyparams(estimator, LightGBM.BOOSTERPARAMS) * " verbosity=-1"
    estimator.booster = LightGBM.LGBM_BoosterCreate(train_dataset, bst_parameters)
    LightGBM.LGBM_BoosterAddValidData(estimator.booster, test_dataset)

    bigger_is_better = Dict{String,Float64}("auc" => 1.0)
    best_scores = Dict{String,Dict{String,Real}}("auc" => Dict("test_bla" => 1))
    best_iterations = Dict{String,Dict{String,Real}}("auc" => Dict("test_bla" => 1))

    # Act and assert each round returns expected output
    for iter in 1:10
        LightGBM.LGBM_BoosterUpdateOneIter(estimator.booster)
        output = LightGBM.eval_metrics!(
            results_fixture, estimator, ["test_bla"], iter, -1,
            bigger_is_better, best_scores, best_iterations, ["auc"]
        )
        
        # reset scores to round 1 being best
        best_scores = Dict{String,Dict{String,Real}}("auc" => Dict("test_bla" => 1))
        best_iterations = Dict{String,Dict{String,Real}}("auc" => Dict("test_bla" => 1))

        if iter < 6
            @test output == false
        else
            @test output == true
        end
    end
end


@testset "test truncate_model!" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(num_iterations = 100)
    verbosity = "verbose=-1"
    mymat = randn(10000, 2)
    labels = randn(10000)
    dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)    
    LightGBM.LGBM_DatasetSetField(dataset, "label", labels)
    estimator.booster = LightGBM.LGBM_BoosterCreate(dataset, verbosity)

    # update learning 100 times
    for _ in [1:100;]
        finished = LightGBM.LGBM_BoosterUpdateOneIter(estimator.booster)
    end
    @test LightGBM.LGBM_BoosterGetCurrentIteration(estimator.booster) == 100
    
    # Act
    LightGBM.truncate_model!(estimator, 87)

    # Assert
    @test LightGBM.LGBM_BoosterGetCurrentIteration(estimator.booster) == 87
end


end # module
