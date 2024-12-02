module TestFit

using Test
using Dates
using LightGBM
using SparseArrays

# test fixtures
train_matrix = rand(5000,70) # create random dataset
train_sparse = sparse(train_matrix)
train_labels = rand([0, 1], 5000)
train_dataset = LightGBM.LGBM_DatasetCreateFromMat(train_matrix, "")
LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)


test_matrix = rand(2000,70) # create random dataset
test_labels = rand([0, 1], 2000)
test_dataset = LightGBM.LGBM_DatasetCreateFromMat(test_matrix, "", train_dataset)
LightGBM.LGBM_DatasetSetField(test_dataset, "label", test_labels)

test2_matrix = rand(1500,70) # create second random dataset
test2_labels = rand([0, 1], 1500)
test2_dataset = LightGBM.LGBM_DatasetCreateFromMat(test2_matrix, "", train_dataset)
LightGBM.LGBM_DatasetSetField(test2_dataset, "label", test2_labels)

@testset "test fit! with dataset -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(objective = "binary", num_class = 1, verbosity = -1)

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
    estimator = LightGBM.LGBMClassification(objective = "binary", num_class = 1, verbosity = -1)

    # Act
    LightGBM.fit!(estimator, train_dataset, verbosity = -1);
    p_binary = LightGBM.predict(estimator, train_matrix, verbosity = -1)
    binary_classes = LightGBM.predict_classes(estimator, train_matrix, verbosity = -1)

    # Assert
    # check also that binary outputs and classes are invariant in prediction output shapes, i.e. they're matrices
    @test length(size(p_binary)) == 2
    @test length(size(binary_classes)) == 2

end

@testset "test fit! with sparse matches dense" begin

    estimator_dense = LightGBM.LGBMClassification(objective = "binary", num_class = 1, verbosity = -1)
    estimator_sparse = LightGBM.LGBMClassification(objective = "binary", num_class = 1, verbosity = -1)

    LightGBM.fit!(estimator_dense, train_matrix, train_labels, verbosity = -1)
    LightGBM.fit!(estimator_sparse, train_sparse, train_labels, verbosity = -1)

    p_dense = LightGBM.predict(estimator_dense, test_matrix, verbosity = -1)
    p_sparse = LightGBM.predict(estimator_sparse, test_matrix, verbosity = -1)

    @test isapprox(p_dense, p_sparse, rtol=1e-16)

end

@testset "test train! single test set -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(
        objective = "binary",
        num_class = 1,
        is_provide_training_metric = true,
        metric = ["auc"],
        verbosity = -1
    )

    bst_parameters = LightGBM.stringifyparams(estimator)
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
        is_provide_training_metric = true,
        metric = ["auc", "l2"],
        verbosity = -1
    )

    bst_parameters = LightGBM.stringifyparams(estimator)
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
        is_provide_training_metric = true,
        metric = ["auc", "l2"],
        verbosity = -1
    )

    bst_parameters = LightGBM.stringifyparams(estimator)
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
        is_provide_training_metric = false, metric = ["auc"],
        early_stopping_round = 0, # default value, but stating explicitly to test!
        verbosity = -1
    )

    bst_parameters = LightGBM.stringifyparams(estimator)
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
        is_provide_training_metric = false, metric = ["auc"],
        early_stopping_round = 5,
        verbosity = -1
    )

    bst_parameters = LightGBM.stringifyparams(estimator)
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
    estimator = LightGBM.LGBMClassification(num_iterations = 100, verbosity = -1)
    verbosity = "verbose=-1"

    estimator.booster = LightGBM.LGBM_BoosterCreate(train_dataset, verbosity)

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

@testset "test fit! when truncate_booster" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(
        num_class = 1,
        num_iterations = 500,
        early_stopping_round = 5,
        metric = ["auc"],
        objective = "binary",
        verbosity = -1,
    )
    verbosity = "verbose=-1"

    # Act
    output = LightGBM.fit!(estimator, train_dataset, test_dataset; truncate_booster=true, verbosity=-1)

    # Assert
    eval_metrics_run_count = length(output["metrics"]["test_1"]["auc"])
    @test eval_metrics_run_count >= 5 # at least this will be more than or equal to early_stopping_round
    @test LightGBM.LGBM_BoosterGetCurrentIteration(estimator.booster) == output["best_iter"] # test this matches as expected

end

@testset "test fit! when NOT truncate_booster" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(
        num_class = 1,
        num_iterations = 200,
        early_stopping_round = 5,
        metric = ["auc"],
        objective = "binary",
        verbosity = -1,
    )
    verbosity = "verbose=-1"

    # Act
    output = LightGBM.fit!(estimator, train_dataset, test_dataset; truncate_booster=false, verbosity=-1)

    # Assert
    eval_metrics_run_count = length(output["metrics"]["test_1"]["auc"])
    @test eval_metrics_run_count >= 5 # at least this will be more than or equal to early_stopping_round
    @test LightGBM.LGBM_BoosterGetCurrentIteration(estimator.booster) >= 5 # given no truncation, this must be greater than early_stopping_round
    @test LightGBM.LGBM_BoosterGetCurrentIteration(estimator.booster) > output["best_iter"] # unless this runs to the very last num_iterations this should pass...should buy the lottery if this fails

end

@testset "test fit! when truncate_booster inactive when no early_stopping" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(
        num_class = 1,
        num_iterations = 100,
        early_stopping_round = 0,
        metric = ["auc"],
        objective = "binary",
        verbosity = -1,
    )
    verbosity = "verbose=-1"

    # Act
    output = LightGBM.fit!(estimator, train_dataset, test_dataset; truncate_booster=true, verbosity=-1)

    # Assert
    eval_metrics_run_count = length(output["metrics"]["test_1"]["auc"])
    @test eval_metrics_run_count == 100
    @test LightGBM.LGBM_BoosterGetCurrentIteration(estimator.booster) == 100
    @test output["best_iter"] == 0

end

@testset "stringifyparams -- convert to zero-based" begin
    indices = [1, 3, 5, 7, 9]
    interaction_constraints = [[1, 3], [5, 7], [9]]
    classifier = LightGBM.LGBMClassification(
        categorical_feature = indices, 
        interaction_constraints = interaction_constraints, 
        verbosity = -1)
    ds_parameters = LightGBM.stringifyparams(classifier)

    expected = "categorical_feature=0,2,4,6,8"
    expected_constraints = "interaction_constraints=[0,2],[4,6],[8]"
    @test occursin(expected, ds_parameters)
    @test occursin(expected_constraints, ds_parameters)
end

@testset "stringifyparams -- multiple calls won't mutate fields" begin
    indices = [1, 3, 5, 7, 9]
    classifier = LightGBM.LGBMClassification(categorical_feature = indices, verbosity = -1)
    expected_indices = deepcopy(classifier.categorical_feature)

    LightGBM.stringifyparams(classifier)
    @test expected_indices == classifier.categorical_feature

    LightGBM.stringifyparams(classifier)
    @test expected_indices == classifier.categorical_feature
end



end # module
