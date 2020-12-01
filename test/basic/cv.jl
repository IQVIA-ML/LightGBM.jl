module TestCv

include("../../src/LightGBM.jl")

using Test
using LightGBM

train_matrix = rand(1000,70) # create random dataset
train_labels = rand([0, 1], 1000)
splits = (
    collect(1:200), collect(201:400),
    collect(401:600), collect(601:800),
    collect(801:1000)
)

estimator = LightGBM.LGBMClassification(
    objective = "binary", 
    num_class = 1,
    is_training_metric = true, 
    metric = ["auc"]
)


@testset "test cv with X y input -- binary" begin
    # Act
    output = LightGBM.cv(estimator, train_matrix, train_labels, splits, verbosity = -1);

    # Assert
    @test output isa Dict{String,Dict{String,Vector{Float64}}}
    @test length(output["validation"]["auc"]) == 5
    @test length(output["training"]["auc"]) == 5
end


@testset "test cv with Dataset input -- binary" begin
    # Arrange
    train_dataset = LightGBM.LGBM_DatasetCreateFromMat(train_matrix, "")
    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)

    # Act
    output = LightGBM.cv(estimator, train_dataset, splits, verbosity = -1);

    # Assert
    @test output isa Dict{String,Dict{String,Vector{Float64}}}
    @test length(output["validation"]["auc"]) == 5
    @test length(output["training"]["auc"]) == 5
end



end # module
