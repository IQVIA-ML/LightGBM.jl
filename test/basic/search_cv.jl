module TestFit

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
params = [
    Dict(:num_iterations => num_iterations, :num_leaves => num_leaves)
    for num_iterations in (1, 2), num_leaves in (5, 10)
]

estimator = LightGBM.LGBMClassification(
    objective = "binary", 
    num_class = 1,
    is_training_metric = true, 
    metric = ["auc"]
)



@testset "test search_cv with X y input -- binary" begin
    # Act
    output = LightGBM.search_cv(estimator, train_matrix, train_labels, splits, params, verbosity = -1);

    # Assert
    @test output isa Array{Tuple{Dict{Symbol,Any},Dict{String,Dict{String,Vector{Float64}}}}}
    @test :num_iterations in keys(output[1][1])
    @test :num_leaves in keys(output[1][1])
    @test length(output[1][2]["validation"]["auc"]) == 5
    @test length(output[1][2]["training"]["auc"]) == 5
end


@testset "test search_cv with Dataset input -- binary" begin
    # Arrange
    train_dataset = LightGBM.LGBM_DatasetCreateFromMat(train_matrix, "")
    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)

    # Act
    output = LightGBM.search_cv(estimator, train_dataset, splits, params, verbosity = -1);

    # Assert
    @test output isa Array{Tuple{Dict{Symbol,Any},Dict{String,Dict{String,Vector{Float64}}}}}
    @test :num_iterations in keys(output[1][1])
    @test :num_leaves in keys(output[1][1])
    @test length(output[1][2]["validation"]["auc"]) == 5
    @test length(output[1][2]["training"]["auc"]) == 5
end



end # module