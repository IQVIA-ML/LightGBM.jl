module TestEvaluationMetrics

using Test
using LightGBM


@testset "merge_scores" begin

    a = Dict(
        "dataset_a" => Dict(
            "some_metric" => [1., 2.],
            "another_metric" => [2.2, 3.3],
        ),
        "dataset_b" => Dict(
            "dodo_metric" => [0., 0.],
            "great_metric" => [0.1, 0.2],
        ),
    )

    b = Dict(
        "dataset_a" => Dict(
            "some_metric" => [3., 4.],
            "another_metric" => [4.4, 5.5],
        ),
        "dataset_b" => Dict(
            "dodo_metric" => [1., 1.],
            "great_metric" => [0.3, 0.4],
        ),
    )

    a_copy = deepcopy(a)
    b_copy = deepcopy(b)

    c = LightGBM.merge_scores(a, b)

    # check no mutate
    @test a == a_copy
    @test b == b_copy

    # check no weird key munging from top/lower layer
    @test sort(collect(keys(c))) == ["dataset_a", "dataset_b"]
    @test sort(collect(keys(c["dataset_a"]))) == ["another_metric", "some_metric"]
    @test sort(collect(keys(c["dataset_b"]))) == ["dodo_metric", "great_metric"]

    # check the metrics concated properly
    @test c["dataset_a"]["some_metric"] == [1., 2., 3., 4.]
    @test c["dataset_a"]["another_metric"] == [2.2, 3.3, 4.4, 5.5]
    @test c["dataset_b"]["dodo_metric"] == [0., 0., 1., 1.]
    @test c["dataset_b"]["great_metric"] == [0.1, 0.2, 0.3, 0.4]


    d1 = Dict("breakme" => Dict{String, Vector{Float64}}())
    d2 = Dict("breakme2" => Dict{String, Vector{Float64}}())

    @test_throws ErrorException LightGBM.merge_scores(d1, d2)

    e1 = Dict("ametric" => [1., 1., 1.])
    e2 = Dict("bmetric" => [1., 1., 1.])

    @test_throws ErrorException LightGBM.merge_scores(e1, e2)

end


@testset "empty merge_scores" begin

    # basically this test is just to make sure it works properly even when metrics is empty

    a = Dict{String, Vector{Float64}}()
    b = Dict{String, Vector{Float64}}()

    c = LightGBM.merge_scores(a, b)

    @test typeof(c) == typeof(a)
    @test isempty(c)

    d1 = Dict{String, typeof(a)}()
    d2 = Dict{String, typeof(a)}()

    d3 = LightGBM.merge_scores(d1, d2)

    @test typeof(d3) == typeof(d1)
    @test isempty(d3)

end

end # module
