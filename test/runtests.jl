
using Test


@testset "FFI unit tests" begin

    @testset "Dataset" begin
        include(joinpath("ffi", "datasets.jl"))
    end

    @testset "Booster" begin
        include(joinpath("ffi", "booster.jl"))
    end

    @testset "Base Utils" begin
        include(joinpath("ffi", "base_utils.jl"))
    end

end

@testset "MLJ interface tests" begin

    @testset "Binary LightGBM" begin
        include(joinpath("mlj", "binary_classifier.jl"))
    end

    @testset "Multiclass LightGBM" begin
        include(joinpath("mlj", "multiclass_classifier.jl"))
    end

    @testset "Regression LightGBM" begin
        include(joinpath("mlj", "regression.jl"))
    end

    @testset "MLJ update interface" begin
        include(joinpath("mlj", "update.jl"))
    end

end


@testset "Basic tests" begin

    @testset "Estimator parameters" begin
        include(joinpath("basic", "parameters.jl"))
    end

end

@testset "Integration tests" begin

    @testset "(OLD) Basic Tests" begin
        include("basic_tests.jl")
    end

    @testset "Weights Test" begin
        include("weightsTest.jl")
    end

    @testset "Init Score Test" begin
        include("initScoreTest.jl")
    end

end
