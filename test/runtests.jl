
using Test


@testset "FFI unit tests" begin

    @testset "Dataset" begin
        include(joinpath("ffi", "datasets.jl"))
    end

end


@testset "Integration tests" begin

    @testset "Basic Tests" begin
        include("basic_tests.jl")
    end

    @testset "Weights Test" begin
        include("weightsTest.jl")
    end

    @testset "Init Score Test" begin
        include("initScoreTest.jl")
    end

end
