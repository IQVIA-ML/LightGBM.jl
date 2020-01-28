using LightGBM
using Test
using DelimitedFiles


@testset "LightGBM.jl" begin
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

