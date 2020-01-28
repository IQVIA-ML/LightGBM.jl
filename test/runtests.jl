
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

@testset "FFI" begin

    @testset "Dataset" begin
        include(joinpath("ffi", "datasets.jl"))l
    end

end
