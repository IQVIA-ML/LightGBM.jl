
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

    # Skippable for the CI purposes (and they can be slow)
    # Specifically for CI older julia versions are a problem to get test deps
    # to resolve properly. So we run a seperate set of tests with MLJ enabled for supported versions
    if !("DISABLE_MLJ_TESTS" in keys(ENV))

        @testset "Parameters" begin
            include(joinpath("mlj", "parameters.jl"))
        end

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

        @testset "MLJ user report" begin
            include(joinpath("mlj", "user_report.jl"))
        end

    end

end


@testset "Basic tests" begin

    @testset "Estimator parameters" begin
        include(joinpath("basic", "test_parameters.jl"))
    end

    @testset "Estimator parameters" begin
        include(joinpath("basic", "test_evaluation_metrics.jl"))
    end

    @testset "Utils" begin
        include(joinpath("basic", "test_utils.jl"))
    end

    @testset "Fit" begin
        include(joinpath("basic", "test_fit.jl"))
    end

    @testset "CV" begin
        include(joinpath("basic", "test_cv.jl"))
    end

    @testset "Search CV" begin
        include(joinpath("basic", "test_search_cv.jl"))
    end

    @testset "LightGBM" begin
        include(joinpath("basic", "test_lightgbm.jl"))
    end

end

@testset "Integration tests" begin

    @testset "(OLD) Basic Tests" begin
        include("integration/basic_tests.jl")
    end

    @testset "Weights Test" begin
        include("integration/weightsTest.jl")
    end

    @testset "Init Score Test" begin
        include("integration/initScoreTest.jl")
    end

    @testset "Group Query Test" begin
        include("integration/groupQueryTest.jl")
    end

end
