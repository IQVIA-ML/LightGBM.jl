module TestMLJUserReport

using MLJBase
using Test

import LightGBM


# Setup stuff
nrows = 500
nfeatures = 10

x = randn(nrows, nfeatures)
y = randn(nrows)


@testset "MLJ user report structure" begin

    # by settoing metric freq to 2 we can check the frequency is occurring correctly,
    # first time we expect only 2 metrics and then (metric @iter0 and metric @iter2)
    # if we go for 3 more iterations we expect 1 more metric
    estimator = LightGBM.LGBMRegression(;num_iterations=3, is_training_metric=true, metric=["l2"], metric_freq=2)
    metrics = LightGBM.fit!(estimator, x, y; verbosity=-1)

    report = LightGBM.MLJInterface.user_fitreport(estimator, metrics)

    # check types (loosely)
    @test report isa NamedTuple{(:training_metrics, :importance)}
    @test report.training_metrics isa Dict
    @test report.importance isa NamedTuple{(:gain, :split)}
    # check value properties
    @test length(report.importance.gain) == nfeatures
    @test length(report.importance.split) == nfeatures
    @test length(report.training_metrics["training"]["l2"]) == 2

    # continue for another 3 iterations, and check that the metrics lengths update properly
    new_metrics = LightGBM.train!(estimator, 3, String[], -1, LightGBM.Dates.now())

    new_report = LightGBM.MLJInterface.user_fitreport(estimator, metrics, new_metrics)

    # repeat all same tests except training metrics is now 3, not 1
    @test new_report isa NamedTuple{(:training_metrics, :importance)}
    @test new_report.training_metrics isa Dict
    @test new_report.importance isa NamedTuple{(:gain, :split)}
    # check value properties
    @test length(new_report.importance.gain) == nfeatures
    @test length(new_report.importance.split) == nfeatures
    @test length(new_report.training_metrics["training"]["l2"]) == 3

end


end # module
