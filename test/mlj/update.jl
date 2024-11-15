module TestUpdateMethod


using MLJBase
using Test

import LightGBM

# Setup stuff
nrows = 3_000
nfeatures = 10

x = randn(nrows, nfeatures)
y = randn(nrows)
k = Dict(Symbol(i) => Int64(i) for i in 1:nfeatures)
X = MLJBase.Tables.MatrixTable(collect(keys(k)), k, x)
w = abs.(randn(nrows))
w = w / sum(w)

# setup MLJ machinery -- will be used in several testsets
r = LightGBM.MLJInterface.LGBMRegressor(num_iterations=1, min_data_in_leaf=20, verbosity = -1)
m = MLJBase.machine(r, X, y)

# initial fit
MLJBase.fit!(m; verbosity=0)


@testset "MLJ update" begin
    # machine keeps a field `fitresult` which is the the fitresult,
    # the first element of this tuple is the LGBMEstimator
    iteration_count = LightGBM.get_iter_number(first(m.fitresult))
    cached_iteration_count = sum(m.cache.num_boostings_done)
    @test iteration_count == 1
    @test iteration_count == cached_iteration_count
    @test m.cache.num_boostings_done == [1]

    # check an update chaining on from a fit works
    m.model.num_iterations += 10

    MLJBase.fit!(m; verbosity=0)

    iteration_count = LightGBM.get_iter_number(first(m.fitresult))
    cached_iteration_count = sum(m.cache.num_boostings_done)
    @test iteration_count == 11
    @test iteration_count == cached_iteration_count
    @test m.cache.num_boostings_done == [1, 10]
end


@testset "MLJ update chain" begin
    # check an update chaining on from an update works
    m.model.num_iterations += 5

    MLJBase.fit!(m; verbosity=0)

    iteration_count = LightGBM.get_iter_number(first(m.fitresult))
    cached_iteration_count = sum(m.cache.num_boostings_done)
    @test iteration_count == 16
    @test iteration_count == cached_iteration_count
    @test m.cache.num_boostings_done == [1, 10, 5]

end


@testset "MLJ update less iterations triggers refit" begin
    m.model.num_iterations -= 1
    MLJBase.fit!(m; verbosity=0)

    iteration_count = LightGBM.get_iter_number(first(m.fitresult))
    cached_iteration_count = sum(m.cache.num_boostings_done)
    @test iteration_count == 15
    @test iteration_count == cached_iteration_count
    @test m.cache.num_boostings_done == [15]
end


@testset "MLJ update non-updatable parameters" begin
    # next, check that  changing parameters other than num_iterations results in entirely refitting
    m.model.num_iterations += 1
    m.model.min_data_in_leaf += 5

    MLJBase.fit!(m; verbosity=0)

    iteration_count = LightGBM.get_iter_number(first(m.fitresult))
    cached_iteration_count = sum(m.cache.num_boostings_done)
    @test iteration_count == 16
    @test iteration_count == cached_iteration_count
    @test m.cache.num_boostings_done == [16]
end


@testset "MLJ User provided weights" begin
# check with user provided weights
    r = LightGBM.MLJInterface.LGBMRegressor(num_iterations=1, min_data_in_leaf=20, verbosity = -1)
    weights_machine = MLJBase.machine(r, X, y, w)

    MLJBase.fit!(weights_machine; verbosity=0)

    iteration_count = LightGBM.get_iter_number(first(weights_machine.fitresult))
    cached_iteration_count = sum(weights_machine.cache.num_boostings_done)
    @test iteration_count == 1
    @test iteration_count == cached_iteration_count
    @test weights_machine.cache.num_boostings_done == [1]

    weights_machine.model.num_iterations += 1
    MLJBase.fit!(weights_machine; verbosity=0)

    iteration_count = LightGBM.get_iter_number(first(weights_machine.fitresult))
    cached_iteration_count = sum(weights_machine.cache.num_boostings_done)
    @test iteration_count == 2
    @test iteration_count == cached_iteration_count
    @test weights_machine.cache.num_boostings_done == [1, 1]

end


end # module
