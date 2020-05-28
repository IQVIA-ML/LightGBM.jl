module TestUpdateMethod


using MLJBase
using Test

import LightGBM

# First, generate a large-ish garbage dataset
nrows = 20_000
nfeatures = 200

x = randn(nrows, nfeatures)
y = randn(nrows)
k = Dict(Symbol(i) => Int64(i) for i in 1:nfeatures)
X = MLJBase.Tables.MatrixTable(collect(keys(k)), k, x)

# setup MLJ machinery
r = LightGBM.MLJInterface.LGBMRegressor(num_iterations=1, min_data_in_leaf=20)
m = MLJBase.machine(r, X, y)

# lets fit it
MLJBase.fit!(m; verbosity=0)

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

# check an update chaining on from an update works
m.model.num_iterations += 5

MLJBase.fit!(m; verbosity=0)

iteration_count = LightGBM.get_iter_number(first(m.fitresult))
cached_iteration_count = sum(m.cache.num_boostings_done)
@test iteration_count == 16
@test iteration_count == cached_iteration_count
@test m.cache.num_boostings_done == [1, 10, 5]


# refit from scratch, num_iterations is 16 (via modifications)
# and MLJ should call the fit interface internally -- demonstrate this
MLJBase.fit!(m; force=true, verbosity=0)

iteration_count = LightGBM.get_iter_number(first(m.fitresult))
cached_iteration_count = sum(m.cache.num_boostings_done)
@test iteration_count == 16
@test iteration_count == cached_iteration_count
@test m.cache.num_boostings_done == [16]


# next, check that  changing parameters other than num_iterations results in entirely refitting
m.model.num_iterations += 1
m.model.min_data_in_leaf += 5

MLJBase.fit!(m; verbosity=0)

iteration_count = LightGBM.get_iter_number(first(m.fitresult))
cached_iteration_count = sum(m.cache.num_boostings_done)
@test iteration_count == 17
@test iteration_count == cached_iteration_count
@test m.cache.num_boostings_done == [17]

end # module
true
