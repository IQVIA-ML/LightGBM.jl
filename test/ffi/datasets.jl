module TestFFIDatasets

using LightGBM
using Test
using Random
using SparseArrays

# we don't want the LightGBM vom

verbosity = "verbose=-1"


@testset "LGBM_DatasetCreateFromMat -- floats/column major" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)

    @test created_dataset.handle != C_NULL
    @test LightGBM.LGBM_DatasetGetNumData(created_dataset) == 3
    @test LightGBM.LGBM_DatasetGetNumFeature(created_dataset) == 2

    @test LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b"]) == nothing
    # cause its not row major and has 2 cols
    @test_throws ErrorException LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b", "c"])

end


@testset "LGBM_DatasetCreateFromMat -- floats/row major" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity, true)

    @test created_dataset.handle != C_NULL
    @test LightGBM.LGBM_DatasetGetNumData(created_dataset) == 2
    @test LightGBM.LGBM_DatasetGetNumFeature(created_dataset) == 3

    # this kind of stuff is a bit circular but without a better way to interact with a dataset, its hard to test
    @test_throws ErrorException LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b"])
    # cause its row major with 2 initial cols and 3 rows
    @test LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b", "c"]) == nothing

end


@testset "LGBM_DatasetCreateFromMat -- ints/column major" begin

    mymat = [1 2; 3 4; 5 6]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)

    @test created_dataset.handle != C_NULL
    @test LightGBM.LGBM_DatasetGetNumData(created_dataset) == 3
    @test LightGBM.LGBM_DatasetGetNumFeature(created_dataset) == 2

    @test LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b"]) == nothing
    # cause its not row major and has 2 cols
    @test_throws ErrorException LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b", "c"])

end


@testset "LGBM_DatasetCreateFromMat -- ints/row major" begin

    mymat = [1 2; 3 4; 5 6]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity, true)

    @test created_dataset.handle != C_NULL
    @test LightGBM.LGBM_DatasetGetNumData(created_dataset) == 2
    @test LightGBM.LGBM_DatasetGetNumFeature(created_dataset) == 3

    @test_throws ErrorException LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b"])
    # cause its row major with 2 initial cols and 3 rows
    @test LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b", "c"]) == nothing

end


@testset "LGBM_DatasetCreateFromCSC" begin

    mymat = sparse([1. 2.; 3. 4.; 5. 6.])
    created_dataset = LightGBM.LGBM_DatasetCreateFromCSC(mymat, verbosity)

    @test created_dataset.handle != C_NULL
    @test LightGBM.LGBM_DatasetGetNumData(created_dataset) == 3
    @test LightGBM.LGBM_DatasetGetNumFeature(created_dataset) == 2

end

@testset "LGBM_DatasetGetSubset" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)

    get_idx = [1, 3] # check julia indices are correctly translated
    # gotta create the same so its not a reference, so we can check the mutation is correct (or at least it looks like it hasn't been mutated)
    get_idx_copy = [1, 3]
    faulty_idx = [1, 5]
    # this isn't a julia matrix -- its another dataset
    fetched = LightGBM.LGBM_DatasetGetSubset(created_dataset, Int32.(get_idx), verbosity)
    fetched2 = LightGBM.LGBM_DatasetGetSubset(created_dataset, get_idx, verbosity) # check int64s

    # check it isn't the same dataset...
    @test fetched.handle != created_dataset.handle
    @test get_idx == get_idx_copy

    @test_throws ErrorException LightGBM.LGBM_DatasetGetSubset(created_dataset, faulty_idx, verbosity)


end


@testset "LGBM_Dataset<Get|Set>FeatureNames" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)
    fieldnames = [randstring(22), randstring(50)]

    @test LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, fieldnames) == nothing
    @test LightGBM.LGBM_DatasetGetFeatureNames(created_dataset) == fieldnames

    # test string overlengths
    name_1 = randstring(3000)
    name_2 = randstring(3000)

    @test LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, [name_1, name_2]) == nothing
    retrieved = LightGBM.LGBM_DatasetGetFeatureNames(created_dataset)

    @test retrieved[1] == name_1#[1:256]
    @test retrieved[2] == name_2#[1:256]


end


@testset "LGBM_DatasetFree" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)

    # These tests exposed a double-free
    @test LightGBM.LGBM_DatasetFree(created_dataset) == nothing
    @test created_dataset.handle == C_NULL

end


@testset "LGBM_DatasetSetField" begin
    # The current implementation for LGBM_DatasetSetField has questionable type promotions
    # but it touches too much code in order to tidy up right now
    @test_broken false

end


@testset "LGBM_DatasetGetField" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)

    # pre-emptively use the right types in anticipation of typing for SetField being fixed in the future
    weights = [0.25f0, 0.5f0, 0.75f0] # Float32
    labels = [0.2f0, 0.5f0, -1.2f0] # Float32
    init_scores = [2.2, 0.1, 0.7] # Float64
    groups = Int32.([2, 1]) # Int32

    LightGBM.LGBM_DatasetSetField(created_dataset, "weight", weights)
    LightGBM.LGBM_DatasetSetField(created_dataset, "label", labels)
    LightGBM.LGBM_DatasetSetField(created_dataset, "group", groups)
    LightGBM.LGBM_DatasetSetField(created_dataset, "init_score", init_scores)

    @test LightGBM.LGBM_DatasetGetField(created_dataset, "weight") == weights
    @test isa(LightGBM.LGBM_DatasetGetField(created_dataset, "weight"), Vector{Float32})

    @test LightGBM.LGBM_DatasetGetField(created_dataset, "label") == labels
    @test isa(LightGBM.LGBM_DatasetGetField(created_dataset, "label"), Vector{Float32})

    @test LightGBM.LGBM_DatasetGetField(created_dataset, "init_score") == init_scores
    @test isa(LightGBM.LGBM_DatasetGetField(created_dataset, "init_score"), Vector{Float64})

    # I dont understand the discrepancy between input and output
    # It looks like LightGBM returns the cumsum of the group query
    # counts with 0 prepended but that isn't mentioned in docs
    @test_broken LightGBM.LGBM_DatasetGetField(created_dataset, "group") == groups
    @test isa(LightGBM.LGBM_DatasetGetField(created_dataset, "group"), Vector{Int32})

end


end # module
