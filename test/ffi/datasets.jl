module TestFFIDatasets

using LightGBM
using Test
using Random
using SparseArrays


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


@testset "LGBM_DatasetCreateFromFile" begin
    # Create a sample .csv file with a header row
    sample_data = """
    feature1,feature2,feature3,feature4,feature5
    0.1,0.2,0.3,0.4,0.5
    0.6,0.7,0.8,0.9,1.0
    1.1,1.2,1.3,1.4,1.5
    """
    sample_file = "sample_data.csv"
    open(sample_file, "w") do f
        write(f, sample_data)
    end

    # Define dataset parameters as strings
    params = [
        # this shows a bug or un undocumented behaviour in the C API as there aren't 10 columns/features in the file so it should throw an error
        # or the ignore_column parameter is not considered at all
        "header=true ignore_column=0,1,2,3,4,5,6,7,8,9,10",
        # this also shows an unexpected behaviour as it should return 3 rows (num_data) 3 columns (2 num_features + label_column)
        # as the feature2 and feature3 columns are ignored but it doesn't
        "header=true ignore_column=name:feature2, feature3",
        # given the above here the ignore_column seems to be somehow working as an error is thrown: "Could not find ignore column feature1 in data file"
        "header=true ignore_column=name:feature1",
        # expected to have 3 rows (num_data) and 4 columns (3 num_features + label_column) as default label_column is 0 so it's not used as a feature
        "two_round=false header=true",
        # this should cause an error as there is a header in the file so the header should be set to true
        "two_round=true header=false"  
    ]

    expected_num_data = [3, 3, 3, 3, 3]
    expected_num_feature = [4, 4, 4, 4, 4]

    for (i, param) in enumerate(params)
        if i == 5 || i == 3
            @test_throws ErrorException LightGBM.LGBM_DatasetCreateFromFile(sample_file, param)
        else
            # Create dataset from file
            dataset = LightGBM.LGBM_DatasetCreateFromFile(sample_file, param)

            # Check if dataset is created successfully
            @test dataset != C_NULL

            # Check the number of rows and columns
            num_data = LightGBM.LGBM_DatasetGetNumData(dataset)
            num_feature = LightGBM.LGBM_DatasetGetNumFeature(dataset)
            @test num_data == expected_num_data[i]
            @test num_feature == expected_num_feature[i]
        end
    end

    # Clean up
    rm(sample_file)
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
    mymat = [1. 2.; 3. 4.; 5. 6.]
    ds = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)

    @testset "_LGBM_DatasetSetField invalid field name test" begin
        # "test_field" is not a valid field name
        err = @test_throws(ErrorException, LightGBM._LGBM_DatasetSetField(ds, "test_field", [1.0, 2.0, 3.0]))
        @test err.value.msg == "call to LightGBM's LGBM_DatasetSetField failed: Input data type error or field not found"
    end

    @testset "with input Vector{Int64} returning required Vector{Float32}" begin
        LightGBM.LGBM_DatasetSetField(ds, "label", [1, 2, 3])
        @test LightGBM.LGBM_DatasetGetField(ds, "label") == Float32[1.0, 2.0, 3.0]
    end

    @testset "with intput Vector{Float64} returning required Vector{Float32}" begin
        LightGBM.LGBM_DatasetSetField(ds, "label", [1.0, 2.0, 3.0])
        @test LightGBM.LGBM_DatasetGetField(ds, "label") == Float32[1.0, 2.0, 3.0]
    end

    @testset "with Vector{Float64} returning Int32 cumulative sum" begin
        LightGBM.LGBM_DatasetSetField(ds, "group", [2.0, 1.0])
        # `LGBM_DatasetGetField` returns the cumulative sum of group sizes
        @test LightGBM.LGBM_DatasetGetField(ds, "group") == Int32[0, 2, 3]
    end

    @testset "with input Vector{Float32} returning Vector{Float64}" begin
        LightGBM.LGBM_DatasetSetField(ds, "init_score", [1.f0, 2.f0, 3.f0])
        @test LightGBM.LGBM_DatasetGetField(ds, "init_score") == Float64[1.0, 2.0, 3.0]
    end

    @testset "with input Vector{Int64} returning Vector{Float64}" begin
        LightGBM.LGBM_DatasetSetField(ds, "init_score", [1, 2, 3])
        @test LightGBM.LGBM_DatasetGetField(ds, "init_score") == Float64[1.0, 2.0, 3.0]
    end

end


@testset "LGBM_DatasetGetField" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, verbosity)

    # pre-emptively use the right types in anticipation of typing for SetField being fixed in the future
    weights = [0.25f0, 0.5f0, 0.75f0] # Float32
    labels = [0.2f0, 0.5f0, -1.2f0] # Float32
    init_scores = [2.2, 0.1, 0.7] # Float64
    groups = Int32.([2, 1]) # Int32
    cumsumgroups = Int32.([0, 2, 3]) # Int32

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

    # In the C API, the function `LGBM_DatasetGetField` returns the cumulative sum of group sizes,
    # while `LGBM_DatasetSetField` expects the actual group sizes.
    # The "group" field is used internally by LightGBM for ranking tasks,
    # and the library uses a cumulative sum representation for efficiency.
    # In this tested example the group = [2, 1] means 2 rows  of created_dataset belong to the first group
    # and 1 to the second group with its cumulative sum representation [0, 2, 3]
    @test LightGBM.LGBM_DatasetGetField(created_dataset, "group") == cumsumgroups
    @test isa(LightGBM.LGBM_DatasetGetField(created_dataset, "group"), Vector{Int32})

end


end # module
