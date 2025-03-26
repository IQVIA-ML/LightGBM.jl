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


@testset "LGBM_DatasetCreateFromFile - ignore_column parse logs" begin
    # Create a small sample dataset
    header = "label,feature1,feature2,feature3,feature4,feature5,group_id\n"
    data_points = """
    1,0.1,0.2,0.3,0.4,0.5,1
    0,0.2,0.3,0.4,0.5,0.6,1
    1,0.3,0.4,0.5,0.6,0.7,2
    0,0.4,0.5,0.6,0.7,0.8,2
    1,0.5,0.6,0.7,0.8,0.9,3
    0,0.6,0.7,0.8,0.9,1.0,3
    """

    # Write the dataset to a file, repeating only the data points to exceed 40 rows
    # Due to [LightGBM] [Warning] There are no meaningful features, as all feature values are constant.
    # For datasets < 40 rows with a small number of features, LightGBM doesn't use any features for training
    # Despite them being informative for the model
    sample_file = "sample_data_large.csv"
    open(sample_file, "w") do f
        write(f, header)  # Write the header only once
        for _ in 1:10  # Repeat the data points 10 times (6 rows * 10 = 60 rows)
            write(f, data_points)
        end
    end

    # Define dataset parameters 
    dataset_params_list = [
        # Case 1 ignoring feature2 columns and group_id treated as feature (expected 5 features)
        "header=true ignore_column=name:feature2 verbosity=-1"
        # Case 2 ignoring feature2 columns and group_id treated as group (expected 4 features)
        "header=true ignore_column=name:feature2 query=name:group_id verbosity=-1" 
        # Case 3 ignoring all columns except the label column (expected 0 features)
        "header=true ignore_column=0,1,2,3,4,5 verbosity=-1"
        # Case 4 default parameters (expected 6 features, group_id treated as feature)
        "two_round=false header=true verbosity=-1"
    ]

    # Expected number of features from the logs
    expected_features = [5, 4, 0, 6]

    # Test the actual modelling features for both cases by parsing logs due to LGBM_DatasetGetNumFeature
    # or LGBM_BoosterGetFeatureNames getting the data points not the actual features used in training
    for (i, dataset_params) in enumerate(dataset_params_list)
        # Use a temporary file to capture logs (julia 1.6 doesn't support `Pipe` for `redirect_stdout`, later versions do)
        log_file = "temp_log.txt"
        open(log_file, "w") do log_stream
            redirect_stdout(log_stream) do
                # Create dataset
                dataset = LightGBM.LGBM_DatasetCreateFromFile(sample_file, dataset_params)

                # Create booster
                booster_params = "objective=binary metric=binary_logloss verbosity=2"
                booster = LightGBM.LGBM_BoosterCreate(dataset, booster_params)
            end
        end

        # Read the captured logs
        logs = read(log_file, String)
        rm(log_file)  # Clean up the temporary log file

        # Parse the captured logs
        match_result = Base.match(r"Number of data points in the train set: \d+, number of used features: (\d+)", logs)

        # Assert
        @test !isnothing(match_result)  # Ensure the line exists in the logs
        num_features = parse(Int, match_result.captures[1])

        @test num_features == expected_features[i]
    end

    # Clean up
    rm(sample_file)
end


@testset "LGBM_DatasetCreateFromFile - other dataset params" begin
    # Create a sample .csv file with a header row
    sample_data = """
    label,feature2,feature3,feature4,group_id
    0.1,0.2,0.3,0.4,1
    0.6,0.7,0.8,0.9,2
    1.1,1.2,1.3,1.4,2
    """
    sample_file = "sample_data.csv"
    open(sample_file, "w") do f
        write(f, sample_data)
    end

    # Combinations of parameters that should fail
    params_fail = [
        # This should throw an error as the column to ignore does not exist in the file
        "header=true ignore_column=name:any_column verbosity=-1",
        # This should throw an error as there is a header in the file so the header should be set to true
        "two_round=true header=false verbosity=-1", 
        # This should throw an error as the query parameter which is called `group_column` in docs is not a valid name
        "header=true query=name:some_column verbosity=-1",
    ]
    for param in params_fail
        @test_throws ErrorException LightGBM.LGBM_DatasetCreateFromFile(sample_file, param)
    end

    # Query/group parameter is used to test the group column functionality
    # However, the actual parameter name is `group_column` as per the documentation and `query` and `group` are aliases
    # But the C++ LGBM_DatasetCreateFromFile accepts `query` or `group` as a valid parameter name and not `group_column`
    # Which can be tricky when passing this parameter directly from estimator.group_column
    params_group_column = "header=true query=name:group_id verbosity=-1"
    dataset_group_info = LightGBM.LGBM_DatasetCreateFromFile(sample_file, params_group_column)
    @test dataset_group_info != C_NULL
    # Check the group column: `LGBM_DatasetGetField` returns the expected group boundaries/indices of the groups
    # In the sample dataset there are only 2 groups: [1, 2] where the first row belongs to group 1 and two next rows to group 2
    # The output [0, 1, 3] means that Group 1 starts at index 0 and ends at index 1 and Group 2 starts at index 1 and ends at index 3
    @test LightGBM.LGBM_DatasetGetField(dataset_group_info, "group") == [0, 1, 3]
    # It also works with the `query` parameter (and the actual parameter name as per documentation is `group_column` 
    # so both `query` and `group` are not quite correct and they're considered aliases)
    @test LightGBM.LGBM_DatasetGetField(dataset_group_info, "query") == [0, 1, 3]

    # Test label column (label is by default the first column but making it explicit)
    dataset = LightGBM.LGBM_DatasetCreateFromFile(sample_file, "header=true label_column=name:label verbosity=-1")
    label_col = LightGBM.LGBM_DatasetGetField(dataset, "label")
    @test label_col == Float32[0.1, 0.6, 1.1]

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
