module TestFFIDatasets

using LightGBM
using Test

# we don't want the LightGBM vom
# redirect_stdout()

@testset "LGBM_DatasetCreateFromMat -- floats/column major" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, "min_data_in_leaf=1")

    @test created_dataset.handle != C_NULL

    @test LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b"]) == nothing
    # cause its not row major and has 2 cols
    @test_throws ErrorException LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b", "c"])

end

@testset "LGBM_DatasetCreateFromMat -- floats/row major" begin

    mymat = [1. 2.; 3. 4.; 5. 6.]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, "min_data_in_leaf=1", true)

    @test created_dataset.handle != C_NULL

    @test_throws ErrorException LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b"])
    # cause its row major with 2 initial cols and 3 rows
    @test LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b", "c"]) == nothing

end

@testset "LGBM_DatasetCreateFromMat -- ints/column major" begin

    mymat = [1 2; 3 4; 5 6]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, "min_data_in_leaf=1")

    @test created_dataset.handle != C_NULL

    @test LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b"]) == nothing
    # cause its not row major and has 2 cols
    @test_throws ErrorException LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b", "c"])

end

@testset "LGBM_DatasetCreateFromMat -- ints/row major" begin

    mymat = [1 2; 3 4; 5 6]
    created_dataset = LightGBM.LGBM_DatasetCreateFromMat(mymat, "min_data_in_leaf=1", true)

    @test created_dataset.handle != C_NULL

    @test_throws ErrorException LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b"])
    # cause its row major with 2 initial cols and 3 rows
    @test LightGBM.LGBM_DatasetSetFeatureNames(created_dataset, ["a", "b", "c"]) == nothing

end


end # module
