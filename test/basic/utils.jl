using LightGBM

@testset "stringifyparams -- convert to zero-based" begin
    indices = [1, 3, 5, 7, 9]
    classifier = LightGBM.LGBMClassification(categorical_feature = indices)
    ds_parameters = LightGBM.stringifyparams(classifier, LightGBM.DATASETPARAMS)

    expected = "categorical_feature=0,2,4,6,8"
    @test occursin(expected, ds_parameters)
end