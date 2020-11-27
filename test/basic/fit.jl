module TestFit

include("../../src/LightGBM.jl") # remove later

using Test
using LightGBM

@testset "test fit with dataset as input type runs fine -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(objective = "binary", num_class = 1)

    train_matrix = rand(100,70) # create random dataset
    train_labels = rand([0, 1], 100)
    train_dataset = LightGBM.LGBM_DatasetCreateFromMat(train_matrix, "")
    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)

    test_matrix = rand(50,70) # create random dataset
    test_labels = rand([0, 1], 50)
    test_dataset = LightGBM.LGBM_DatasetCreateFromMat(test_matrix, "", train_dataset)
    LightGBM.LGBM_DatasetSetField(test_dataset, "label", test_labels)
    
    # Act
    LightGBM.fit!(estimator, train_dataset, test_dataset, verbosity = -1);
    p_binary = LightGBM.predict(estimator, train_matrix, verbosity = -1)
    binary_classes = LightGBM.predict_classes(estimator, train_matrix, verbosity = -1)

    # Assert
    # check also that binary outputs and classes are invariant in prediction output shapes, i.e. they're matrices
    @test length(size(p_binary)) == 2
    @test length(size(binary_classes)) == 2
    
end

@testset "test fit with dataset as input type runs fine without test set -- binary" begin
    # Arrange
    estimator = LightGBM.LGBMClassification(objective = "binary", num_class = 1)

    train_matrix = rand(100,70) # create random dataset
    train_labels = rand([0, 1], 100)
    train_dataset = LightGBM.LGBM_DatasetCreateFromMat(train_matrix, "")
    LightGBM.LGBM_DatasetSetField(train_dataset, "label", train_labels)

    # Act
    LightGBM.fit!(estimator, train_dataset, verbosity = -1);
    p_binary = LightGBM.predict(estimator, train_matrix, verbosity = -1)
    binary_classes = LightGBM.predict_classes(estimator, train_matrix, verbosity = -1)

    # Assert
    # check also that binary outputs and classes are invariant in prediction output shapes, i.e. they're matrices
    @test length(size(p_binary)) == 2
    @test length(size(binary_classes)) == 2
    
end

end # module