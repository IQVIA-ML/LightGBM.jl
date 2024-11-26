module TestUtils


using MLJBase
using Test

import LightGBM


@testset "mlj_to_kwargs removes classifier truncate_booster flag" begin
    # Arrange
    fixture = LightGBM.MLJInterface.LGBMClassifier(verbosity = -1)

    # Act
    output = LightGBM.MLJInterface.mlj_to_kwargs(fixture)

    # Assert
    @test :truncate_booster ∉ keys(output)
end

@testset "mlj_to_kwargs removes regressor truncate_booster flag" begin
    # Arrange
    fixture = LightGBM.MLJInterface.LGBMRegressor(verbosity = -1)

    # Act
    output = LightGBM.MLJInterface.mlj_to_kwargs(fixture)

    # Assert
    @test :truncate_booster ∉ keys(output)
end

@testset "mlj_to_kwargs adds classifier num_class" begin
    # Arrange
    fixture = LightGBM.MLJInterface.LGBMClassifier(verbosity = -1)

    # Act
    output = LightGBM.MLJInterface.mlj_to_kwargs(fixture, [0,1])

    # Assert
    @test :num_class in keys(output)
end


end # Module
