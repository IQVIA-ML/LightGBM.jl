module TestBaseUtils

using LightGBM
using Test

@testset "deepcopy: Booster" begin

    b = LightGBM.LGBM_BoosterCreateFromModelfile(joinpath(@__DIR__, "data", "test_tree"))
    bb = deepcopy(b)

    @test b != bb
    @test b !== bb
    @test LightGBM.LGBM_BoosterSaveModelToString(b, 0, 0) == LightGBM.LGBM_BoosterSaveModelToString(bb, 0, 0)

    # test for empty boosters too
    b_empty = LightGBM.Booster()
    bb_empty = deepcopy(b_empty)

    @test b_empty != bb_empty
    @test b_empty !== bb_empty

end

end # module
