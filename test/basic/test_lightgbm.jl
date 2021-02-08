module TestLightGBM

import Libdl
using LightGBM
using Test

src_dir = abspath(joinpath(@__DIR__, "..", "..", "src"))

# These set of tests use common libraries of each system to test `find_library` without having to modify env variables
function setup_env()
    output = Dict()

    if Sys.islinux()
        output["sample_lib"] = "libcrypt"
    elseif Sys.isunix()
        output["sample_lib"] = "libevent"
    elseif Sys.iswindows()
        output["sample_lib"] = "netmsg"
    end

    output["ref_path"] = joinpath(src_dir, "lib_lightgbm.$(Libdl.dlext)")
    output["custom_fixture_path"] = joinpath(src_dir, "$(output["sample_lib"]).$(Libdl.dlext)")
    output["lib_not_on_sys_fixture_path"] = joinpath(src_dir, "lib_not_on_sys.$(Libdl.dlext)")

    return output

end

function teardown(settings::Dict)

    rm(settings["custom_fixture_path"], force=true)
    rm(settings["lib_not_on_sys_fixture_path"], force=true)

    return nothing

end

@testset "find_library" begin

    @testset "find_library works with no system lib" begin

        # Arrange
        settings = setup_env()
        cp(settings["ref_path"], settings["lib_not_on_sys_fixture_path"]) # fake file copied from lightgbm

        # Act
        output = LightGBM.find_library("lib_not_on_sys", [src_dir])

        # Assert
        @test output == joinpath(src_dir, "lib_not_on_sys") # custom path detected (without extension)

        teardown(settings)
    end

    @testset "find_library finds system lib first" begin

        # Arrange
        settings = setup_env()
        cp(settings["ref_path"], settings["custom_fixture_path"]) # fake file copied from lightgbm

        # Act
        output = LightGBM.find_library(settings["sample_lib"], [src_dir])

        # Assert
        @test output == settings["sample_lib"] # sys lib detected

        teardown(settings)
    end

    @testset "find_library finds system lib" begin

        # Arrange
        settings = setup_env()

        # Act
        output = LightGBM.find_library(settings["sample_lib"], [src_dir]) # sample_lib should only exist in syspath

        # Assert
        @test output == settings["sample_lib"]

        teardown(settings)
    end

    @testset "find_library throws error if cannot find lib" begin

        # Arrange
        settings = setup_env()

        # Act and assert
        @test_throws LightGBM.LibraryNotFoundError LightGBM.find_library("lib_that_simply_doesnt_exist", [src_dir])

        teardown(settings)

    end
end

end # module
