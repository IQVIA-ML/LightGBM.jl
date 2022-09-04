module TestLightGBM

import Libdl
using LightGBM
using Test

src_dir = abspath(joinpath(@__DIR__, "..", "..", "src"))

# These set of tests use common libraries of each system to test `find_library` without having to modify env variables
function setup_env()

    output = Dict()
    output["sample_lib"] = ""

    if Sys.islinux()
        output["sample_lib"] = "libcrypt"
    elseif Sys.isunix()
        output["sample_lib"] = "libpython"
    elseif Sys.iswindows()
        output["sample_lib"] = "netmsg"
    end

    loaded = Libdl.find_library(output["sample_lib"])
    # If we can't load the expected sample library treat this as a broken system for the purpose of tests
    # it isn't important these tests pass on all systems,
    # because we had to be able to load a LightGBM library at all to run them
    # and sometimes they fail and cause user consternation because assumptions aren't satisfied
    broken_system = loaded == ""
    fullpath = Libdl.dlpath(loaded)

    if !broken_system
        # fullpath of a linkable lib to copy off the sys path
        output["linkable_path"] = fullpath
        # where to create a fixture library file (custom path) where such library exists in the syspath
        output["custom_fixture_path"] = joinpath(src_dir, "$(output["sample_lib"]).$(Libdl.dlext)")
        # where to create a fixture library file (custom path) where such library does NOT exist in the syspath
        output["lib_not_on_sys_fixture_path"] = joinpath(src_dir, "lib_not_on_sys.$(Libdl.dlext)")
        # move some files for the tests
        cp(output["linkable_path"], output["lib_not_on_sys_fixture_path"], force=true, follow_symlinks=true)
        cp(output["linkable_path"], output["custom_fixture_path"], force=true, follow_symlinks=true)
    end

    return output, broken_system

end

function teardown(settings::Dict)

    rm(settings["custom_fixture_path"], force=true)
    rm(settings["lib_not_on_sys_fixture_path"], force=true)

    return nothing

end

@testset "find_library" begin

    # Arrange -- once only because it isn't necessary to repeat this over and over
    settings, broken_system = setup_env()

    @testset "find_library works with no system lib" begin
        # Act
        output = LightGBM.find_library("lib_not_on_sys", [src_dir])

        # Assert
        if !broken_system
            @test output == joinpath(src_dir, "lib_not_on_sys") # custom path detected (without extension)
        else
            @test_broken output == joinpath(src_dir, "lib_not_on_sys")
        end
    end

    @testset "find_library finds system lib before fallback" begin
        # Act
        output = LightGBM.find_library(settings["sample_lib"], [src_dir])

        # Assert
        if !broken_system
            @test output == settings["sample_lib"] # sys lib detected
        else
            @test_broken output == settings["sample_lib"]
        end
    end

    @testset "find_library returns empty and logs error" begin
        # Act and assert
        @test_throws LightGBM.LibraryNotFoundError LightGBM.find_library("lib_that_simply_doesnt_exist", [src_dir])
    end

    teardown(settings)

end

end # module
