module TestLightGBM

import Libdl
using LightGBM
using Test

src_dir = abspath(joinpath(@__DIR__, "..", "..", "src"))

# little helper to get base libnames
libname(x) = first(splitext(basename(x)))

# These set of tests use common libraries of each system to test `find_library` without having to modify env variables
function setup_env()

    # get libs that are definitely present and linkable for tests (with just system extension)
    # by virtue of being loaded
    loaded = Libdl.dllist()
    # get all the ones which just have the system extension and no funny business
    withext = loaded[endswith.(loaded, Ref(Libdl.dlext))]
    # Some versions of LLVM are ... weird ... in that they don't like being double loaded. Use something else
    withext = withext[.!occursin.(Ref("LLVM"), withext)]
    # get the libname without prefix and extension
    withoutext = libname.(withext)
    # check which are loadable (or findable by find_library, this is WILD if they're not all found)
    libnames = Libdl.find_library.(withoutext)
    # get the first one matching or nothing
    libidx = findfirst(libnames .== withoutext)
    borked = libidx == nothing

    output = Dict()
    output["sample_lib"] = ""
    output["linkable_path"] = ""

    if !borked
        output["sample_lib"] = withoutext[libidx]
        # fullpath of a linkable lib to copy off the sys path
        output["linkable_path"] = withext[libidx]
    end

    # where to create a fixture library file (custom path) where such library exists in the syspath
    output["custom_fixture_path"] = joinpath(src_dir, "$(output["sample_lib"]).$(Libdl.dlext)")

    # where to create a fixture library file (custom path) where such library does NOT exist in the syspath
    output["lib_not_on_sys_fixture_path"] = joinpath(src_dir, "lib_not_on_sys.$(Libdl.dlext)")

    return output, borked

end

function teardown(settings::Dict)

    rm(settings["custom_fixture_path"], force=true)
    rm(settings["lib_not_on_sys_fixture_path"], force=true)

    return nothing

end

@testset "find_library" begin

    @testset "find_library works with no system lib" begin

        # Arrange
        settings, isborked = setup_env()
        if !isborked
            cp(settings["linkable_path"], settings["lib_not_on_sys_fixture_path"], force=true, follow_symlinks=true)
        end

        # Act
        output = LightGBM.find_library("lib_not_on_sys", [src_dir])

        # Assert
        if !isborked
            @test output == joinpath(src_dir, "lib_not_on_sys") # custom path detected (without extension)
        else
            @test_broken output == joinpath(src_dir, "lib_not_on_sys")
        end

        teardown(settings)
    end

    @testset "find_library finds system lib first" begin

        # Arrange
        settings, isborked = setup_env()
        if !isborked
            cp(settings["linkable_path"], settings["custom_fixture_path"], force=true, follow_symlinks=true)
        end

        # Act
        output = LightGBM.find_library(settings["sample_lib"], [src_dir])

        # Assert
        if !isborked
            @test output == settings["sample_lib"] # sys lib detected
        else
            @test_broken output == settings["sample_lib"]
        end

        teardown(settings)
    end

    @testset "find_library finds system lib" begin

        # Arrange
        settings, isborked = setup_env()

        # Act
        output = LightGBM.find_library(settings["sample_lib"], [src_dir]) # library should only exist in syspath, not custom path

        # Assert
        @test output == settings["sample_lib"] # sys lib detected

        teardown(settings)
    end

    @testset "find_library returns empty and logs error" begin

        # Arrange
        settings, _ = setup_env()

        # Act and assert
        @test_throws LightGBM.LibraryNotFoundError LightGBM.find_library("lib_that_simply_doesnt_exist", [src_dir])

        teardown(settings)

    end
end

end # module
