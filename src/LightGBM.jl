module LightGBM

using Dates
import Base
import Libdl
import SparseArrays
import Statistics

if !((VERSION ∈ (v"1.8.4", v"1.8.5")) && Sys.iswindows())
    # See https://github.com/JuliaLang/julia/issues/48187
    import LightGBM_jll
end

const LGBM_library = Ref{Ptr{Cvoid}}(C_NULL)
const LOG_LIBRARY_DISCOVERY = Base.get_bool_env("LIGHTGBM_LOG_LIBRARY_DISCOVERY", true)


struct LibraryNotFoundError <: Exception
    msg::String
end


function find_library(library_name::String, custom_paths::Vector{String})

    # Search system filedirs first, returns empty string if not found
    libpath = Libdl.find_library(library_name)

    if libpath == ""
        # try specified paths
        if LOG_LIBRARY_DISCOVERY
            @info("$(library_name) not found in system dirs, trying fallback")
        end
        libpath = Libdl.find_library(library_name, custom_paths)
    else
        if LOG_LIBRARY_DISCOVERY
            @info("$(library_name) found in system dirs!")
        end
    end

    if libpath == ""
        throw(LibraryNotFoundError("$(library_name) not found. Please ensure this library is either in system dirs or the dedicated paths: $(custom_paths)"))
    end

    return libpath

end


function __init__()
    LGBM_library[] = Libdl.dlopen(find_library("lib_lightgbm", [@__DIR__]))
    if (VERSION ∈ (v"1.8.4", v"1.8.5")) && Sys.iswindows()
        printstyled(stdout,
        """
        \nIncompatibility warning: LightGBM

        LightGBM_jll does not work correctly for julia v1.8.4-v1.8.5 in windows, precompiled libraries will be downloaded instead
        See https://github.com/JuliaLang/julia/issues/48187
        \n
        """; color = :light_magenta)
    end
    return nothing
end


include("wrapper.jl")
include("estimators.jl")
include("utils.jl")
include("fit.jl")
include("predict.jl")
include("refit.jl")
include("cv.jl")
include("search_cv.jl")
include(joinpath(@__DIR__, "MLJInterface.jl"))

export fit!, predict, predict_classes, cv, search_cv, savemodel, loadmodel!
export LGBMEstimator, LGBMRegression, LGBMClassification

end # module LightGBM
