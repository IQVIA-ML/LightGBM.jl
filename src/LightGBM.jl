module LightGBM

using Dates
import Base
import Libdl
import SparseArrays
import Statistics



const LGBM_library = Ref{Ptr{Cvoid}}(C_NULL)


struct LibraryNotFoundError <: Exception
    msg::String
end


function find_library(library_names::Vector{String}, custom_paths::Vector{String})

    # If any of library_names is not an absolute path name, then the paths in the DL_LOAD_PATH and 
    # system load path are searched. Returns empty string if not found.
    libpath = Libdl.find_library(library_names)

    if libpath != ""
        @info("$(library_names) found in `DL_LOAD_PATH`, or system library paths $(ENV["PATH"])!")
    else
        # try specified paths
        @info("$(library_names) not found in `DL_LOAD_PATH`, or system library paths, trying fallback")
        libpath = Libdl.find_library(library_names, custom_paths)
    end

    if libpath != ""
        @info("$(library_names) found in $(custom_paths)")
    else
        throw(LibraryNotFoundError("$(library_names) not found. Please check this library using " *
        "Libdl.dlopen(l; throw_error=true) where l = joinpath(custom_paths, lib)"))
    end

    return libpath

end


function __init__()
    location = [@__DIR__]
    LGBM_library[] = Libdl.dlopen(find_library(["lib_lightgbm"], location))
    return nothing
end


include("wrapper.jl")
include("estimators.jl")
include("utils.jl")
include("fit.jl")
include("predict.jl")
include("cv.jl")
include("search_cv.jl")
include(joinpath(@__DIR__, "MLJInterface.jl"))

export fit!, predict, predict_classes, cv, search_cv, savemodel, loadmodel!
export LGBMEstimator, LGBMRegression, LGBMClassification

end # module LightGBM
