module LightGBM

using Dates
import Base
import Libdl
import StatsBase
import Libdl


const LGBM_library = Ref{Ptr{Cvoid}}(C_NULL)


struct LibraryNotFoundError <: Exception
    msg::String
end


function find_library(library_name::String, custom_paths::Vector{String})

    # Search system filedirs first, returns empty string if not found
    libpath = Libdl.find_library(library_name)

    if libpath == ""
        # try specified paths
        @info("$(library_name) not found in system dirs, trying fallback")
        libpath = Libdl.find_library(library_name, custom_paths)
    else
        @info("$(library_name) found in system dirs!")
    end

    if libpath == ""
        throw(LibraryNotFoundError("$(library_name) not found. Please ensure this library is either in system dirs or the dedicated paths: $(custom_paths)"))
    end

    return libpath

end


function __init__()
    LGBM_library[] = Libdl.dlopen(find_library("lib_lightgbm", [@__DIR__]))
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
