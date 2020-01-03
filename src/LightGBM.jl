__precompile__()

module LightGBM
using Libdl
using Dates
import StatsBase

const LGBM_library = abspath(find_library(["lib_lightgbm.$(Libdl.dlext)"], [@__DIR__]))

if LGBM_library == nothing
    # Lets get it to spit out why, by trying to directly dlopen the expected file
    ptr = dlopen(joinpath(@__DIR__, "lib_lightgbm.$(Libdl.dlext)"))
    # erk we shouldn't get here, close and throw
    dlclose(ptr)
    throw(LoadError("LightGBM.jl", 1, "find_library couldn't find the library but a direct dlopen also didn't throw, so kinda stumped and throwing for safety!"))
end

include("wrapper.jl")
include("estimators.jl")
include("utils.jl")
include("fit.jl")
include("predict.jl")
include("cv.jl")
include("search_cv.jl")
include("LightGBM-util2.jl")

export fit, predict, cv, search_cv, savemodel, loadmodel
export LGBMEstimator, LGBMRegression, LGBMBinary, LGBMMulticlass
export metaformattedclassresult, metaformattedclassresult, formattedclassfit, predict2

end # module LightGBM
