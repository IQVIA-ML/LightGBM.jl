__precompile__()

module LightGBM
using Libdl
using Dates
import StatsBase

# we build it so we can assert it is present...
const LGBM_library = abspath(joinpath(@__DIR__, "lib_lightgbm.$(Libdl.dlext)"))

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
