module LightGBM

using Dates

import Libdl
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

function __init__()
    include(joinpath(@__DIR__, "MLJInterface.jl"))
end

export fit, predict, predict_classes, cv, search_cv, savemodel, loadmodel
export LGBMEstimator, LGBMRegression, LGBMClassification

end # module LightGBM
