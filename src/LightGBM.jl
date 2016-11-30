__precompile__()

module LightGBM

function __init__()
    if !haskey(ENV, "LIGHTGBM_PATH")
        error("Environment variable LIGHTGBM_PATH not found. ",
            "Set this variable to point to the LightGBM directory prior to loading LightGBM.jl ",
            "(e.g. `ENV[\"LIGHTGBM_PATH\"] = \"../LightGBM\"`).")
    else
        global const LGBM_library = Libdl.find_library(["lib_lightgbm.so", "lib_lightgbm.dll",
            "lib_lightgbm.dylib"], [ENV["LIGHTGBM_PATH"]])
        if LGBM_library == ""
            error("Could not open the LightGBM library at $(ENV["LIGHTGBM_PATH"]). ",
                  "Set this variable to point to the LightGBM directory prior to loading LightGBM.jl ",
                  "(e.g. `ENV[\"LIGHTGBM_PATH\"] = \"../LightGBM\"`).")
        end
    end
end

include("wrapper.jl")
include("estimators.jl")
include("utils.jl")
include("fit.jl")
include("predict.jl")
include("cv.jl")
include("search_cv.jl")

export fit, predict, cv, search_cv
export LGBMEstimator, LGBMRegression, LGBMBinary, LGBMLambdaRank, LGBMMulticlass

end # module LightGBM
