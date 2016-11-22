__precompile__()

module LightGBM

function __init__()
    if !haskey(ENV, "LIGHTGBM")
        warn("Environment variable LIGHTGBM not found.",
            " Set this variable to point to the LightGBM binary",
            " (e.g. `ENV[\"LIGHTGBM\"] = \"../lightgbm\"`).")
    end
    include("../deps/deps.jl")
end

include("wrapper.jl")
include("estimators.jl")
include("base.jl")
include("cli.jl")
include("fit.jl")

export fit, predict, LGBMEstimator, LGBMRegression, LGBMBinary, LGBMLambdaRank, LGBMMulticlass

end # module LightGBM
