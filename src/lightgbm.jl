__precompile__()

module LightGBM

function __init__()
    if !haskey(ENV, "LIGHTGBM")
        warn("Environment variable LIGHTGBM not found.",
            " Set this variable to point to the LightGBM binary",
            " (e.g. `ENV[\"LIGHTGBM\"] = \"../lightgbm\"`).")
    end
end

include("estimators.jl")
include("base.jl")
include("cli.jl")

export fit, predict, LightGBMRegression, LightGBMBinary, LightGBMLambdaRank

end # module LightGBM