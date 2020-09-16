using Documenter, LightGBM

makedocs(;
    modules=[LightGBM],
    repo="https://github.com/IQVIA-ML/LightGBM.jl/blob/{commit}{path}#L{line}",
    sitename="LightGBM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://IQVIA-ML.github.io/LightGBM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Function" => "function.md",
    ],
)

deploydocs(;
    repo="github.com/IQVIA-ML/LightGBM.jl",
)
