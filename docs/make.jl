using Documenter, LightGBM

makedocs(;
    modules=[LightGBM],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/wakakusa/LightGBM.jl/blob/{commit}{path}#L{line}",
    sitename="LightGBM.jl",
    authors="Allardvm",
    assets=String[],
)

deploydocs(;
    repo="github.com/wakakusa/LightGBM.jl",
)
