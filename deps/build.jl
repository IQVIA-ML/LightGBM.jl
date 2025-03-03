using Libdl

if (VERSION ∈ (v"1.8.4", v"1.8.5")) && Sys.iswindows()
    # See https://github.com/JuliaLang/julia/issues/48187

    # LightGBM_jll UUID
    savepath = normpath(joinpath(@__DIR__, "..", "src", "lib_lightgbm.$(Libdl.dlext)"))
    GH_URL = "https://github.com/microsoft/LightGBM/releases/download/v3.3.5/lib_lightgbm.$(Libdl.dlext)"
    download(GH_URL, savepath)
end
