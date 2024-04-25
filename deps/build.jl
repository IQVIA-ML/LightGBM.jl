using Libdl
import Pkg
import UUIDs

if (VERSION ∈ (v"1.8.4", v"1.8.5")) && Sys.iswindows()
    # See https://github.com/JuliaLang/julia/issues/48187

    # LightGBM_jll UUID
    JLL_version = Pkg.dependencies()[UUID("0e4427ef-1ff7-5cd7-8faa-8ff0877bb2ec")].version
    version_string = "$(JLL_version.major).$(JLL_version.minor).$(JLL_version.patch)"
    savepath = normpath(joinpath(@__DIR__, "..", "src", "lib_lightgbm.$(Libdl.dlext)"))
    GH_URL = "https://github.com/microsoft/LightGBM/releases/download/v$version_string/lib_lightgbm.$(Libdl.dlext)"
    download(GH_URL, savepath)
end
