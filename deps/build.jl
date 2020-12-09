
using Libdl

savepath = normpath(joinpath(@__DIR__, "..", "src", "lib_lightgbm.$(Libdl.dlext)"))

# These precompiled ones are CPU-only, can incorporate stuff later for user-built GPU capable binaries
GH_URL = "https://github.com/microsoft/LightGBM/releases/download/v3.1.0/lib_lightgbm.$(Libdl.dlext)"

download(GH_URL, savepath)

# The end
