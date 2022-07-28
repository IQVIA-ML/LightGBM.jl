
using Libdl

current_file_path = @__DIR__
savepath = normpath(joinpath(current_file_path, "..", "src"))

if !isdir(savepath)
    mkdir(savepath)
end

cd(savepath)


# These precompiled ones are CPU-only, can incorporate stuff later for user-built GPU capable binaries
GH_URL = "https://github.com/microsoft/LightGBM/releases/download/v3.3.2/lib_lightgbm.$(Libdl.dlext)"

download(GH_URL, "lib_lightgbm.$(Libdl.dlext)")
cd(current_file_path)
# The end
