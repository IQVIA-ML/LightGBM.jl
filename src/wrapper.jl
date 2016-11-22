# TODO: Add option to specify library location.

type DatasetHandle
    handle::Ptr{Void}

    function DatasetHandle(handle::Ptr{Void})
        ds = new(handle)
        finalizer(ds, free)
        return ds
    end

    function free(ds::DatasetHandle)
        if ds.handle != C_NULL
            LGBM_DatasetFree(ds)
        end
    end
end

type BoosterHandle
    handle::Ptr{Void}

    function BoosterHandle(handle::Ptr{Void})
        bst = new(handle)
        finalizer(bst, free)
        return bst
    end

    function free(bst::BoosterHandle)
        if bst.handle != C_NULL
            LGBM_BoosterFree(bst)
        end
    end
end

function idtotype32(id::Integer)
    if id == 0
        return Float32
    elseif id == 1
        return Int32
    else
        error("unknown LightGBM return type id, got ", id)
    end
end

function typetoid32(datatype::Type)
    if datatype == Float32
        return 0
    elseif datatype == Int32
        return 1
    else
        error("unsupported datatype, got ", datatype)
    end
end

function typetoid64(datatype::Type)
    if datatype == Float32
        return 0
    elseif datatype == Float64
        return 1
    elseif datatype == Int32
        return 2
    elseif datatype == Int64
        return 3
    else
        error("unsupported datatype, got ", datatype)
    end
end

macro lightgbm(f, params...)
    temp_libpath = "/Users/Allard/GitHub/LightGBM/lib_lightgbm.so"
    args = [param.args[1] for param in params]
    types = [param.args[2] for param in params]

    return quote
        err = ccall(($f, $temp_libpath), Cint, ($(types...),), $(args...))
        if err != 0
            msg = unsafe_string(ccall((:LGBM_GetLastError, $temp_libpath), Cstring,()))
            error("call to LightGBM's ", string($f), " failed: ", msg)
        end
    end
end

# function LGBM_CreateDatasetFromFile()
# function LGBM_CreateDatasetFromBinaryFile()
# function LGBM_CreateDatasetFromCSR()
# function LGBM_CreateDatasetFromCSC()

function LGBM_CreateDatasetFromMat{T<:Real, R<:Union{DatasetHandle,Ptr{Void}}}(data::Array{T,2},
    parameters::String, reference::R = C_NULL)
    lgbm_data_type = typetoid64(T)
    nrow, ncol = size(data)
    is_row_major = 0
    out = Ref{DatasetHandle}(DatasetHandle(C_NULL))
    @lightgbm(:LGBM_CreateDatasetFromMat,
            data => Ref{T},
            lgbm_data_type => Cint,
            nrow => Int32,
            ncol => Int32,
            is_row_major => Cint,
            parameters => Cstring,
            reference => ifelse(R == DatasetHandle, Ref{DatasetHandle}, Ptr{Void}),
            out => Ref{DatasetHandle})
    return out[]
end

function LGBM_DatasetFree(handle::DatasetHandle)
    @lightgbm(:LGBM_DatasetFree,
              handle => DatasetHandle)
    return nothing
end

function LGBM_DatasetSaveBinary(handle::DatasetHandle, filename::String)
    @lightgbm(:LGBM_DatasetSaveBinary,
              handle => DatasetHandle,
              filename => Cstring)
    return nothing
end

function LGBM_DatasetSetField{T<:Real}(handle::DatasetHandle, field_name::String,
                                       field_data::Array{T,1})
    data_type = typetoid32(T)
    num_element = length(field_data)
    @lightgbm(:LGBM_DatasetSetField,
              handle => DatasetHandle,
              field_name => Cstring,
              field_data => Ref{T},
              num_element => Int64,
              data_type => Cint)
    return nothing
end

function LGBM_DatasetGetField(handle::DatasetHandle, field_name::String)
    out_len = Ref{Int64}()
    out_ptr = Ref{Ptr{Void}}()
    out_type = Ref{Cint}()
    @lightgbm(:LGBM_DatasetGetField,
              handle => DatasetHandle,
              field_name => Cstring,
              out_len => Ref{Int64},
              out_ptr => Ref{Ptr{Void}},
              out_type => Ref{Cint})
    jl_out_type = idtotype32(out_type[])
    jl_out_ptr = convert(Ptr{jl_out_type}, out_ptr[])
    return unsafe_wrap(Array{jl_out_type,1}, jl_out_ptr, out_len[], false)
end

function LGBM_DatasetGetNumData(handle::DatasetHandle)
    out = Ref{Int64}()
    @lightgbm(:LGBM_DatasetGetNumData,
              handle => DatasetHandle,
              out => Ref{Int64})
    return out[]
end

function LGBM_DatasetGetNumFeature(handle::DatasetHandle)
    out = Ref{Int64}()
    @lightgbm(:LGBM_DatasetGetNumFeature,
              handle => DatasetHandle,
              out => Ref{Int64})
    return out[]
end

function LGBM_BoosterCreate(train_data::DatasetHandle, valid_datas::Array{DatasetHandle,1},
                            valid_names::Array{String,1}, parameters::String)
    out = Ref{BoosterHandle}(BoosterHandle(C_NULL))
    n_valid_datas = length(valid_datas)
    @lightgbm(:LGBM_BoosterCreate,
              train_data => DatasetHandle,
              valid_datas => Ref{DatasetHandle},
              valid_names => Ptr{Cstring},
              n_valid_datas => Cint,
              parameters => Cstring,
              out => Ref{BoosterHandle})
    return out[]
end

# function LGBM_BoosterLoadFromModelfile()

function LGBM_BoosterFree(handle::BoosterHandle)
    @lightgbm(:LGBM_BoosterFree,
              handle => BoosterHandle)
    return nothing
end

function LGBM_BoosterUpdateOneIter(handle::BoosterHandle)
    is_finished = Ref{Cint}()
    @lightgbm(:LGBM_BoosterUpdateOneIter,
              handle => BoosterHandle,
              is_finished => Ref{Cint})
    return is_finished[]
end

# function LGBM_BoosterUpdateOneIterCustom()

function LGBM_BoosterEval(handle::BoosterHandle, data::Integer, n_metrics::Integer)
    out_results = Array(Cfloat, n_metrics)
    out_len = Ref{Int64}()
    @lightgbm(:LGBM_BoosterEval,
              handle => BoosterHandle,
              data => Cint,
              out_len => Ref{Int64},
              out_results => Ref{Cfloat})
    return out_results # TODO: check what out_len is supposed to be used for.
end

function LGBM_BoosterGetScore(handle::BoosterHandle)
    out_len = Ref{Int64}()
    out_results = Ref{Ptr{Cfloat}}()
    @lightgbm(:LGBM_BoosterGetScore,
              handle => BoosterHandle,
              out_len => Ref{Int64},
              out_results => Ref{Ptr{Cfloat}})
    return unsafe_wrap(Array{Float32,1}, out_results[], out_len[])
end

function LGBM_BoosterGetPredict(handle::BoosterHandle, data::Integer, n_data::Integer)
    out_len = Ref{Int64}()
    out_results = Array(Cfloat, n_data)
    @lightgbm(:LGBM_BoosterGetPredict,
              handle => BoosterHandle,
              data => Cint,
              out_len => Ref{Int64},
              out_results => Ref{Cfloat})
    return out_results # TODO: check what out_len is supposed to be used for.
end

# function LGBM_BoosterPredictForFile()
# function LGBM_BoosterPredictForCSR()

function LGBM_BoosterPredictForMat{T<:Real}(handle::BoosterHandle, data::Array{T,2},
                                            predict_type::Integer, n_used_trees::Integer)
    lgbm_data_type = typetoid64(T)
    nrow, ncol = size(data)
    is_row_major = 0
    out_result = Array(Cdouble, nrow)
    @lightgbm(:LGBM_BoosterPredictForMat,
              handle => BoosterHandle,
              data => Ref{T},
              lgbm_data_type => Cint,
              nrow => Int32,
              ncol => Int32,
              is_row_major => Cint,
              predict_type => Cint,
              n_used_trees => Int64,
              out_result => Ref{Cdouble})
    return out_result # TODO: check what out_len is supposed to be used for.
end

# function LGBM_BoosterSaveModel()
