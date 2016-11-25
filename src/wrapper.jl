# TODO: Add option to specify library location.
typealias DatasetHandle Ptr{Void}
type Dataset
    handle::DatasetHandle

    function Dataset(handle::DatasetHandle)
        ds = new(handle)
        finalizer(ds, Dataset_finalizer)
        return ds
    end

    function Dataset_finalizer(ds::Dataset)
        if ds.handle != C_NULL
            LGBM_DatasetFree(ds)
            ds.handle = C_NULL
        end
    end
end

typealias BoosterHandle Ptr{Void}
type Booster
    handle::BoosterHandle

    function Booster(handle::BoosterHandle)
        bst = new(handle)
        finalizer(bst, Booster_finalizer)
        return bst
    end

    function Booster_finalizer(bst::Booster)
        if bst.handle != C_NULL
            LGBM_BoosterFree(bst)
            bst.handle = C_NULL
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
    args = [param.args[1] for param in params]
    types = [param.args[2] for param in params]

    return quote
        err = ccall(($f, LGBM_library), Cint, ($(types...),), $(args...))
        if err != 0
            msg = unsafe_string(ccall((:LGBM_GetLastError, LGBM_library), Cstring,()))
            error("call to LightGBM's ", string($f), " failed: ", msg)
        end
    end
end

# function LGBM_CreateDatasetFromFile()
# function LGBM_CreateDatasetFromBinaryFile()
# function LGBM_CreateDatasetFromCSR()
# function LGBM_CreateDatasetFromCSC()

function LGBM_CreateDatasetFromMat{T<:Union{Float32,Float64}}(data::Matrix{T}, parameters::String)
    lgbm_data_type = typetoid64(T)
    nrow, ncol = size(data)
    is_row_major = 0
    out = Ref{DatasetHandle}()
    @lightgbm(:LGBM_CreateDatasetFromMat,
              data => Ref{T},
              lgbm_data_type => Cint,
              nrow => Int32,
              ncol => Int32,
              is_row_major => Cint,
              parameters => Cstring,
              C_NULL => Ptr{Void},
              out => Ref{DatasetHandle})
    return Dataset(out[])
end

function LGBM_CreateDatasetFromMat{T<:Real}(data::Matrix{T}, parameters::String)
    return LGBM_CreateDatasetFromMat(convert(Matrix{Float64}, data), parameters)
end

function LGBM_CreateDatasetFromMat{T<:Union{Float32,Float64}}(data::Matrix{T}, parameters::String,
                                                              reference::Dataset)
    lgbm_data_type = typetoid64(T)
    nrow, ncol = size(data)
    is_row_major = 0
    out = Ref{DatasetHandle}()
    @lightgbm(:LGBM_CreateDatasetFromMat,
              data => Ref{T},
              lgbm_data_type => Cint,
              nrow => Int32,
              ncol => Int32,
              is_row_major => Cint,
              parameters => Cstring,
              reference.handle => Ref{DatasetHandle},
              out => Ref{DatasetHandle})
    return Dataset(out[])
end

function LGBM_CreateDatasetFromMat{T<:Real}(data::Matrix{T}, parameters::String,
                                            reference::Dataset)
    return LGBM_CreateDatasetFromMat(convert(Matrix{Float64}, data), parameters, reference)
end

function LGBM_DatasetFree(ds::Dataset)
    @lightgbm(:LGBM_DatasetFree,
              ds.handle => DatasetHandle)
    return nothing
end

function LGBM_DatasetSaveBinary(ds::Dataset, filename::String)
    @lightgbm(:LGBM_DatasetSaveBinary,
              ds.handle => DatasetHandle,
              filename => Cstring)
    return nothing
end

function LGBM_DatasetSetField{T<:Union{Float32,Int32}}(ds::Dataset, field_name::String,
                                                       field_data::Vector{T})
    data_type = typetoid32(T)
    num_element = length(field_data)
    @lightgbm(:LGBM_DatasetSetField,
              ds.handle => DatasetHandle,
              field_name => Cstring,
              field_data => Ref{T},
              num_element => Int64,
              data_type => Cint)
    return nothing
end

function LGBM_DatasetSetField{T<:Real}(ds::Dataset, field_name::String, field_data::Vector{T})
    return LGBM_DatasetSetField(ds, field_name, convert(Vector{Float32}, field_data))
end

function LGBM_DatasetSetField{T<:Integer}(ds::Dataset, field_name::String, field_data::Vector{T})
    return LGBM_DatasetSetField(ds, field_name, convert(Vector{Int32}, field_data))
end

function LGBM_DatasetGetField(ds::Dataset, field_name::String)
    out_len = Ref{Int64}()
    out_ptr = Ref{Ptr{Void}}()
    out_type = Ref{Cint}()
    @lightgbm(:LGBM_DatasetGetField,
              ds.handle => DatasetHandle,
              field_name => Cstring,
              out_len => Ref{Int64},
              out_ptr => Ref{Ptr{Void}},
              out_type => Ref{Cint})
    jl_out_type = idtotype32(out_type[])
    jl_out_ptr = convert(Ptr{jl_out_type}, out_ptr[])
    return copy(unsafe_wrap(Vector{jl_out_type}, jl_out_ptr, out_len[], false))
end

function LGBM_DatasetGetNumData(ds::Dataset)
    out = Ref{Int64}()
    @lightgbm(:LGBM_DatasetGetNumData,
              ds.handle => DatasetHandle,
              out => Ref{Int64})
    return out[]
end

function LGBM_DatasetGetNumFeature(ds::Dataset)
    out = Ref{Int64}()
    @lightgbm(:LGBM_DatasetGetNumFeature,
              ds.handle => DatasetHandle,
              out => Ref{Int64})
    return out[]
end

function LGBM_BoosterCreate(train_data::Dataset, valid_datas::Vector{Dataset},
                            valid_names::Vector{String}, parameters::String)
    n_valid_datas = length(valid_datas)
    lgbm_valid_datas = [ds.handle for ds in valid_datas]
    out = Ref{BoosterHandle}()
    @lightgbm(:LGBM_BoosterCreate,
              train_data.handle => DatasetHandle,
              lgbm_valid_datas => Ref{DatasetHandle},
              valid_names => Ref{Cstring},
              n_valid_datas => Cint,
              parameters => Cstring,
              out => Ref{BoosterHandle})
    return Booster(out[])
end

function LGBM_BoosterLoadFromModelfile(filename::String)
    out = Ref{BoosterHandle}()
    @lightgbm(:LGBM_BoosterLoadFromModelfile,
              filename => Cstring,
              out => Ref{BoosterHandle})
    return Booster(out[])
end

function LGBM_BoosterFree(bst::Booster)
    @lightgbm(:LGBM_BoosterFree,
              bst.handle => BoosterHandle)
    return nothing
end

function LGBM_BoosterUpdateOneIter(bst::Booster)
    is_finished = Ref{Cint}()
    @lightgbm(:LGBM_BoosterUpdateOneIter,
              bst.handle => BoosterHandle,
              is_finished => Ref{Cint})
    return is_finished[]
end

# function LGBM_BoosterUpdateOneIterCustom()

# Note: returns the reverse output of LGBM, because it currently stores scores in reverse order.
# TODO: Consider version that avoids allocation
function LGBM_BoosterEval(bst::Booster, data::Integer, n_metrics::Integer)
    out_results = Array(Cfloat, n_metrics)
    out_len = Ref{Int64}()
    @lightgbm(:LGBM_BoosterEval,
              bst.handle => BoosterHandle,
              data => Cint,
              out_len => Ref{Int64},
              out_results => Ref{Cfloat})
    return reverse!(out_results[1:out_len[]]) # TODO: check whether there's any real use for out_len.
end

function LGBM_BoosterGetScore(bst::Booster)
    out_len = Ref{Int64}()
    out_results = Ref{Ptr{Cfloat}}()
    @lightgbm(:LGBM_BoosterGetScore,
              bst.handle => BoosterHandle,
              out_len => Ref{Int64},
              out_results => Ref{Ptr{Cfloat}})
    return copy(unsafe_wrap(Vector{Cfloat}, out_results[], out_len[]))
end

function LGBM_BoosterGetPredict(bst::Booster, data::Integer, n_data::Integer)
    out_len = Ref{Int64}()
    out_results = Array(Cfloat, n_data)
    @lightgbm(:LGBM_BoosterGetPredict,
              bst.handle => BoosterHandle,
              data => Cint,
              out_len => Ref{Int64},
              out_results => Ref{Cfloat})
    return out_results[1:out_len[]] # TODO: check whether there's any real use for out_len.
end

# function LGBM_BoosterPredictForFile()
# function LGBM_BoosterPredictForCSR()

function LGBM_BoosterPredictForMat{T<:Union{Float32,Float64}}(bst::Booster, data::Matrix{T},
                                                              predict_type::Integer,
                                                              n_used_trees::Integer)
    lgbm_data_type = typetoid64(T)
    nrow, ncol = size(data)
    is_row_major = 0
    out_result = Array(Cdouble, nrow)
    @lightgbm(:LGBM_BoosterPredictForMat,
              bst.handle => BoosterHandle,
              data => Ref{T},
              lgbm_data_type => Cint,
              nrow => Int32,
              ncol => Int32,
              is_row_major => Cint,
              predict_type => Cint,
              n_used_trees => Int64,
              out_result => Ref{Cdouble})
    return out_result
end

function LGBM_BoosterPredictForMat{T<:Real}(bst::Booster, data::Matrix{T}, predict_type::Integer,
                                            n_used_trees::Integer)
    return LGBM_BoosterPredictForMat(bst, convert(Matrix{Float64}, data), predict_type,
                                     n_used_trees)
end

function LGBM_BoosterSaveModel(bst::Booster, num_used_model::Integer, filename::String)
    @lightgbm(:LGBM_BoosterSaveModel,
              bst.handle => BoosterHandle,
              num_used_model => Cint,
              filename => Cstring)
    return nothing
end
