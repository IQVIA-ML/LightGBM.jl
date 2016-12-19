typealias DatasetHandle Ptr{Void}
typealias BoosterHandle Ptr{Void}

const C_API_DTYPE_FLOAT32 = 0
const C_API_DTYPE_FLOAT64 = 1
const C_API_DTYPE_INT32 = 2
const C_API_DTYPE_INT64 = 3

const C_API_PREDICT_NORMAL = 0
const C_API_PREDICT_RAW_SCORE = 1
const C_API_PREDICT_LEAF_INDEX = 2

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

type Booster
    handle::BoosterHandle
    datasets::Vector{Dataset}

    function Booster(handle::BoosterHandle, datasets::Vector{Dataset})
        bst = new(handle, datasets)
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

function Booster()
    return Booster(C_NULL, Dataset[])
end

function Booster(handle::BoosterHandle)
    return Booster(handle, Dataset[])
end


function jltype_to_lgbmid(datatype::Type)
    if datatype == Float32
        return C_API_DTYPE_FLOAT32
    elseif datatype == Float64
        return C_API_DTYPE_FLOAT64
    elseif datatype == Int32
        return C_API_DTYPE_INT32
    elseif datatype == Int64
        return C_API_DTYPE_INT64
    else
        error("unsupported datatype, got ", datatype)
    end
end

function lgbmid_to_jltype(id::Integer)
    if id == C_API_DTYPE_FLOAT32
        return Float32
    elseif id == C_API_DTYPE_FLOAT64
        return Float64
    elseif id == C_API_DTYPE_INT32
        return Int32
    elseif id == C_API_DTYPE_INT64
        return Int64
    else
        error("unknown LightGBM return type id, got ", id)
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

# function LGBM_DatasetCreateFromFile()
# function LGBM_DatasetCreateFromCSR()
# function LGBM_DatasetCreateFromCSC()

function LGBM_DatasetCreateFromMat{T<:Union{Float32,Float64}}(data::Matrix{T}, parameters::String)
    lgbm_data_type = jltype_to_lgbmid(T)
    nrow, ncol = size(data)
    is_row_major = 0
    out = Ref{DatasetHandle}()
    @lightgbm(:LGBM_DatasetCreateFromMat,
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

function LGBM_DatasetCreateFromMat{T<:Real}(data::Matrix{T}, parameters::String)
    return LGBM_DatasetCreateFromMat(convert(Matrix{Float64}, data), parameters)
end

function LGBM_DatasetCreateFromMat{T<:Union{Float32,Float64}}(data::Matrix{T}, parameters::String,
                                                              reference::Dataset)
    lgbm_data_type = jltype_to_lgbmid(T)
    nrow, ncol = size(data)
    is_row_major = 0
    out = Ref{DatasetHandle}()
    @lightgbm(:LGBM_DatasetCreateFromMat,
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

function LGBM_DatasetCreateFromMat{T<:Real}(data::Matrix{T}, parameters::String,
                                            reference::Dataset)
    return LGBM_DatasetCreateFromMat(convert(Matrix{Float64}, data), parameters, reference)
end

# Automatically converts to C's zero-based indices.
function LGBM_DatasetGetSubset(ds::Dataset, used_row_indices::Vector{Int32}, parameters::String)
    num_used_row_indices = length(used_row_indices)
    for idx in 1:num_used_row_indices
        @inbounds used_row_indices[idx] -= 1
    end
    out = Ref{DatasetHandle}()
    @lightgbm(:LGBM_DatasetGetSubset,
              ds.handle => Ref{DatasetHandle},
              used_row_indices => Ref{Int32},
              num_used_row_indices => Int32,
              parameters => Cstring,
              out => Ref{DatasetHandle})
    for idx in 1:num_used_row_indices
        @inbounds used_row_indices[idx] += 1
    end
    return Dataset(out[])
end

function LGBM_DatasetGetSubset(ds::Dataset, used_row_indices::Vector{Int64}, parameters::String)
    LGBM_DatasetGetSubset(ds, convert(Vector{Int32}, used_row_indices), parameters)
end

# TODO: implement test
function LGBM_DatasetSetFeatureNames(ds::Dataset, feature_names::Vector{String})
    num_feature_names = length(feature_names)
    @lightgbm(:LGBM_DatasetSetFeatureNames,
              ds.handle => DatasetHandle,
              feature_names => Ref{Cstring},
              num_feature_names => Int64)
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

function _LGBM_DatasetSetField{T<:Union{Float32,Int32}}(ds::Dataset, field_name::String,
                                                        field_data::Vector{T})
    data_type = jltype_to_lgbmid(T)
    num_element = length(field_data)
    @lightgbm(:LGBM_DatasetSetField,
              ds.handle => DatasetHandle,
              field_name => Cstring,
              field_data => Ref{T},
              num_element => Int64,
              data_type => Cint)
    return nothing
end

function LGBM_DatasetSetField(ds::Dataset, field_name::String, field_data::Vector{Float32})
    if field_name == "label" || field_name == "weight"
        _LGBM_DatasetSetField(ds, field_name, field_data)
    else
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Int32}, field_data))
    end
    return nothing
end

function LGBM_DatasetSetField(ds::Dataset, field_name::String, field_data::Vector{Int32})
    if field_name == "group" || field_name == "group_id"
        _LGBM_DatasetSetField(ds, field_name, field_data)
    else
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Float32}, field_data))
    end
    return nothing
end

function LGBM_DatasetSetField{T<:Real}(ds::Dataset, field_name::String, field_data::Vector{T})
    if field_name == "label" || field_name == "weight"
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Float32}, field_data))
    elseif field_name == "group" || field_name == "group_id"
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Int32}, field_data))
    end
    return nothing
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
    jl_out_type = lgbmid_to_jltype(out_type[])
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

function LGBM_BoosterCreate(train_data::Dataset, parameters::String)
    out = Ref{BoosterHandle}()
    @lightgbm(:LGBM_BoosterCreate,
              train_data.handle => DatasetHandle,
              parameters => Cstring,
              out => Ref{BoosterHandle})
    return Booster(out[], [train_data])
end

function LGBM_BoosterCreateFromModelfile(filename::String)
    out_num_iterations = Ref{Int64}()
    out = Ref{BoosterHandle}()
    @lightgbm(:LGBM_BoosterCreateFromModelfile,
              filename => Cstring,
              out_num_iterations => Ref{Int64},
              out => Ref{BoosterHandle})
    return Booster(out[])
end

function LGBM_BoosterFree(bst::Booster)
    @lightgbm(:LGBM_BoosterFree,
              bst.handle => BoosterHandle)
    return nothing
end

function LGBM_BoosterMerge(bst::Booster, other_booster::Booster)
    @lightgbm(:LGBM_BoosterMerge,
              bst.handle => BoosterHandle,
              other_booster.handle => BoosterHandle)
    return nothing
end

function LGBM_BoosterAddValidData(bst::Booster, valid_data::Dataset)
    @lightgbm(:LGBM_BoosterAddValidData,
              bst.handle => BoosterHandle,
              valid_data.handle => DatasetHandle)
    push!(bst.datasets, valid_data)
    return nothing
end

function LGBM_BoosterResetTrainingData(bst::Booster, train_data::Dataset)
    @lightgbm(:LGBM_BoosterResetTrainingData,
              bst.handle => BoosterHandle,
              train_data.handle => DatasetHandle)
    if length(bst.datasets) > 0
        @inbounds bst.datasets[1] = train_data
    else
        bst.datasets = [train_data]
    end
    return nothing
end

function LGBM_BoosterResetParameter(bst::Booster, parameters::String)
    @lightgbm(:LGBM_BoosterResetParameter,
              bst.handle => BoosterHandle,
              parameters => Cstring)
    return nothing
end

function LGBM_BoosterGetNumClasses(bst::Booster)
    out_len = Ref{Int64}()
    @lightgbm(:LGBM_BoosterGetNumClasses,
              bst.handle => BoosterHandle,
              out_len => Ref{Int64})
    return out_len[]
end

function LGBM_BoosterUpdateOneIter(bst::Booster)
    is_finished = Ref{Cint}()
    @lightgbm(:LGBM_BoosterUpdateOneIter,
              bst.handle => BoosterHandle,
              is_finished => Ref{Cint})
    return is_finished[]
end

# function LGBM_BoosterUpdateOneIterCustom()

function LGBM_BoosterRollbackOneIter(bst::Booster)
    @lightgbm(:LGBM_BoosterRollbackOneIter,
              bst.handle => BoosterHandle)
    return nothing
end

function LGBM_BoosterGetCurrentIteration(bst::Booster)
    out_iteration = Ref{Int64}()
    @lightgbm(:LGBM_BoosterGetCurrentIteration,
              bst.handle => BoosterHandle,
              out_iteration => Ref{Int64})
    return out_iteration[]
end

function LGBM_BoosterGetEvalCounts(bst::Booster)
    out_len = Ref{Int64}()
    @lightgbm(:LGBM_BoosterGetEvalCounts,
              bst.handle => BoosterHandle,
              out_len => Ref{Int64})
    return out_len[]
end

function LGBM_BoosterGetEvalNames(bst::Booster)
    out_len = Ref{Int64}()
    n_metrics = LGBM_BoosterGetEvalCounts(bst)
    out_strs = [Vector{UInt8}(256) for i in 1:n_metrics]
    @lightgbm(:LGBM_BoosterGetEvalNames,
              bst.handle => BoosterHandle,
              out_len => Ref{Int64},
              out_strs => Ref{Ptr{UInt8}})
    jl_out_strs = [unsafe_string(pointer(out_str)) for out_str in out_strs[1:out_len[]]]
    return jl_out_strs
end

function LGBM_BoosterGetEval(bst::Booster, data::Integer)
    n_metrics = LGBM_BoosterGetEvalCounts(bst)
    out_results = Array(Cfloat, n_metrics)
    out_len = Ref{Int64}()
    @lightgbm(:LGBM_BoosterGetEval,
              bst.handle => BoosterHandle,
              data => Cint,
              out_len => Ref{Int64},
              out_results => Ref{Cfloat})
    return out_results[1:out_len[]]
end

function LGBM_BoosterGetPredict(bst::Booster, data_idx::Integer, num_data::Integer)
    num_class = LGBM_BoosterGetNumClasses(bst)
    out_len = Ref{Int64}()
    out_results = Array(Cfloat, num_class * num_data)
    @lightgbm(:LGBM_BoosterGetPredict,
              bst.handle => BoosterHandle,
              data_idx => Cint,
              out_len => Ref{Int64},
              out_results => Ref{Cfloat})
    return out_results[1:out_len[]]
end

# function LGBM_BoosterPredictForFile()
# function LGBM_BoosterPredictForCSR()

function LGBM_BoosterPredictForMat{T<:Union{Float32,Float64}}(bst::Booster, data::Matrix{T},
                                                              predict_type::Integer,
                                                              num_iteration::Integer)
    num_class = LGBM_BoosterGetNumClasses(bst)
    lgbm_data_type = jltype_to_lgbmid(T)
    nrow, ncol = size(data)
    is_row_major = 0
    out_len = Ref{Int64}()
    alloc_len = ifelse(predict_type == C_API_PREDICT_LEAF_INDEX,
                       num_class * nrow * num_iteration,
                       num_class * nrow)
    out_result = Array(Cfloat, alloc_len)
    @lightgbm(:LGBM_BoosterPredictForMat,
              bst.handle => BoosterHandle,
              data => Ref{T},
              lgbm_data_type => Cint,
              nrow => Int32,
              ncol => Int32,
              is_row_major => Cint,
              predict_type => Cint,
              num_iteration => Int64,
              out_len => Ref{Int64},
              out_result => Ref{Cfloat})
    return out_result[1:out_len[]]
end

function LGBM_BoosterPredictForMat{T<:Real}(bst::Booster, data::Matrix{T}, predict_type::Integer,
                                            num_iteration::Integer)
    return LGBM_BoosterPredictForMat(bst, convert(Matrix{Float64}, data), predict_type,
                                     num_iteration)
end

function LGBM_BoosterSaveModel(bst::Booster, num_iteration::Integer, filename::String)
    @lightgbm(:LGBM_BoosterSaveModel,
              bst.handle => BoosterHandle,
              num_iteration => Cint,
              filename => Cstring)
    return nothing
end
