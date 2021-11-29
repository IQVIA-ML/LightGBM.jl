const DatasetHandle = Ptr{Nothing}
const BoosterHandle = Ptr{Nothing}

const C_API_DTYPE_FLOAT32 = 0
const C_API_DTYPE_FLOAT64 = 1
const C_API_DTYPE_INT32 = 2
const C_API_DTYPE_INT64 = 3
const C_API_MATRIX_TYPE_CSC = 1
const C_API_MATRIX_TYPE_CSR = 0

mutable struct Dataset
    handle::DatasetHandle

    function Dataset(handle::DatasetHandle)
        ds = new(handle)
        finalizer(Dataset_finalizer, ds)
        return ds
    end

    function Dataset_finalizer(ds::Dataset)
        if ds.handle != C_NULL
            LGBM_DatasetFree(ds)
        end
    end
end

mutable struct Booster
    handle::BoosterHandle
    datasets::Vector{Dataset}

    function Booster(handle::BoosterHandle, datasets::Vector{Dataset})
        bst = new(handle, datasets)
        finalizer(Booster_finalizer, bst)
        return bst
    end

    function Booster_finalizer(bst::Booster)
        if bst.handle != C_NULL
            LGBM_BoosterFree(bst)
        end
    end
end

function Booster()
    return Booster(C_NULL, Dataset[])
end

function Booster(handle::BoosterHandle)
    return Booster(handle, Dataset[])
end


# deepcopy utils, but we can't reasonably do this for datasets
function Base.deepcopy_internal(x::Booster, stackdict::IdDict)

    if haskey(stackdict, x)
        return stackdict[x]
    end

    if x.handle == C_NULL
        # just init a new object
        return Booster()
    end

    serialised = LGBM_BoosterSaveModelToString(x)
    y = LGBM_BoosterLoadModelFromString(serialised)

    return y

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

# sparse helpers for getting the API enums
sparsetypes(x::SparseArrays.SparseMatrixCSC) = sparseidxtype(x), sparsedatatype(x)


sparseidxtype(x::SparseArrays.SparseMatrixCSC{<:Any, Int64}) = C_API_DTYPE_INT64
sparseidxtype(x::SparseArrays.SparseMatrixCSC{<:Any, Int32}) = C_API_DTYPE_INT32


sparsedatatype(x::SparseArrays.SparseMatrixCSC{Float32, <:Integer}) = C_API_DTYPE_FLOAT32
sparsedatatype(x::SparseArrays.SparseMatrixCSC{Float64, <:Integer}) = C_API_DTYPE_FLOAT64
sparsedatatype(x::SparseArrays.SparseMatrixCSC{<:Any, <:Integer}) = throw(TypeError(:sparsedatatype, AbstractFloat, one(eltype(x.nzval))))


# Floating point conversion helpers
tofloat32(x::Vector{<:AbstractFloat}) = Float32.(x)
tofloat32(x::Vector{Float32}) = x


macro lightgbm(f, params...)
    return quote
        call_sym = Libdl.dlsym(LGBM_library[], $f)
        err_sym = Libdl.dlsym(LGBM_library[], :LGBM_GetLastError)
        err = ccall(call_sym, Cint,
                    ($((esc(i.args[end]) for i in params)...),),
                    $((esc(i.args[end - 1]) for i in params)...))
        if err != 0
            msg = unsafe_string(ccall(err_sym, Cstring, ()))
            error("call to LightGBM's ", string($(esc(f))), " failed: ", msg)
        end
    end
end


# function LGBM_DatasetCreateFromFile()


function LGBM_DatasetCreateFromCSC(
    data::SparseArrays.SparseMatrixCSC,
    parameters::String,
    reference::Dataset = Dataset(C_NULL),
)

    if data.m > typemax(Int32)
        throw(DomainError(data.m, "Cannot accept CSC matrices with more than $(typemax(Int32)) rows"))
    end

    idx_type, data_type = sparsetypes(data)

    out = Ref{DatasetHandle}()
    @lightgbm(
        :LGBM_DatasetCreateFromCSC,
        data.colptr .- 1 => Ptr{Nothing},
        idx_type => Cint,
        Int32.(data.rowval .- 1) => Ptr{Cint},
        data.nzval => Ptr{Nothing},
        data_type => Cint,
        data.n + 1 => Clonglong,
        SparseArrays.nnz(data) => Clonglong,
        data.m => Clonglong,
        parameters => Cstring,
        reference.handle => DatasetHandle,
        out => Ref{DatasetHandle}
    )
    return Dataset(out[])

end

function LGBM_DatasetCreateFromMat(
    data::Matrix{T},
    parameters::String,
    is_row_major::Bool = false
) where T<:Union{Float32,Float64}

    lgbm_data_type = jltype_to_lgbmid(T)
    nrow, ncol = ifelse(is_row_major, reverse(size(data)), size(data))
    out = Ref{DatasetHandle}()
    @lightgbm(
        :LGBM_DatasetCreateFromMat,
        data => Ptr{Nothing},
        lgbm_data_type => Cint,
        nrow => Int32,
        ncol => Int32,
        is_row_major => Cint,
        parameters => Cstring,
        C_NULL => Ptr{Nothing},
        out => Ref{DatasetHandle}
    )
    return Dataset(out[])
end

function LGBM_DatasetCreateFromMat(data::Matrix{T}, parameters::String, is_row_major::Bool = false) where T<:Real
    return LGBM_DatasetCreateFromMat(convert(Matrix{Float64}, data), parameters, is_row_major)
end

function LGBM_DatasetCreateFromMat(
    data::Matrix{T},
    parameters::String,
    reference::Dataset,
    is_row_major::Bool = false
) where T<:Union{Float32,Float64}

    lgbm_data_type = jltype_to_lgbmid(T)
    nrow, ncol = ifelse(is_row_major, reverse(size(data)), size(data))
    out = Ref{DatasetHandle}()
    @lightgbm(
        :LGBM_DatasetCreateFromMat,
        data => Ptr{Nothing},
        lgbm_data_type => Cint,
        nrow => Int32,
        ncol => Int32,
        is_row_major => Cint,
        parameters => Cstring,
        reference.handle => DatasetHandle,
        out => Ref{DatasetHandle}
    )
    return Dataset(out[])
end

function LGBM_DatasetCreateFromMat(data::Matrix{T}, parameters::String,
                                            reference::Dataset, is_row_major::Bool = false) where T<:Real
    return LGBM_DatasetCreateFromMat(convert(Matrix{Float64}, data), parameters, reference,
                                     is_row_major)
end

# Automatically converts to C's zero-based indices.
function LGBM_DatasetGetSubset(ds::Dataset, used_row_indices::Vector{Int32}, parameters::String)
    num_used_row_indices = length(used_row_indices)
    for idx in 1:num_used_row_indices
        @inbounds used_row_indices[idx] -= 1
    end
    out = Ref{DatasetHandle}()
    @lightgbm(:LGBM_DatasetGetSubset,
              ds.handle => DatasetHandle,
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

function LGBM_DatasetSetFeatureNames(ds::Dataset, feature_names::Vector{String})
    num_feature_names = length(feature_names)
    @lightgbm(:LGBM_DatasetSetFeatureNames,
              ds.handle => DatasetHandle,
              feature_names => Ref{Cstring},
              num_feature_names => Cint)
end

function LGBM_DatasetGetFeatureNames(ds::Dataset)
    len = Cint(2)
    buffer_len = Csize_t(2)
    feature_names = [Vector{UInt8}(undef, buffer_len) for i in 1:len]

   # setting these so that the first C API call informs us of their allocations
    num_feature_names = Ref{Cint}()
    out_buffer_len = Ref{Csize_t}()

    @lightgbm(
        :LGBM_DatasetGetFeatureNames,
        ds.handle => DatasetHandle,
        len => Cint,
        num_feature_names => Ref{Cint},
        buffer_len => Csize_t,
        out_buffer_len => Ref{Csize_t},
        feature_names => Ref{Ptr{UInt8}}
    )

    # allocating memory
    new_len = num_feature_names[]
    new_buffer_len = out_buffer_len[]
    feature_names = [Vector{UInt8}(undef, new_buffer_len) for i in 1:new_len]

    @lightgbm(
        :LGBM_DatasetGetFeatureNames,
        ds.handle => DatasetHandle,
        new_len => Cint,
        num_feature_names => Ref{Cint},
        new_buffer_len => Csize_t,
        out_buffer_len => Ref{Csize_t},
        feature_names => Ref{Ptr{UInt8}}
    )

    return [unsafe_string(pointer(feature_name)) for feature_name in feature_names[1:num_feature_names[]]]
end

function LGBM_DatasetFree(ds::Dataset)
    @lightgbm(:LGBM_DatasetFree,
              ds.handle => DatasetHandle)
    ds.handle = C_NULL # avoid a class of double free bugs please
    return nothing
end

function LGBM_DatasetSaveBinary(ds::Dataset, filename::String)
    @lightgbm(:LGBM_DatasetSaveBinary,
              ds.handle => DatasetHandle,
              filename => Cstring)
    return nothing
end

function _LGBM_DatasetSetField(ds::Dataset, field_name::String,
                                                        field_data::Vector{T}) where T <:Union{Float32,Float64,Int32}
    data_type = jltype_to_lgbmid(T)
    num_element = length(field_data)
    @lightgbm(
        :LGBM_DatasetSetField,
        ds.handle => DatasetHandle,
        field_name => Cstring,
        field_data => Ptr{Nothing},
        num_element => Cint,
        data_type => Cint
    )
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
    if field_name == "group"
        _LGBM_DatasetSetField(ds, field_name, field_data)
    else
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Float32}, field_data))
    end
    return nothing
end

function LGBM_DatasetSetField(ds::Dataset, field_name::String, field_data::Vector{T}) where T<:Real
    if field_name == "label" || field_name == "weight"
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Float32}, field_data))
    elseif field_name == "init_score"
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Float64}, field_data))
    elseif field_name == "group"
        _LGBM_DatasetSetField(ds, field_name, convert(Vector{Int32}, field_data))
    end
    return nothing
end

function LGBM_DatasetGetField(ds::Dataset, field_name::String)
    out_len = Ref{Cint}()
    out_ptr = Ref{Ptr{Nothing}}()
    out_type = Ref{Cint}()
    @lightgbm(
        :LGBM_DatasetGetField,
        ds.handle => DatasetHandle,
        field_name => Cstring,
        out_len => Ref{Cint},
        out_ptr => Ref{Ptr{Nothing}},
        out_type => Ref{Cint}
    )
    jl_out_type = lgbmid_to_jltype(out_type[])
    jl_out_ptr = convert(Ptr{jl_out_type}, out_ptr[])
    return copy(unsafe_wrap(Vector{jl_out_type}, jl_out_ptr, out_len[], own=false))
end

function LGBM_DatasetGetNumData(ds::Dataset)
    out = Ref{Cint}()
    @lightgbm(:LGBM_DatasetGetNumData,
              ds.handle => DatasetHandle,
              out => Ref{Cint})
    return out[]
end

function LGBM_DatasetGetNumFeature(ds::Dataset)
    out = Ref{Cint}()
    @lightgbm(:LGBM_DatasetGetNumFeature,
              ds.handle => DatasetHandle,
              out => Ref{Cint})
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
    out_num_iterations = Ref{Cint}()
    out = Ref{BoosterHandle}()
    @lightgbm(:LGBM_BoosterCreateFromModelfile,
              filename => Cstring,
              out_num_iterations => Ref{Cint},
              out => Ref{BoosterHandle})
    return Booster(out[])
end

function LGBM_BoosterLoadModelFromString(model_str::String)
    out_num_iterations = Ref{Cint}()
    out = Ref{BoosterHandle}()
    @lightgbm(:LGBM_BoosterLoadModelFromString,
              model_str => Cstring,
              out_num_iterations => Ref{Cint},
              out => Ref{BoosterHandle})
    return Booster(out[])
end

function LGBM_BoosterFree(bst::Booster)
    @lightgbm(:LGBM_BoosterFree,
              bst.handle => BoosterHandle)
    bst.handle = C_NULL # avoid a class of double free bugs
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
    out_len = Ref{Cint}()
    @lightgbm(:LGBM_BoosterGetNumClasses,
              bst.handle => BoosterHandle,
              out_len => Ref{Cint})
    return out_len[]
end

function LGBM_BoosterUpdateOneIter(bst::Booster)
    is_finished = Ref{Cint}()
    @lightgbm(:LGBM_BoosterUpdateOneIter,
              bst.handle => BoosterHandle,
              is_finished => Ref{Cint})
    return is_finished[]
end

"""
LGBM_BoosterUpdateOneIterCustom
Pass grads and 2nd derivatives corresponding to some custom loss function
grads and 2nd derivatives must be same cardinality as training data * number of models
Also, trying to run this on a booster without data will fail.
"""
function LGBM_BoosterUpdateOneIterCustom(bst::Booster, grads::Vector{<:AbstractFloat}, hessian::Vector{<:AbstractFloat})

    if length(bst.datasets) == 0
        throw(ErrorException("Booster does not have any training data associated"))
    end
    numdata = LGBM_DatasetGetNumData(first(bst.datasets))
    nummodels = LGBM_BoosterNumModelPerIteration(bst)

    if !((numdata*nummodels) == length(grads) == length(hessian))
        throw(DimensionMismatch(
            "Gradients sizes ($(length(grads)), $(length(hessian))) don't match training data size ($numdata) * ($nummodels)"
        ))
    end

    grads = tofloat32(grads)
    hessian = tofloat32(hessian)

    is_finished = Ref{Cint}()
    @lightgbm(:LGBM_BoosterUpdateOneIterCustom,
              bst.handle => BoosterHandle,
              grads => Ptr{Cfloat},
              hessian => Ptr{Cfloat},
              is_finished => Ref{Cint})
    return is_finished[]

end

function LGBM_BoosterRollbackOneIter(bst::Booster)
    @lightgbm(:LGBM_BoosterRollbackOneIter,
              bst.handle => BoosterHandle)
    return nothing
end

function LGBM_BoosterGetCurrentIteration(bst::Booster)
    out_iteration = Ref{Cint}()
    @lightgbm(:LGBM_BoosterGetCurrentIteration,
              bst.handle => BoosterHandle,
              out_iteration => Ref{Cint})
    return out_iteration[]
end

function LGBM_BoosterGetEvalCounts(bst::Booster)
    out_len = Ref{Cint}()
    @lightgbm(:LGBM_BoosterGetEvalCounts,
              bst.handle => BoosterHandle,
              out_len => Ref{Cint})
    return out_len[]
end

function LGBM_BoosterGetEvalNames(bst::Booster)
    len = Cint(2)
    buffer_len = Csize_t(2)
    out_strs = [Vector{UInt8}(undef, buffer_len) for i in 1:len]

    # setting these so that the first C API call informs us of their allocations
    out_len = Ref{Cint}()
    out_buffer_len = Ref{Csize_t}()

    @lightgbm(
        :LGBM_BoosterGetEvalNames,
        bst.handle => BoosterHandle,
        len => Cint,
        out_len => Ref{Cint},
        buffer_len => Csize_t,
        out_buffer_len => Ref{Csize_t},
        out_strs => Ref{Ptr{UInt8}}
    )

    # allocating memory
    new_len = out_len[]
    new_buffer_len = out_buffer_len[]
    out_strs = [Vector{UInt8}(undef, new_buffer_len) for i in 1:new_len]

    @lightgbm(
        :LGBM_BoosterGetEvalNames,
        bst.handle => BoosterHandle,
        new_len => Cint,
        out_len => Ref{Cint},
        new_buffer_len => Csize_t,
        out_buffer_len => Ref{Csize_t},
        out_strs => Ref{Ptr{UInt8}}
    )

    jl_out_strs = [unsafe_string(pointer(out_str)) for out_str in out_strs[1:out_len[]]]
    return jl_out_strs
end

function LGBM_BoosterGetFeatureNames(bst::Booster)
    len = Cint(2)
    buffer_len = Csize_t(2)
    out_strs = [Vector{UInt8}(undef, buffer_len) for i in 1:len]

    # setting these so that the first C API call informs us of their allocations
    out_len = Ref{Cint}()
    out_buffer_len = Ref{Csize_t}()

    @lightgbm(
        :LGBM_BoosterGetFeatureNames,
        bst.handle => BoosterHandle,
        len => Cint,
        out_len => Ref{Cint},
        buffer_len => Csize_t,
        out_buffer_len => Ref{Csize_t},
        out_strs => Ref{Ptr{UInt8}}
    )

    # allocating memory
    new_len = out_len[]
    new_buffer_len = out_buffer_len[]
    out_strs = [Vector{UInt8}(undef, new_buffer_len) for i in 1:new_len]

    @lightgbm(
        :LGBM_BoosterGetFeatureNames,
        bst.handle => BoosterHandle,
        new_len => Cint,
        out_len => Ref{Cint},
        new_buffer_len => Csize_t,
        out_buffer_len => Ref{Csize_t},
        out_strs => Ref{Ptr{UInt8}}
    )

    jl_out_strs = [unsafe_string(pointer(out_str)) for out_str in out_strs[1:out_len[]]]
    return jl_out_strs
end

function LGBM_BoosterGetNumFeature(bst::Booster)
    out_len = Ref{Cint}()
    @lightgbm(:LGBM_BoosterGetNumFeature,
              bst.handle => BoosterHandle,
              out_len => Ref{Cint})
    return out_len[]
end

function LGBM_BoosterGetEval(bst::Booster, data::Integer)
    n_metrics = LGBM_BoosterGetEvalCounts(bst)
    out_results = Array{Cdouble}(undef, n_metrics)
    out_len = Ref{Cint}()
    @lightgbm(:LGBM_BoosterGetEval,
              bst.handle => BoosterHandle,
              data => Cint,
              out_len => Ref{Cint},
              out_results => Ref{Cdouble})
    return out_results[1:out_len[]]
end

function LGBM_BoosterGetNumPredict(bst::Booster, data_idx::Integer)
    out_len = Ref{Int64}()
    @lightgbm(:LGBM_BoosterGetNumPredict,
              bst.handle => BoosterHandle,
              data_idx => Cint,
              out_len => Ref{Int64})
    return out_len[]
end

function LGBM_BoosterGetPredict(bst::Booster, data_idx::Integer)
    out_len = Ref{Int64}()
    num_class = LGBM_BoosterGetNumClasses(bst)
    num_data = LGBM_BoosterGetNumPredict(bst, data_idx)
    out_results = Array{Cdouble}(undef, num_class * num_data)
    @lightgbm(:LGBM_BoosterGetPredict,
              bst.handle => BoosterHandle,
              data_idx => Cint,
              out_len => Ref{Int64},
              out_results => Ref{Cdouble})
    return out_results[1:out_len[]]
end

# function LGBM_BoosterPredictForFile()

function LGBM_BoosterCalcNumPredict(bst::Booster, num_row::Integer, predict_type::Integer, start_iteration::Integer,
                                    num_iteration::Int)
    out_len = Ref{Int64}()

    @lightgbm(
        :LGBM_BoosterCalcNumPredict,
        bst.handle => BoosterHandle,
        num_row => Cint,
        predict_type => Cint,
        start_iteration => Cint,
        num_iteration => Cint,
        out_len => Ref{Int64}
    )

    return out_len[]
end

# function LGBM_BoosterPredictForCSR()
# function LGBM_BoosterPredictForCSC()

function LGBM_BoosterPredictForMat(
    bst::Booster,
    data::Matrix{T},
    predict_type::Integer,
    start_iteration::Integer,
    num_iteration::Integer,
    is_row_major::Bool = false
) where T<:Union{Float32,Float64}

    lgbm_data_type = jltype_to_lgbmid(T)
    nrow, ncol = ifelse(is_row_major, reverse(size(data)), size(data))
    out_len = Ref{Int64}()
    alloc_len = LGBM_BoosterCalcNumPredict(bst, nrow, predict_type, start_iteration, num_iteration)
    out_result = Array{Cdouble}(undef, alloc_len)

    parameter = ""  # full prediction, no early stopping
    @lightgbm(
        :LGBM_BoosterPredictForMat,
        bst.handle => BoosterHandle,
        data => Ptr{Nothing},
        lgbm_data_type => Cint,
        nrow => Int32,
        ncol => Int32,
        is_row_major => Cint,
        predict_type => Cint,
        start_iteration => Cint,
        num_iteration => Cint,
        parameter => Cstring,
        out_len => Ref{Int64},
        out_result => Ref{Cdouble}
    )

    return out_result[1:out_len[]]
end

function LGBM_BoosterPredictForMat(bst::Booster, data::Matrix{T}, predict_type::Integer,
                                            num_iteration::Integer) where T<:Real
    return LGBM_BoosterPredictForMat(bst, convert(Matrix{Float64}, data), predict_type,
                                     num_iteration)
end

function LGBM_BoosterSaveModel(
    bst::Booster,
    start_iteration::Integer,
    num_iteration::Integer,
    feature_importance_type::Integer,
    filename::String
)
    @lightgbm(
        :LGBM_BoosterSaveModel,
        bst.handle => BoosterHandle,
        start_iteration => Cint,
        num_iteration => Cint,
        feature_importance_type => Cint,
        filename => Cstring
    )
    return nothing
end

function LGBM_BoosterSaveModelToString(
    bst::Booster,
    start_iteration::Integer=0,
    num_iteration::Integer=0,
    feature_importance_type::Integer=0
)::String

    # places for the call to write to
    out_len = Ref{Int64}()
    out_str = Vector{UInt8}(undef, 2)
    buffer_len = Int64(1)

    # first time will not work, we calling it to be told out_len
    @lightgbm(
        :LGBM_BoosterSaveModelToString,
        bst.handle => BoosterHandle,
        start_iteration => Cint,
        num_iteration => Cint,
        feature_importance_type => Cint,
        buffer_len => Int64,
        out_len => Ref{Int64},
        out_str => Ref{UInt8}
    )

    out_str = Vector{UInt8}(undef, out_len[] + 1)
    buffer_len = out_len[]

    # now it works, and we have our serialised model in out_str, in c-memory
    @lightgbm(
        :LGBM_BoosterSaveModelToString,
        bst.handle => BoosterHandle,
        start_iteration => Cint,
        num_iteration => Cint,
        feature_importance_type  => Cint,
        buffer_len => Int64,
        out_len => Ref{Int64},
        out_str => Ref{UInt8}
    )

    jl_out_str = unsafe_string(pointer(out_str))
    return jl_out_str

end


function LGBM_BoosterFeatureImportance(bst::Booster, num_iteration::Integer, importance_type::Integer)::Vector{Float64}

    num_features = LGBM_BoosterGetNumFeature(bst)
    out_result = Array{Cdouble}(undef, num_features)

    @lightgbm(
        :LGBM_BoosterFeatureImportance,
        bst.handle => BoosterHandle,
        num_iteration => Cint,
        importance_type => Cint,
        out_result => Ref{Cdouble},
    )

    return out_result
end

# function LGBM_BoosterDumpModel()
# function LGBM_BoosterGetLeafValue()
# function LGBM_BoosterSetLeafValue()

function LGBM_BoosterNumModelPerIteration(bst::Booster)
    out_models = Ref{Cint}()
    @lightgbm(:LGBM_BoosterNumModelPerIteration,
              bst.handle => BoosterHandle,
              out_models => Ref{Cint})
    return out_models[]
end
