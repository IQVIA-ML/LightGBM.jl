# C-API wrapper types

const DatasetHandle = Ptr{Nothing}
const BoosterHandle = Ptr{Nothing}

"""
Base type wrapping the LGBM C Dataset object
"""
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


"""
Base type wrapping the LGBM C Booster object
"""
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


# Base types for estimators
abstract type Estimator end
abstract type LGBMEstimator <: Estimator end


# A type for wrapping an objective function, whether a string or user supplied function
# `Base.print` needs to be overloaded to provide `string` functionality
# `Base.isempty` needs to be overloaded for empty checks (for the string represented type)
# Plus, a constructor given the user-supplied objective function
abstract type LGBMObjective end
Base.print(io::IO, obj::LGBMObjective) = print(io, obj.objective)
Base.isempty(x::LGBMObjective) = isempty(x.objective)
struct PredefinedObjective <: LGBMObjective
    objective::String
end
struct CustomObjective <: LGBMObjective
    objective::String
    custom_function::Function
end
LGBMObjective(x::String) = PredefinedObjective(x)
LGBMObjective(x::Function) = CustomObjective("custom", x)


abstract type LGBMFitData end
struct EmptyFitData <: LGBMFitData end
"""
Datatype holding data which is useful during fitting iterations
"""
struct CustomFitData <: LGBMFitData
    labels::Vector{Float32}
    weights::Vector{Float32}
    num_models::Integer
end
function CustomFitData(b::Booster)
    if length(bst.datasets) == 0
        throw(ErrorException("Booster does not have any training data associated"))
    end
    dataset = first(bst.datasets)
    labels = LGBM_DatasetGetField(dataset, "label")
    weights = LGBM_DatasetGetField(dataset, "weight")
    nummodels = LGBM_BoosterNumModelPerIteration(bst)

    return CustomFitData(labels, weights, nummodels)
end
LGBMFitData(::Booster, ::PredefinedObjective, ::Any) = EmptyFitData(), nothing

