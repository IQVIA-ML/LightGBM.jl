"""
metaformattedclassresult(result::Array,Xtest::Array)  

LightGBM実行形式で出力される行列フォーマットに変換  

Converts to matrix format output in LightGBM executable format  
# Arguments
`result::Array`:prediction  
`Xtest::Array`:the features data.  
"""
function metaformattedclassresult(result::Array,X::Array)
    rowsize=size(X,1)
    colsize=size(result,1)/rowsize
    colsize=Int(colsize)
    metaformattedresult=zeros(rowsize,colsize)

    for i in 0:rowsize-1
        for j in 1:colsize
            metaformattedresult[i+1,j]=result[i*colsize+j,1]
        end
    end

    return metaformattedresult
end


"""
metaformattedclassresult(metaformattedresult::Array)  

分類予測結果から、予測精度の一番高い結果だけをLightGBM実行形式で出力される形式で出力  

From the classification prediction result, only the result with the highest prediction accuracy is output in the output format in the LightGBM execution format  
# Arguments
`metaformattedresult::Array`:formatted prediction result.  
"""
function metaformattedclassresult(metaformattedresult::Matrix)
    rowsize,colsize=size(metaformattedresult)
    metaformattedclassresult=zeros(Int,rowsize,1)
    metaformattedclassresult=Array(metaformattedclassresult)

    for i in 1:rowsize
        work=-Inf
        for j in 1:colsize
            if(metaformattedresult[i,j] >= work)
                work =metaformattedresult[i,j]
                metaformattedclassresult[i,1]=j-1
            end
        end
    end

    return metaformattedclassresult
end

"""
formattedclassfit(result::Array,Xtest::Array)  

予測精度の一番高い結果だけをLightGBM実行形式で出力される形式で出力  

Only the result with the highest prediction accuracy is output in the output format in the LightGBM execution format  
# Arguments
`result::Array`:prediction result.  
`Xtest::Array`:the features data.  
"""
function formattedclassfit(result::Array,X::Array)
    return metaformattedclassresult(metaformattedclassresult(result,X))
end

"""
predict2(estimator::LGBMEstimator, Xtest::Array)  

分類予測の場合、予測精度の一番高い結果だけを出力する。その他の予測は結果をそのまま出力  

 In the case of classification prediction, only the result with the highest prediction accuracy is output. Other predictions output the results as they are  
# Arguments
* `estimator::LGBMEstimator`: the estimator to use in the prediction.
* `X::Matrix{T<:Real}`: the features data.
* `predict_type::Integer`: keyword argument that controls the prediction type. `0` for normal
    scores with transform (if needed), `1` for raw scores, `2` for leaf indices.
* `num_iterations::Integer`: keyword argument that sets the number of iterations of the model to
    use in the prediction. `< 0` for all iterations.
* `verbosity::Integer`: keyword argument that controls LightGBM's verbosity. `< 0` for fatal logs
    only, `0` includes warning logs, `1` includes info logs, and `> 1` includes debug logs.
* `is_row_major::Bool`: keyword argument that indicates whether or not `X` is row-major. `true`
    indicates that it is row-major, `false` indicates that it is column-major (Julia's default).
"""
function predict2(estimator::LGBMEstimator, X::Matrix{TX}; predict_type::Integer = 0,
                           num_iterations::Integer = -1, verbosity::Integer = 1,
                           is_row_major::Bool = false) where TX<:Real
    result=LightGBM.predict(estimator, X, predict_type = predict_type,
                           num_iterations = num_iterations, verbosity = verbosity,
                           is_row_major = is_row_major)

    if(typeof(estimator) == LGBMMulticlass )
        result=formattedclassfit(result,X)
    elseif(typeof(estimator) == LGBMRegression )
        result=result
    elseif(typeof(estimator) == LGBMBinary )
        result=result
    else
        println("Error")
    end

    return result
end
