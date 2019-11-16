"""
metaformattedclassresult(result::Array,Xtest::Array)
LightGBM実行形式で出力される行列フォーマットに変換
# Arguments
`result::Array`:prediction
`Xtest::Array`:the features data.
"""
function metaformattedclassresult(result::Array,Xtest::Array)
    rowsize=size(Xtest,1)
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
分類予測結果から、予測精度の一番高い結果だけをLightGBM実行形式で出力される形式で出力する
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
予測精度の一番高い結果だけをLightGBM実行形式で出力される形式で出力する。
# Arguments
`result::Array`:prediction result.
`Xtest::Array`:the features data.
"""
function formattedclassfit(result::Array,Xtest::Array)
    return metaformattedclassresult(metaformattedclassresult(result,Xtest))
end

"""
predict2(estimator::LGBMEstimator, Xtest::Array)
分類予測の場合、予測精度の一番高い結果だけを出力する。その他の予測は結果を出力
# Arguments
`estimator::LGBMEstimator`: the estimator to use in the prediction.
`Xtest::Array`:the features data.
"""
function predict2(estimator::LGBMEstimator, Xtest::Array)
    result=LightGBM.predict(estimator, Xtest)

    if(typeof(estimator) == LGBMMulticlass )
        result=formattedclassfit(result,Xtest)
    elseif(typeof(estimator) == LGBMRegression )
        result=result
    elseif(typeof(estimator) == LGBMBinary )
        result=result
    else
        println("Error")
    end

    return result
end
