function metaformattedclassresult(result::Array,Xtest::Array)
    rowsize=size(Xtest)[1]
    colsize=size(result)[1]/rowsize
    colsize=Int(colsize)
    metaformattedresult=zeros(rowsize,colsize)

    for i in 0:rowsize-1
        for j in 1:colsize
            metaformattedresult[i+1,j]=result[i*colsize+j,1]
        end
    end

    return metaformattedresult
end

function metaformattedclassresult(metaformattedresult::Array)
    rowsize,colsize=size(metaformattedresult)
    metaformattedclassresult=zeros(rowsize,1)

    for i in 1:rowsize
        work=0.0
        for j in 1:colsize
            if(metaformattedresult[i,j] >= work)
                work =metaformattedresult[i,j]
                metaformattedclassresult[i,1]=j
            end
        end
    end

    return metaformattedclassresult
end

function formattedclassfit(result::Array,Xtest::Array)
    return metaformattedclassresult(metaformattedclassresult(result,Xtest))
end

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
