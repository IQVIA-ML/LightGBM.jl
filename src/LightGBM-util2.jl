function metaformattedclassresult(result::Array,Xtest)
    rowsize,colsize=size(Xtest)
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
                metaformattedclassresult[i,1]=j-1
            end
        end
    end

    return metaformattedclassresult
end

function formattedclassfit(result,Xtest)
    return metaformattedclassresult(metaformattedclassresult(result,Xtest))
end

function predict2(estimatorclass, Xtest)
    result=LightGBM.predict(estimatorclass, Xtest) 

    if(typeof(estimatorclass) == LGBMMulticlass )
        return formattedclassfit(result,Xtest)
    elseif(typeof(estimatorclass) == LGBMRegression )
        return 0 
    elseif(typeof(estimatorclass) == LGBMBinary )
        return 0
    else
        println("Error")
    end
end
