# Tests that init_score works for multiclass
# reference URL
# https://github.com/Microsoft/LightGBM/issues/1778
# https://stackoverflow.com/questions/57275029/using-the-score-from-first-lightgbm-as-init-score-to-second-lightgbm-gives-diffe

using BenchmarkTools
# Create a 3-class problem
N1 = 1000
N2 = 1000
N3 = 1000
N = N1 + N2 + N3

numClasses = 3
numFeats = 5

X = rand(N, numFeats)
y = vcat(zeros(Int,N1), ones(Int,N2), 2*ones(Int,N3))

estimator1 = LightGBM.LGBMMulticlass(num_iterations = 50,
                                    learning_rate = .5,
                                    feature_fraction = 0.5,
                                    bagging_fraction = 1.0,
                                    bagging_freq = 1,
                                    num_leaves = 5,
                                    metric = ["multi_logloss"],
                                    num_class = numClasses,
                                    min_data_in_leaf=1,
                                    min_sum_hessian_in_leaf=1,
                                    early_stopping_round = 1);
                                    
# using the incorrect size of init_score should throw
estimator2=estimator1
LightGBM.fit(estimator1, X, y)

pre = LightGBM.predict(estimator1, X, verbosity = 0);
a=@benchmark LightGBM.predict(estimator1, X, verbosity = 0)

try
#    LightGBM.fit(estimator2, X, y, init_score = Vector(rand(numClasses*N)))
    LightGBM.fit(estimator2, X, y, init_score = Vector(pre))
    post = LightGBM.predict(estimator2, X, verbosity = 0) +pre
    b=@benchmark (LightGBM.predict(estimator2, X, verbosity = 0) +pre )
    ratio( mean(a), mean(b) )
    formattedclassfit(pre,X)
    formattedclassfit(post,X)
    
    @test  
    @test pre == post
    
    #false  # LightGBM.fit did not throw with incorrect init_score size
catch
end

# Create init score as if it were already predicting very well the data
init_score = -10000.0 * ones(numClasses, N)
for i in 1:N
    init_score[Int(y[i] + 1), i] *= -1
end
init_score = reshape(init_score, length(init_score))

LightGBM.fit(estimator, X, y; init_score = init_score)

# then it should have early stopped without training any tree
@test LightGBM.LGBM_BoosterGetCurrentIteration(estimator1.booster) == 50

