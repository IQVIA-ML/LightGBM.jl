# Tests that init_score works for multiclass

# Create a 3-class problem
N1 = 1000
N2 = 1000
N3 = 1000
N = N1 + N2 + N3

numClasses = 3
numFeats = 5

X = rand((N, numFeats))
y = vcat(zeros(N1), ones(N2), 2*ones(N3))

estimator = LightGBM.LGBMMulticlass(num_iterations = 50,
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
try
    LightGBM.fit(estimator, X, y; init_score = [1.2, 3.4])
    @test false  # LightGBM.fit did not throw with incorrect init_score size
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
@test LightGBM.LGBM_BoosterGetCurrentIteration(estimator.booster) == 0

