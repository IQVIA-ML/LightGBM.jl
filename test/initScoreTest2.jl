# Tests that init_score works for multiclass
# reference URL
# https://github.com/Microsoft/LightGBM/issues/1778
# https://stackoverflow.com/questions/57275029/using-the-score-from-first-lightgbm-as-init-score-to-second-lightgbm-gives-diffe

# Test regression estimator.
regression_test = readdlm(ENV["LIGHTGBM_PATH"] * "/examples/regression/regression.test", '\t');
regression_train = readdlm(ENV["LIGHTGBM_PATH"] * "/examples/regression/regression.train", '\t');
X_train = regression_train[:, 2:end]
y_train = regression_train[:, 1]
X_test = regression_test[:, 2:end]
y_test = regression_test[:, 1]

regression_test_init = readdlm(ENV["LIGHTGBM_PATH"] * "/examples/regression/regression.test.init", '\t')[:,1];
regression_train_init = readdlm(ENV["LIGHTGBM_PATH"] * "/examples/regression/regression.train.init", '\t')[:,1];

estimator = LightGBM.LGBMRegression(num_iterations = 100,
                                    learning_rate = .05,
                                    feature_fraction = .9,
                                    bagging_fraction = .8,
                                    bagging_freq = 5,
                                    num_leaves = 31,
                                    metric = ["l2"],
                                    metric_freq = 1,
                                    is_training_metric = true,
                                    max_bin = 255,
                                    min_sum_hessian_in_leaf = 5.,
                                    min_data_in_leaf = 100,
                                    max_depth = -1);

scores = LightGBM.fit(estimator, X_train, y_train, (X_test, y_test), verbosity = 0,init_score=regression_train_init);
@test scores["test_1"]["l2"][end] < .5