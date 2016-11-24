using LightGBM
using Base.Test

binary_test = readdlm(ENV["LIGHTGBM_PATH"] * "/examples/binary_classification/binary.test", '\t');
binary_train = readdlm(ENV["LIGHTGBM_PATH"] * "/examples/binary_classification/binary.train", '\t');
X_train = binary_train[:, 2:end];
y_train = convert(Vector{Float32}, binary_train[:, 1]);
X_test = binary_test[:, 2:end];
y_test = convert(Vector{Float32}, binary_test[:, 1]);

estimator = LightGBM.LGBMBinary(num_iterations = 20,
                                learning_rate = .1,
                                early_stopping_round = 1,
                                feature_fraction = .8,
                                bagging_fraction = .9,
                                bagging_freq = 1,
                                num_leaves = 1000,
                                metric = ["auc", "binary_logloss"],
                                metric_freq = 1,
                                is_training_metric = true,
                                max_bin = 255,
                                is_sigmoid = true,
                                min_sum_hessian_in_leaf = 0.,
                                min_data_in_leaf = 1);

LightGBM.fit(estimator, X_train, y_train, (X_test, y_test), verbosity = 2);
LightGBM.predict(estimator, X_train, verbosity = 2);

train_ds = LightGBM.LGBM_CreateDatasetFromMat(X_train, "objective=binary");
@test LightGBM.LGBM_DatasetGetNumData(train_ds) == 7000
@test LightGBM.LGBM_DatasetGetNumFeature(train_ds) == 28
@test LightGBM.LGBM_DatasetSetField(train_ds, "label", y_train) == nothing
@test LightGBM.LGBM_DatasetGetField(train_ds, "label") == y_train