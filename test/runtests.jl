ENV["LIGHTGBM"] = "/Users/Allard/GitHub/LightGBM/lightgbm"
githubdir = "/Users/Allard/Github"
include(githubdir * "/LightGBM.jl/src/LightGBM.jl")
using LightGBM


binary_test = readdlm("/Users/Allard/GitHub/LightGBM/examples/binary_classification/binary.test", '\t');
binary_train = readdlm("/Users/Allard/GitHub/LightGBM/examples/binary_classification/binary.train", '\t');
x_train = binary_train[:, 2:end];
y_label = convert(Array{Float32,1}, binary_train[:, 1]);
x_test = binary_test[:, 2:end];
y_test = convert(Array{Float32,1}, binary_test[:, 1]);

estimator = LGBMBinary(num_iterations = 10,
                       learning_rate = .1,
                       feature_fraction = .8,
                       bagging_fraction = .8,
                       bagging_freq = 5,
                       num_leaves = 63,
                       metric = ["auc", "binary_logloss"],
                       metric_freq = 4,
                       is_training_metric = true,
                       max_bin = 255,
                       is_sigmoid = true,
                       min_sum_hessian_in_leaf = 5.,
                       min_data_in_leaf = 50);

a = LightGBM.fit(estimator, x_train, y_label, (x_test, y_test), verbosity = 1)




#using LightGBM
using Base.Test




train_handle = LGBM_CreateDatasetFromMat(x_train, "objective=binary boosting=gbdt metric=binary_logloss,auc metric_freq=1 is_training_metric=true max_bin=255 num_trees=100 learning_rate=0.1 num_leaves=63 tree_learner=serial feature_fraction=0.8 bagging_freq=5 bagging_fraction=0.8 min_data_in_leaf=50 min_sum_hessian_in_leaf=5.", C_NULL)

@test LGBM_DatasetGetNumData(train_handle) == 7000
@test LGBM_DatasetGetNumFeature(train_handle) == 28

LGBM_DatasetSetField(train_handle, "label", y_label);
@test LGBM_DatasetGetField(train_handle, "label")[1:5] == [1f0, 1f0, 1f0, 0f0, 1f0]

test_handle = LGBM_CreateDatasetFromMat(x_test, "objective=binary boosting=gbdt metric=binary_logloss,auc metric_freq=1 is_training_metric=true max_bin=255 num_trees=100 learning_rate=0.1 num_leaves=63 tree_learner=serial feature_fraction=0.8 bagging_freq=5 bagging_fraction=0.8 min_data_in_leaf=50 min_sum_hessian_in_leaf=5.", C_NULL)

bst_handle = LGBM_BoosterCreate(train_handle, [test_handle], ["test-1"], "objective=binary boosting=gbdt metric=binary_logloss,auc metric_freq=1 is_training_metric=true max_bin=255 num_trees=100 learning_rate=0.1 num_leaves=63 tree_learner=serial feature_fraction=0.8 bagging_freq=5 bagging_fraction=0.8 min_data_in_leaf=50 min_sum_hessian_in_leaf=5.")
LGBM_BoosterUpdateOneIter(bst_handle)
LGBM_BoosterUpdateOneIter(bst_handle)
LGBM_BoosterUpdateOneIter(bst_handle)
LGBM_BoosterEval(bst_handle, 0, 2)
LGBM_BoosterGetScore(bst_handle)
LGBM_BoosterGetPredict(bst_handle, 0, 7000)

LGBM_BoosterPredictForMat(bst_handle, x_test, 0, 1)
