import main


origin_model_timestamp = "1529995862"
origin_model_dir = "./" + origin_model_timestamp + "/"
origin_training_dir = './training_data/taipei/training.data'
origin_testing_dir = './training_data/taipei/testing.data'
filter_threshold = 0.5

main.remove_model_with_low_accuracy(filter_threshold, origin_model_dir, './training_data/taipei/training.data')
origin_bagging_result, origin_accuracy = main.test_multi_classifier_bagging(origin_model_dir, origin_testing_dir)
main.store_bagging_detail(origin_bagging_result, origin_accuracy,"baseline.detail")