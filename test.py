import main


origin_model_timestamp = "1529995862"
origin_model_dir = "./" + origin_model_timestamp + "/"
origin_training_dir = './training_data/taipei/training.data'
origin_testing_dir = './training_data/taipei/testing.data'
filter_threshold = 0.5

main.remove_model_with_low_accuracy(filter_threshold, origin_model_dir, './training_data/taipei/training.data')

