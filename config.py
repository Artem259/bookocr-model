image_size = 32
char_labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!:;-()&'"
dataset_folder = "fonts"


# for dataset
dataset_seed = 0
fonts_file = "font_names.txt"
size_factors = [0.5, 1, 2]
chars_types = ["aemnoru-.", "csvwxz", "gpqy,", "bdfhikltABDEFGHIJKLMNOPQRTUY0123456789?!:&'", "CSVWXZ", "()j;"]


# for cnn training
batch_size = 32
training_seed = 0
validation_split = 0.1
epochs_amount = 1000
early_stopping_patience = 50
early_stopping_start = 10
logs_folder = "logs"
model_weights_file = "model_weights.h5"
model_file = "model.h5"
