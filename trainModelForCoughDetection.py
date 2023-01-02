import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For prettier confusion matrix
import csv
warnings.filterwarnings('ignore')
# Keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, cohen_kappa_score, roc_auc_score

# NOTE ON NAMING CONVENTIONS - MODELS ARE NAMED ALPHABETICALLY USING INDICATORS OF DB / OPTIMIZERS / MODEL
# A = ADAM OPTIMIZER
# S = SDG OPTIMIZER
# FOLLOWED BY A NUMBER INDICATOR FOR ITS ITERATION
# STANDARD = S
# NORMALIZATION = N
# OVERLAP = O
# NOISE REDUCTION = R
# MIXED = M

# SETTINGS
SIZE_X = 100  # Based on image size generated
SIZE_Y = 64  # Based on image size generated
TRAIN_SAMPLES = 2736
VALID_SAMPLES = 912
TEST_SAMPLES = 910
TRAIN_SAMPLES_OVERLAPPED = 8100
VALID_SAMPLES_OVERLAPPED = 2730
TEST_SAMPLES_OVERLAPPED = 2730
TRAIN_SAMPLES_MIXED = 47128
VALID_SAMPLES_MIXED = 15702
TEST_SAMPLES_MIXED = 15694

# PATHS
BASE_SOURCE_PATH = 'data/spectrograms/cough_detection/'
STANDARD_DATA = BASE_SOURCE_PATH
NORM_DATA = BASE_SOURCE_PATH + 'with_norm/'
NR_DATA = BASE_SOURCE_PATH + 'with_noise_reduction/'
NR_AND_NORM = BASE_SOURCE_PATH + 'with_nr_and_norm/'
OVERLAP_DATA = BASE_SOURCE_PATH + "with_overlap/"
OL_AND_NORM = BASE_SOURCE_PATH + "with_ol_and_norm/"
OL_AND_NR = BASE_SOURCE_PATH + "with_ol_and_nr/"
OL_NR_AND_NORM = BASE_SOURCE_PATH + "with_ol_nr_and_norm/"
MIXED = BASE_SOURCE_PATH + "mixed/"
EXPORT_BASE_PATH = 'results/models/'

# HYPER PARAMETERS
EPOCHS = 30
PATIENCE = 30  # Unused, kept as a potential parameter to impl.
BATCH_SIZE = 16
SHUFFLE = False  # Shuffling may falsify predictions as it automatically reshuffles
LEARNING_RATE = 0.00001
DECAY_RATE = LEARNING_RATE / EPOCHS
STEPS_PER_EPOCH_TRAIN = TRAIN_SAMPLES / BATCH_SIZE - BATCH_SIZE
STEPS_PER_EPOCH_VALID = VALID_SAMPLES / BATCH_SIZE - BATCH_SIZE
STEPS_PER_EPOCH_TRAIN_OVERLAPPED = TRAIN_SAMPLES_OVERLAPPED / BATCH_SIZE - BATCH_SIZE
STEPS_PER_EPOCH_VALID_OVERLAPPED = VALID_SAMPLES_OVERLAPPED / BATCH_SIZE - BATCH_SIZE
STEPS_PER_EPOCH_TRAIN_MIXED = TRAIN_SAMPLES_MIXED / BATCH_SIZE - BATCH_SIZE
STEPS_PER_EPOCH_VALID_MIXED = VALID_SAMPLES_MIXED / BATCH_SIZE - BATCH_SIZE
MOMENTUM = 0.9
SGD_USED = SGD(lr=LEARNING_RATE, momentum=MOMENTUM, decay=DECAY_RATE, nesterov=True)
ADAM = Adam(learning_rate=LEARNING_RATE)
COLOR_MODE = 'grayscale'  # No fancy RGB, input spectrograms are greyscale
CLASS_MODE = 'binary'  # Spectrograms either contain a cough or not, so binary 0 / 1
LOSS_FUNCTION = "binary_crossentropy"  # That's also why a binary loss function is used


# Metrics
def prediction_values_for_set(model, set):
    y_probability = model.predict(set, verbose=0)
    # Calls to model.predict seems to reshuffle using flow_from_directory
    # so getting class predictions with it doesn't match the labels
    # to prevent this SHUFFLE is set to False
    y_classes = []
    for probability in y_probability:
        y_classes.append((probability > 0.5).astype("int32"))
    # Since we evaluate binary for coughs probability > 0.5 is used here
    # Reduced to 1d array as scikit-learn metrics expects 1D array
    y_probability = y_probability[:, 0]
    # Accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(set.labels, y_classes)
    print('Accuracy: %f' % accuracy)
    # Precision tp / (tp + fp)
    precision = precision_score(set.labels, y_classes)
    print('Precision: %f' % precision)
    # Recall: tp / (tp + fn)
    recall = recall_score(set.labels, y_classes)
    print('Recall: %f' % recall)
    # F1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(set.labels, y_classes)
    print('F1 score: %f' % f1)
    # Kappa
    kappa = cohen_kappa_score(set.labels, y_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(set.labels, y_probability)
    print('ROC AUC: %f' % auc)
    # Confusion Matrix
    matrix = confusion_matrix(set.labels, y_classes)
    print(matrix)
    matrix = matrix.ravel()  # To get: (tn, fp, fn, tp)
    return [accuracy, precision, recall, f1, kappa, auc, matrix[0], matrix[1], matrix[2], matrix[3]]


def train_and_evaluate_model(model,
                             model_name="Standard",
                             with_data=STANDARD_DATA,
                             optimizer_to_be_used=ADAM,
                             loss_function_to_use=LOSS_FUNCTION,
                             with_metrics=True,
                             with_predictions=True,
                             save_model=True,
                             epoch_steps_train=STEPS_PER_EPOCH_TRAIN,
                             epoch_steps_valid=STEPS_PER_EPOCH_VALID,
                             export_path=EXPORT_BASE_PATH):
    data_generator = ImageDataGenerator(rescale=1./255)
    # Pixel values from 0-255 are rescaled to range (0, 1)
    training_set = data_generator.flow_from_directory(with_data + "training/",
                                                      target_size=(SIZE_X, SIZE_Y),
                                                      batch_size=BATCH_SIZE,
                                                      class_mode=CLASS_MODE,
                                                      color_mode=COLOR_MODE,
                                                      shuffle=SHUFFLE)
    validation_set = data_generator.flow_from_directory(with_data + "validation/",
                                                        target_size=(SIZE_X, SIZE_Y),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode=CLASS_MODE,
                                                        color_mode=COLOR_MODE,
                                                        shuffle=SHUFFLE)
    test_set = data_generator.flow_from_directory(with_data + "test/",
                                                  target_size=(SIZE_X, SIZE_Y),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode=CLASS_MODE,
                                                  color_mode=COLOR_MODE,
                                                  shuffle=SHUFFLE)
    # Start compiling
    print("> COMPILING MODEL: " + str(model_name))
    model.compile(optimizer=optimizer_to_be_used, loss=loss_function_to_use, metrics=['accuracy'])
    model.summary()
    history = model.fit(training_set, steps_per_epoch=epoch_steps_train, epochs=EPOCHS,
                        validation_data=validation_set, validation_steps=epoch_steps_valid)
    print("> DONE.")
    if save_model:
        # Save the model
        model.save(export_path + model_name + ".h5")
        print("> SAVED MODEL AS: " + export_path + model_name + ".h5")

    if with_metrics:
        print("> EXPORTING TRAINING / VALIDATION METRICS AS GRAPHS...")
        # Summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(model_name + ' Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.xlabel('Epoch')
        plt.xlim(0, EPOCHS)
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(export_path + model_name + "_Accuracy_Metrics.png", dpi=150, figsize=(8, 8))
        plt.clf()
        print("> EXPORTED: " + export_path + model_name + "_Accuracy_Metrics.png")
        # Summarize loss history
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(model_name + ' Loss')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.xlabel('Epoch')
        plt.xlim(0, EPOCHS)
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(export_path + model_name + "_Loss_Metrics.png", dpi=150, figsize=(8, 8))
        plt.clf()
        print("> EXPORTED: " + export_path + model_name + "_Loss_Metrics.png")
        # Export a csv with all training / validation metrics
        columns = ["Loss", "Accuracy", "Validation Loss", "Validation Accuracy"]
        csv_save_file = export_path + model_name + "_Training_Metrics.csv"
        print("> EXPORTING TRAINING METRICS AS: " + csv_save_file)
        with open(csv_save_file, 'a', newline='\n') as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow([columns[0], columns[1], columns[2], columns[3]])
            accuracy = []
            for acc in history.history["accuracy"]:
                accuracy.append(acc)
            validation_accuracy = []
            for val_acc in history.history["val_accuracy"]:
                validation_accuracy.append(val_acc)
            loss = []
            for loss_his in history.history["loss"]:
                loss.append(loss_his)
            validation_loss = []
            for val_loss in history.history["val_loss"]:
                validation_loss.append(val_loss)
            i = 0
            while i < len(accuracy):
                writer.writerow([loss[i], accuracy[i], validation_loss[i], validation_accuracy[i]])
                i += 1

    if with_predictions:
        print("> PREDICTIONS. TRAINING SET:")
        prediction_values_training_set = prediction_values_for_set(model, training_set)
        # CONFUSION MATRIX PREDICTIONS ON TRAINING DATA
        test_confusion_matrix = np.array([[prediction_values_training_set[9], prediction_values_training_set[8]],
                                          [prediction_values_training_set[7], prediction_values_training_set[6]]])
        ax = plt.subplot()
        sns.heatmap(test_confusion_matrix, annot=True, fmt='g', ax=ax, cmap='Greens')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(model_name + ' Training Confusion Matrix')
        ax.xaxis.set_ticklabels(['Cough', 'Noise'])
        ax.yaxis.set_ticklabels(['Cough', 'Noise'])
        plt.savefig(export_path + model_name + "_Training_Data_Confusion_Matrix.png", dpi=150, figsize=(8, 8))
        plt.clf()
        print("> PREDICTIONS. VALIDATION SET:")
        prediction_values_validation_set = prediction_values_for_set(model, validation_set)
        # CONFUSION MATRIX PREDICTIONS ON VALIDATION DATA
        test_confusion_matrix = np.array([[prediction_values_validation_set[9], prediction_values_validation_set[8]],
                                          [prediction_values_validation_set[7], prediction_values_validation_set[6]]])
        ax = plt.subplot()
        sns.heatmap(test_confusion_matrix, annot=True, fmt='g', ax=ax, cmap='Greens')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(model_name + ' Validation Confusion Matrix')
        ax.xaxis.set_ticklabels(['Cough', 'Noise'])
        ax.yaxis.set_ticklabels(['Cough', 'Noise'])
        plt.savefig(export_path + model_name + "_Validation_Data_Confusion_Matrix.png", dpi=150, figsize=(8, 8))
        plt.clf()
        print("> PREDICTIONS. TEST SET (UNSEEN DATA):")
        prediction_values_test_set = prediction_values_for_set(model, test_set)
        # CONFUSION MATRIX PREDICTIONS ON UNSEEN DATA
        test_confusion_matrix = np.array([[prediction_values_test_set[9], prediction_values_test_set[8]],
                                          [prediction_values_test_set[7], prediction_values_test_set[6]]])
        ax = plt.subplot()
        sns.heatmap(test_confusion_matrix, annot=True, fmt='g', ax=ax, cmap='Greens')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(model_name + ' Test Confusion Matrix')
        ax.xaxis.set_ticklabels(['Cough', 'Noise'])
        ax.yaxis.set_ticklabels(['Cough', 'Noise'])
        plt.savefig(export_path + model_name + "_Test_Data_Confusion_Matrix.png", dpi=150, figsize=(8, 8))
        plt.clf()
        print("> EXPORTED CONFUSION MATRICES FOR DATA SETS USING BASE PATH: " + export_path + model_name)
        # Export a csv with all prediction metrics
        columns = ["Set", "Accuracy", "Precision", "Recall", "F1", "Kappa", "AUC",
                   "True Negative", "False Positive", "False Negative", "True Positive"]
        csv_save_file = export_path + model_name + "_Prediction_Metrics.csv"
        print("> EXPORTING PREDICTION METRICS AS: " + csv_save_file)
        with open(csv_save_file, 'a', newline='\n') as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow([columns[0], columns[1], columns[2], columns[3], columns[4], columns[5], columns[6],
                             columns[7], columns[8], columns[9], columns[10]])
            writer.writerow(["Training", prediction_values_training_set[0], prediction_values_training_set[1],
                             prediction_values_training_set[2], prediction_values_training_set[3],
                             prediction_values_training_set[4], prediction_values_training_set[5],
                             prediction_values_training_set[6], prediction_values_training_set[7],
                             prediction_values_training_set[8], prediction_values_training_set[9]])
            writer.writerow(["Validation", prediction_values_validation_set[0], prediction_values_validation_set[1],
                             prediction_values_validation_set[2], prediction_values_validation_set[3],
                             prediction_values_validation_set[4], prediction_values_validation_set[5],
                             prediction_values_validation_set[6], prediction_values_validation_set[7],
                             prediction_values_validation_set[8], prediction_values_validation_set[9]])
            writer.writerow(["Test", prediction_values_test_set[0], prediction_values_test_set[1],
                             prediction_values_test_set[2], prediction_values_test_set[3],
                             prediction_values_test_set[4], prediction_values_test_set[5],
                             prediction_values_test_set[6], prediction_values_test_set[7],
                             prediction_values_test_set[8], prediction_values_test_set[9]])

    return model, history


###
# THE MODELS USED & TRAINED
###


###
# VERSION 1
# Baseline Sequential Model
# V1 (A1..., S1..., etc.)
###
def get_baseline_model():
    # Simple sequential model with 2 hidden layers
    model = Sequential()
    input_shape = (SIZE_X, SIZE_Y, 1)
    # 1st hidden layer
    model.add(Conv2D(4, kernel_size=3, activation='relu', input_shape=input_shape))
    # 2nd hidden layer
    model.add(Conv2D(4, kernel_size=3, activation='relu'))
    # Flatten
    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(256, activation='relu'))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    return model


###
# VERSION 2
# Introducing additional layers
# V2 (A2..., ...)
###
def get_a2_model():
    # Simple sequential model with 3 hidden layers and 2 fully connected dense layers
    model = Sequential()
    input_shape = (SIZE_X, SIZE_Y, 1)
    # 1st hidden layer
    model.add(Conv2D(4, kernel_size=3, activation='relu', input_shape=input_shape))
    # 2nd hidden layer
    model.add(Conv2D(4, kernel_size=3, activation='relu'))
    # 3rd hidden layer
    model.add(Conv2D(4, kernel_size=3, activation='relu'))
    # Flatten
    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(512, activation='relu'))
    # 2nd fully connected layer with less density
    model.add(Dense(256, activation='relu'))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    return model


###
# VERSION 2
# Introducing more complexity with filters
# V2 (S2..., ...)
###
def get_s2_model():
    # Simple sequential model with 2 hidden layers
    model = Sequential()
    input_shape = (SIZE_X, SIZE_Y, 1)
    # 1st hidden layer
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    # 2nd hidden layer
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # Flatten
    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(256, activation='relu'))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    return model


###
# VERSION 3
# Adding up to 128, 64, and 32 filters on the first 3 layers
# V3 (A3..., ...)
###
def get_a3_model():
    # Simple sequential model with 3 hidden layers and 2 fully connected dense layers
    model = Sequential()
    input_shape = (SIZE_X, SIZE_Y, 1)
    # 1st hidden layer
    model.add(Conv2D(128, kernel_size=3,  activation='relu', input_shape=input_shape))
    # 2nd hidden layer
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # 3rd hidden layer
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # Flatten
    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(512, activation='relu'))
    # 2nd fully connected layer with less density
    model.add(Dense(256, activation='relu'))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    return model


###
# VERSION 3
# Direct comparison with Adam optimizer in model variation V4
# Introduced 3 new layers (1x Convolutional with 16 filters, 1x 512 Node Dense layer, 1x MaxPooling)
# Changed hidden layers kernel size to 2x2
# V3 (S3..., ...)
###
def get_s3_model():
    # Simple sequential model with 3 hidden layers and 2 fully connected dense layers
    model = Sequential()
    input_shape = (SIZE_X, SIZE_Y, 1)
    # 1st hidden layer
    model.add(Conv2D(64, kernel_size=2,  activation='relu', input_shape=input_shape))
    # Add a pooling layer to improve generalization
    model.add(MaxPooling2D(2, 2))
    # 2nd hidden layer
    model.add(Conv2D(32, kernel_size=2, activation='relu'))
    # 3rd hidden layer
    model.add(Conv2D(16, kernel_size=2, activation='relu'))
    # Flatten
    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(512, activation='relu'))
    # 2nd fully connected layer with less density
    model.add(Dense(256, activation='relu'))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    return model


###
# VERSION 4
# Downsizing up to 64, 32, and 16 filters on the first 3 layers while reducing kernels to 2x2
# V4 (A4..., ...)
###
def get_a4_model():
    # Simple sequential model with 3 hidden layers and 2 fully connected dense layers
    model = Sequential()
    input_shape = (SIZE_X, SIZE_Y, 1)
    # 1st hidden layer
    model.add(Conv2D(64, kernel_size=2,  activation='relu', input_shape=input_shape))
    # 2nd hidden layer
    model.add(Conv2D(32, kernel_size=2, activation='relu'))
    # 3rd hidden layer
    model.add(Conv2D(16, kernel_size=2, activation='relu'))
    # Flatten
    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(512, activation='relu'))
    # 2nd fully connected layer with less density
    model.add(Dense(256, activation='relu'))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    return model


###
# VERSION 5
# Introducing MaxPooling2D layer
# V4 (A4..., ...)
###
def get_a5_model():
    # Simple sequential model with 3 hidden layers and 2 fully connected dense layers
    model = Sequential()
    input_shape = (SIZE_X, SIZE_Y, 1)
    # 1st hidden layer
    model.add(Conv2D(64, kernel_size=2,  activation='relu', input_shape=input_shape))
    # Add a pooling layer to improve generalization
    model.add(MaxPooling2D(2, 2))
    # 2nd hidden layer
    model.add(Conv2D(32, kernel_size=2, activation='relu'))
    # 3rd hidden layer
    model.add(Conv2D(16, kernel_size=2, activation='relu'))
    # Flatten
    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(512, activation='relu'))
    # 2nd fully connected layer with less density
    model.add(Dense(256, activation='relu'))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    return model


print("HYPER PARAMETERS USED IN THIS TRAINING SESSION: ")
print("EPOCHS: " + str(EPOCHS))
print("BATCH SIZE: " + str(BATCH_SIZE))
print("LEARNING RATE: " + str(LEARNING_RATE))
print("DECAY RATE (LR / EPOCHS): " + str(DECAY_RATE))
print("MOMENTUM (USED IN SDG): " + str(MOMENTUM))
print("TRAINING SAMPLES: " + str(TRAIN_SAMPLES) + " (" + str(TRAIN_SAMPLES_OVERLAPPED) + " overlapped)")
print("VALIDATION SAMPLES: " + str(VALID_SAMPLES) + " (" + str(VALID_SAMPLES_OVERLAPPED) + " overlapped)")
print("TEST SAMPLES: " + str(TEST_SAMPLES) + " (" + str(TEST_SAMPLES_OVERLAPPED) + "overlapped)")
print("STEPS PER EPOCH (TRAINING): " + str(STEPS_PER_EPOCH_TRAIN)
      + " (" + str(STEPS_PER_EPOCH_TRAIN_OVERLAPPED) + " overlapped)")
print("STEPS PER EPOCH (VALIDATION): " + str(STEPS_PER_EPOCH_VALID)
      + " (" + str(STEPS_PER_EPOCH_VALID_OVERLAPPED) + " overlapped)")


# ADAM OPTIMIZER (A-SERIES)
print("")
print("EVALUATING ADAM OPTIMIZER BASELINE MODEL ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="A1S", with_data=STANDARD_DATA,
                         export_path=EXPORT_BASE_PATH + "A1/")
print("")
print("EVALUATING NORM (N) DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="A1N", with_data=NORM_DATA,
                         export_path=EXPORT_BASE_PATH + "A1/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="A1R", with_data=NR_DATA,
                         export_path=EXPORT_BASE_PATH + "A1/")
print("")
print("EVALUATING NR DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="A1NR", with_data=NR_AND_NORM,
                         export_path=EXPORT_BASE_PATH + "A1/")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="A1O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A1/")
print("")
print("EVALUATING NO DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="A1NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A1/")
print("")
print("EVALUATING OR DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="A1OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A1/")
print("")
print("EVALUATING NOR DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="A1NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A1/")
print("")
print("EVALUATING M DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="A1M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         export_path=EXPORT_BASE_PATH + "A1/")
print("ALL DONE.")

print("")
print("EVALUATING ADAM OPTIMIZER V2 MODEL ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH A2 MODEL: ")
train_and_evaluate_model(get_a2_model(), model_name="A2S", with_data=STANDARD_DATA,
                         export_path=EXPORT_BASE_PATH + "A2/")
print("")
print("EVALUATING NORM (N) DATA SET WITH A2 MODEL: ")
train_and_evaluate_model(get_a2_model(), model_name="A2N", with_data=NORM_DATA,
                         export_path=EXPORT_BASE_PATH + "A2/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH A2 MODEL: ")
train_and_evaluate_model(get_a2_model(), model_name="A2R", with_data=NR_DATA,
                         export_path=EXPORT_BASE_PATH + "A2/")
print("")
print("EVALUATING NR DATA SET WITH A2 MODEL: ")
train_and_evaluate_model(get_a2_model(), model_name="A2NR", with_data=NR_AND_NORM,
                         export_path=EXPORT_BASE_PATH + "A2/")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH A2 MODEL: ")
train_and_evaluate_model(get_a2_model(), model_name="A2O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A2/")
print("")
print("EVALUATING NO DATA SET WITH A2 MODEL: ")
train_and_evaluate_model(get_a2_model(), model_name="A2NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A2/")
print("")
print("EVALUATING OR DATA SET WITH A2 MODEL: ")
train_and_evaluate_model(get_a2_model(), model_name="A2OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A2/")
print("")
print("EVALUATING NOR DATA SET WITH A2 MODEL: ")
train_and_evaluate_model(get_a2_model(), model_name="A2NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A2/")
print("")
print("EVALUATING M DATA SET WITH A2 MODEL: ")
train_and_evaluate_model(get_a2_model(), model_name="A2M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         export_path=EXPORT_BASE_PATH + "A2/")
print("ALL DONE.")

print("")
print("EVALUATING ADAM OPTIMIZER V3 MODEL ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH A3 MODEL: ")
train_and_evaluate_model(get_a3_model(), model_name="A3S", with_data=STANDARD_DATA,
                         export_path=EXPORT_BASE_PATH + "A3/")
print("")
print("EVALUATING NORM (N) DATA SET WITH A3 MODEL: ")
train_and_evaluate_model(get_a3_model(), model_name="A3N", with_data=NORM_DATA,
                         export_path=EXPORT_BASE_PATH + "A3/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH A3 MODEL: ")
train_and_evaluate_model(get_a3_model(), model_name="A3R", with_data=NR_DATA,
                         export_path=EXPORT_BASE_PATH + "A3/")
print("")
print("EVALUATING NR DATA SET WITH A3 MODEL: ")
train_and_evaluate_model(get_a3_model(), model_name="A3NR", with_data=NR_AND_NORM,
                         export_path=EXPORT_BASE_PATH + "A3/")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH A3 MODEL: ")
train_and_evaluate_model(get_a3_model(), model_name="A3O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A3/")
print("")
print("EVALUATING NO DATA SET WITH A3 MODEL: ")
train_and_evaluate_model(get_a3_model(), model_name="A3NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A3/")
print("")
print("EVALUATING OR DATA SET WITH A3 MODEL: ")
train_and_evaluate_model(get_a3_model(), model_name="A3OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A3/")
print("")
print("EVALUATING NOR DATA SET WITH A3 MODEL: ")
train_and_evaluate_model(get_a3_model(), model_name="A3NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A3/")
print("")
print("EVALUATING M DATA SET WITH A3 MODEL: ")
train_and_evaluate_model(get_a3_model(), model_name="A3M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         export_path=EXPORT_BASE_PATH + "A3/")
print("ALL DONE.")

print("")
print("EVALUATING ADAM OPTIMIZER V4 MODEL ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH A4 MODEL: ")
train_and_evaluate_model(get_a4_model(), model_name="A4S", with_data=STANDARD_DATA,
                         export_path=EXPORT_BASE_PATH + "A4/")
print("")
print("EVALUATING NORM (N) DATA SET WITH A4 MODEL: ")
train_and_evaluate_model(get_a4_model(), model_name="A4N", with_data=NORM_DATA,
                         export_path=EXPORT_BASE_PATH + "A4/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH A4 MODEL: ")
train_and_evaluate_model(get_a4_model(), model_name="A4R", with_data=NR_DATA,
                         export_path=EXPORT_BASE_PATH + "A4/")
print("")
print("EVALUATING NR DATA SET WITH A4 MODEL: ")
train_and_evaluate_model(get_a4_model(), model_name="A4NR", with_data=NR_AND_NORM,
                         export_path=EXPORT_BASE_PATH + "A4/")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH A4 MODEL: ")
train_and_evaluate_model(get_a4_model(), model_name="A4O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A4/")
print("")
print("EVALUATING NO DATA SET WITH A4 MODEL: ")
train_and_evaluate_model(get_a4_model(), model_name="A4NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A4/")
print("")
print("EVALUATING OR DATA SET WITH A4 MODEL: ")
train_and_evaluate_model(get_a4_model(), model_name="A4OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A4/")
print("")
print("EVALUATING NOR DATA SET WITH A4 MODEL: ")
train_and_evaluate_model(get_a4_model(), model_name="A4NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A4/")
print("")
print("EVALUATING NOR DATA SET WITH A4 MODEL: ")
train_and_evaluate_model(get_a4_model(), model_name="A4M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         export_path=EXPORT_BASE_PATH + "A4/")
print("ALL DONE.")

print("")
print("EVALUATING ADAM OPTIMIZER V5 MODEL ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH A5 MODEL: ")
train_and_evaluate_model(get_a5_model(), model_name="A5V2S", with_data=STANDARD_DATA,
                         export_path=EXPORT_BASE_PATH + "A5/")
print("")
print("EVALUATING NORM (N) DATA SET WITH A5 MODEL: ")
train_and_evaluate_model(get_a5_model(), model_name="A5V2N", with_data=NORM_DATA,
                         export_path=EXPORT_BASE_PATH + "A5/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH A5 MODEL: ")
train_and_evaluate_model(get_a5_model(), model_name="A5V2R", with_data=NR_DATA,
                         export_path=EXPORT_BASE_PATH + "A5/")
print("")
print("EVALUATING NR DATA SET WITH A5 MODEL: ")
train_and_evaluate_model(get_a5_model(), model_name="A5V2NR", with_data=NR_AND_NORM,
                         export_path=EXPORT_BASE_PATH + "A5/")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH A5 MODEL: ")
train_and_evaluate_model(get_a5_model(), model_name="A5V2O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A5/")
print("")
print("EVALUATING NO DATA SET WITH A5 MODEL: ")
train_and_evaluate_model(get_a5_model(), model_name="A5V2NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A5/")
print("")
print("EVALUATING OR DATA SET WITH A5 MODEL: ")
train_and_evaluate_model(get_a5_model(), model_name="A5V2OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A5/")
print("")
print("EVALUATING NOR DATA SET WITH A5 MODEL: ")
train_and_evaluate_model(get_a5_model(), model_name="A5V2NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "A5/")
print("")
print("EVALUATING M DATA SET WITH A5 MODEL: ")
train_and_evaluate_model(get_a5_model(), model_name="A5M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         export_path=EXPORT_BASE_PATH + "A5/")
print("ALL DONE.")

# SGD MODELS (S-SERIES)
print("")
print("EVALUATING SGD OPTIMIZER BASELINE MODEL ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="S1S", with_data=STANDARD_DATA,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S1/")
print("")
print("EVALUATING NORM (N) DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="S1N", with_data=NORM_DATA,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S1/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="S1R", with_data=NR_DATA,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S1/")
print("")
print("EVALUATING NR DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="S1NR", with_data=NR_AND_NORM,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S1/")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="S1O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "SGD/")
print("")
print("EVALUATING NO DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="S1NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S1/")
print("")
print("EVALUATING OR DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="S1OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S1/")
print("")
print("EVALUATING NOR DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="S1NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S1/")
print("")
print("EVALUATING M DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="S1M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S1/")
print("ALL DONE.")


print("")
print("EVALUATING SGD OPTIMIZER S2 ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH S2 MODEL: ")
train_and_evaluate_model(get_s2_model(), model_name="S2S", with_data=STANDARD_DATA,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S2/")
print("")
print("EVALUATING NORM (N) DATA SET WITH S2 MODEL: ")
train_and_evaluate_model(get_s2_model(), model_name="S2N", with_data=NORM_DATA,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S2/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH S2 MODEL: ")
train_and_evaluate_model(get_s2_model(), model_name="S2R", with_data=NR_DATA,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S2/")
print("")
print("EVALUATING NR DATA SET WITH S2 MODEL: ")
train_and_evaluate_model(get_s2_model(), model_name="S2NR", with_data=NR_AND_NORM,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S2/")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH S2 MODEL: ")
train_and_evaluate_model(get_s2_model(), model_name="S2O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S2/")
print("")
print("EVALUATING NO DATA SET WITH S2 MODEL: ")
train_and_evaluate_model(get_s2_model(), model_name="S2NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S2/")
print("")
print("EVALUATING OR DATA SET WITH S2 MODEL: ")
train_and_evaluate_model(get_s2_model(), model_name="S2OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S2/")
print("")
print("EVALUATING NOR DATA SET WITH S2 MODEL: ")
train_and_evaluate_model(get_s2_model(), model_name="S2NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S2/")
print("")
print("EVALUATING M DATA SET WITH S2 MODEL: ")
train_and_evaluate_model(get_s2_model(), model_name="S2M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S2/")
print("ALL DONE.")

print("")
print("EVALUATING SGD OPTIMIZER S3 ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH S3 AT 50 EPOCHS MODEL: ")
train_and_evaluate_model(get_s3_model(), model_name="S3V2S", with_data=STANDARD_DATA,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S3/")
print("")
print("EVALUATING NORM (N) DATA SET WITH S3 MODEL: ")
train_and_evaluate_model(get_s3_model(), model_name="S3V2N", with_data=NORM_DATA,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S3/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH S3 MODEL: ")
train_and_evaluate_model(get_s3_model(), model_name="S3V2R", with_data=NR_DATA,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S3/")
print("")
print("EVALUATING NR DATA SET WITH S3 MODEL: ")
train_and_evaluate_model(get_s3_model(), model_name="S3V2NR", with_data=NR_AND_NORM,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S3/")
print("")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH S3 MODEL: ")
train_and_evaluate_model(get_s3_model(), model_name="S3V2O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S3/")
print("")
print("EVALUATING NO DATA SET WITH S3 MODEL: ")
train_and_evaluate_model(get_s3_model(), model_name="S3V2NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S3/")
print("")
print("EVALUATING OR DATA SET WITH S3 MODEL: ")
train_and_evaluate_model(get_s3_model(), model_name="S3V2OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S3/")
print("")
print("EVALUATING NOR DATA SET WITH S3 MODEL: ")
train_and_evaluate_model(get_s3_model(), model_name="S3V2NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S3/")
print("")
print("EVALUATING M DATA SET WITH S3 MODEL: ")
train_and_evaluate_model(get_s3_model(), model_name="S3M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         optimizer_to_be_used=SGD_USED,
                         export_path=EXPORT_BASE_PATH + "S3/")
print("ALL DONE.")
