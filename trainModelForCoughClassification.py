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
# FOLLOWED BY A NUMBER INDICATOR FOR ITS ITERATION / VERSION
# STANDARD = S
# NORMALIZATION = N
# OVERLAP = O
# NOISE REDUCTION = R

# SETTINGS
SIZE_X = 100  # Based on image size generated
SIZE_Y = 64  # Based on image size generated
CLASS_NAMES = ["Unmasked", "Masked"]
TRAIN_SAMPLES = 1368
VALID_SAMPLES = 456
TEST_SAMPLES = 454
TRAIN_SAMPLES_OVERLAPPED = 4101
VALID_SAMPLES_OVERLAPPED = 1366
TEST_SAMPLES_OVERLAPPED = 1366
TRAIN_SAMPLES_MIXED = 23558
VALID_SAMPLES_MIXED = 7848
TEST_SAMPLES_MIXED = 7840

# PATHS
BASE_SOURCE_PATH = 'data/spectrograms/cough_classification/'
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
EPOCHS = 50
PATIENCE = 100
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
COLOR_MODE = 'grayscale'  # No fancy RGB
# For more than 2 categories / classes class mode would be 'categorical'
CLASS_MODE = 'binary'  # Classes using categories "masked" / "unmasked" with 2 we can still use binary
# For more than 2 categories loss function would be tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
LOSS_FUNCTION = "binary_crossentropy"


# Metrics (same for classification)
def prediction_values_for_set(model, set):
    y_probability = model.predict(set, verbose=0)
    # Multi-Class classification with softmax gives multiple % for each class as predictions, so max would be used like:
    # y_classes = [np.argmax(y_probability)]
    # Since we only have 2 (masked / unmasked) however, we can still categorize binary
    y_classes = []
    for probability in y_probability:
        y_classes.append((probability > 0.5).astype("int32"))
    # Reduced to 1d array as scikit-learn metrics expects 1D
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
        ax.xaxis.set_ticklabels(CLASS_NAMES)
        ax.yaxis.set_ticklabels(CLASS_NAMES)
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
        ax.xaxis.set_ticklabels(CLASS_NAMES)
        ax.yaxis.set_ticklabels(CLASS_NAMES)
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
        ax.xaxis.set_ticklabels(CLASS_NAMES)
        ax.yaxis.set_ticklabels(CLASS_NAMES)
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
# THE MODELS USED
###

def get_baseline_model():
    # Simple sequential model with 2 hidden layers
    model = Sequential()
    input_shape = (SIZE_X, SIZE_Y, 1)
    # 1st hidden layer
    model.add(Conv2D(4, 3, activation='relu', input_shape=input_shape))
    # 2nd hidden layer
    model.add(Conv2D(4, 3, activation='relu'))
    # Flatten
    model.add(Flatten())
    # Add fully connected layer.
    model.add(Dense(256, activation='relu'))
    # Output layer with two end points for categorical would be Dense using softmax like:
    # model.add(Dense(2, activation='softmax'))
    # But since we only have 2 we can use binary
    model.add(Dense(1, activation='sigmoid'))
    return model


###
# VERSION 2
# Adding two layers (1x Convolutional, 1x Dense) then using 64, 32, and 16 filters while reducing kernels to 2x2
# V2 (C2..., ...)
###
def get_c2_model():
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
    # Output layer with binary node for only two classes (masked / unmasked)
    model.add(Dense(1, activation='sigmoid'))
    return model


###
# VERSION 3
# Adding MaxPooling
# V3 (C3..., ...)
###
def get_c3_model():
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
    # Output layer with binary node for only two classes (masked / unmasked)
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
#
print("")
print("EVALUATING CLASSIFIER BASELINE MODEL ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="C1S", with_data=STANDARD_DATA,
                         export_path=EXPORT_BASE_PATH + "C1/")
print("")
print("EVALUATING NORM (N) DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="C1N", with_data=NORM_DATA,
                         export_path=EXPORT_BASE_PATH + "C1/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="C1R", with_data=NR_DATA,
                         export_path=EXPORT_BASE_PATH + "C1/")
print("")
print("EVALUATING NR DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="C1NR", with_data=NR_AND_NORM,
                         export_path=EXPORT_BASE_PATH + "C1/")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="C1O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C1/")
print("")
print("EVALUATING NO DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="C1NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C1/")
print("")
print("EVALUATING OR DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="C1OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C1/")
print("")
print("EVALUATING NOR DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="C1NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C1/")
print("")
print("EVALUATING M DATA SET WITH BASELINE MODEL: ")
train_and_evaluate_model(get_baseline_model(), model_name="C1M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         export_path=EXPORT_BASE_PATH + "C1/")
print("ALL DONE.")

print("EVALUATING CLASSIFIER V2 MODEL ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH V2 MODEL: ")
train_and_evaluate_model(get_c2_model(), model_name="C2S", with_data=STANDARD_DATA,
                         export_path=EXPORT_BASE_PATH + "C2/")
print("")
print("EVALUATING NORM (N) DATA SET WITH V2 MODEL: ")
train_and_evaluate_model(get_c2_model(), model_name="C2N", with_data=NORM_DATA,
                         export_path=EXPORT_BASE_PATH + "C2/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH V2 MODEL: ")
train_and_evaluate_model(get_c2_model(), model_name="C2R", with_data=NR_DATA,
                         export_path=EXPORT_BASE_PATH + "C2/")
print("")
print("EVALUATING NR DATA SET WITH V2 MODEL: ")
train_and_evaluate_model(get_c2_model(), model_name="C2NR", with_data=NR_AND_NORM,
                         export_path=EXPORT_BASE_PATH + "C2/")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH V2 MODEL: ")
train_and_evaluate_model(get_c2_model(), model_name="C2O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C2/")
print("")
print("EVALUATING NO DATA SET WITH V2 MODEL: ")
train_and_evaluate_model(get_c2_model(), model_name="C2NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C2/")
print("")
print("EVALUATING OR DATA SET WITH V2 MODEL: ")
train_and_evaluate_model(get_c2_model(), model_name="C2OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C2/")
print("")
print("EVALUATING NOR DATA SET WITH V2 MODEL: ")
train_and_evaluate_model(get_c2_model(), model_name="C2NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C2/")
print("")
print("EVALUATING M DATA SET WITH V2 MODEL: ")
train_and_evaluate_model(get_c2_model(), model_name="C2M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         export_path=EXPORT_BASE_PATH + "C2/")
print("ALL DONE.")


print("EVALUATING CLASSIFIER V3 MODEL ON DIFFERENT DATA SETS...")
print("")
print("EVALUATING STANDARD DATA SET WITH V3 MODEL: ")
train_and_evaluate_model(get_c3_model(), model_name="C3S", with_data=STANDARD_DATA,
                         export_path=EXPORT_BASE_PATH + "C3/")
print("")
print("EVALUATING NORM (N) DATA SET WITH V3 MODEL: ")
train_and_evaluate_model(get_c3_model(), model_name="C3N", with_data=NORM_DATA,
                         export_path=EXPORT_BASE_PATH + "C3/")
print("")
print("EVALUATING REDUCTION (R) DATA SET WITH V3 MODEL: ")
train_and_evaluate_model(get_c3_model(), model_name="C3R", with_data=NR_DATA,
                         export_path=EXPORT_BASE_PATH + "C3/")
print("")
print("EVALUATING NR DATA SET WITH V3 MODEL: ")
train_and_evaluate_model(get_c3_model(), model_name="C3NR", with_data=NR_AND_NORM,
                         export_path=EXPORT_BASE_PATH + "C3/")
print("")
print("EVALUATING OVERLAP (O) DATA SET WITH V3 MODEL: ")
train_and_evaluate_model(get_c3_model(), model_name="C3O", with_data=OVERLAP_DATA,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C3/")
print("")
print("EVALUATING NO DATA SET WITH V3 MODEL: ")
train_and_evaluate_model(get_c3_model(), model_name="C3NO", with_data=OL_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C3/")
print("")
print("EVALUATING OR DATA SET WITH V3 MODEL: ")
train_and_evaluate_model(get_c3_model(), model_name="C3OR", with_data=OL_AND_NR,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C3/")
print("")
print("EVALUATING NOR DATA SET WITH V3 MODEL: ")
train_and_evaluate_model(get_c3_model(), model_name="C3NOR", with_data=OL_NR_AND_NORM,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_OVERLAPPED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_OVERLAPPED,
                         export_path=EXPORT_BASE_PATH + "C3/")
print("")
print("EVALUATING M DATA SET WITH V3 MODEL: ")
train_and_evaluate_model(get_c3_model(), model_name="C3M", with_data=MIXED,
                         epoch_steps_train=STEPS_PER_EPOCH_TRAIN_MIXED,
                         epoch_steps_valid=STEPS_PER_EPOCH_VALID_MIXED,
                         export_path=EXPORT_BASE_PATH + "C3/")
print("ALL DONE.")
