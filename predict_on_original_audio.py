import pandas as pd
import numpy
import warnings
import librosa
import time
import skimage
from skimage import io
warnings.filterwarnings('ignore')
#Keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from pydub import AudioSegment
from os import path
from pathlib import Path
from datetime import datetime

# PATHS
MODEL_TO_LOAD = 'results/models/A3/A3M.h5'
ORIGINAL_DATA_TO_ANALYZE = 'data/audio/full_audio/Record_211025.wav'  # 'data/audio/full_audio/Record_190827.wav' # 'data/audio/full_audio/Record_211026.wav'  # 'data/audio/full_audio/Record_190807.wav'
PREDICTION_EXPORT_NAME = "Record_211025"  # "Record_190827"  # "Record_211026"  # "Record_190807"

# SETTINGS
SIZE_X = 100  # Based on image size generated
SIZE_Y = 64  # Based on image size generated
SEGMENT_DURATION = 2000  # milliseconds
SEGMENT_STEPS = 125  # milliseconds
NECESSARY_PREDECESSORS = 8  # necessary positive frames before a cough ist counted (here 9/16 of a 2 sec. segment)
BATCH_SIZE = 16
COLOR_MODE = 'grayscale'  # No fancy RGB, input spectrograms are greyscale
CLASS_MODE = 'binary'  # Spectrograms either contain a cough or not, so binary 0 / 1
PREDICTION_STEPS = 1
EXPECTED_COUGHS = 469  # 940 for 190827

SEGMENTS_EXPORT_PATH = 'data/analyze/currentSegments/'
SEGMENTS_ENDING = '.wav'
SEGMENTS_FORMAT = 'wav'  # Should be the same as used for training
SEGMENTS_SPECTROGRAM_PATH = 'data/analyze/currentSpectrograms/'
SEGMENTS_NAMING = 'segment'
UNPREDICTED_FOLDER = 'unpredicted/'


# MEL SPECTROGRAM SETTINGS
HOP_LENGTH = 441  # Number of samples per time-step in spectrogram
NUMBER_OF_MELS = 64  # Number of bins in spectrogram (Height)
TIME_STEPS = 100  # Number of time-steps (Width)


# STATISTICS AND EXPORT DATA
non_coughs_total = 0
non_coughs_loudness_total = 0
coughs_loudness_total = 0
current_time = 0
current_segments_loudness = [None] * BATCH_SIZE
current_segments_start_time = [None] * BATCH_SIZE
current_segments_duration = [None] * BATCH_SIZE
current_segments = [None] * BATCH_SIZE
prediciton_total_predictions_made = 0
prediction_total_cough_count = 0
predecessors_with_cough = 0

# PREDICTION CSVs
prediction_csv = 'data/analyze/predictions/' + PREDICTION_EXPORT_NAME + "_AI_predicted_coughs.csv"
predicted_noise = 'data/analyze/predictions/' + PREDICTION_EXPORT_NAME + "_AI_predicted_noises.csv"

# TO TRACK TIME
start_time_time = time.time()
start_time = datetime.now()
start_time_string = start_time.strftime("%H:%M:%S")

with open(prediction_csv, mode='w') as prediction_file:
    prediction_file.write('sep=,\n')
    prediction_file.write('Count,Start time (in seconds),Duration (in seconds), Loudness (Highest Amplitude)\n')

with open(predicted_noise, mode='w') as prediction_file:
    prediction_file.write('sep=,\n')
    prediction_file.write('Count,Start time (in seconds),Duration (in seconds), Loudness (Highest Amplitude)\n')


print("PREDICTION PROCESS STARTED")
print("PREDICTING FROM FULL AUDIO WITH PATH: " + str(ORIGINAL_DATA_TO_ANALYZE))
print("USING MODEL FROM PATH: " + str(MODEL_TO_LOAD))
print("LOADING MODEL...")
# Load model
model = load_model(MODEL_TO_LOAD)
# Summarize the model
print("USING FOLLOWING MODEL:")
model.summary()


# Analyzing audio
print("LOADING ORIGINAL AUDIO...")
audio_file_path = Path(ORIGINAL_DATA_TO_ANALYZE)
original_audio = AudioSegment.from_wav(audio_file_path)
y, sr = librosa.load(audio_file_path, mono=True)
original_max_time = int(librosa.get_duration(y=y, sr=sr)*1000)-1
print("AUDIO LENGTH: " + str(original_max_time))
done = False


# Spectrogram solution taken from:
# https://stackoverflow.com/questions/56719138/how-can-i-save-a-librosa-spectrogram-plot-as-a-specific-sized-image/57204349#57204349
def scale_min_max(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length * 2, hop_length=hop_length)
    mels = numpy.log(mels + 1e-8)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_min_max(mels, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0)  # Low frequencies at the bottom
    img = 255 - img  # Inverts image so that black (higher number) has more energy

    skimage.io.imsave(out, img)  # Save the image


while not done:
    print("CREATING NEXT AUDIO BATCH TO ANALYZE... (" + str(current_time) + "/" + str(original_max_time) + ", OR "
          + str(int(current_time / original_max_time * 100)) + "% DONE, "+str(prediction_total_cough_count)
          + " COUGHS DETECTED)")
    for i in range(0, BATCH_SIZE):
        if current_time == original_max_time or done:
            current_segments[i] = AudioSegment.silent(duration=SEGMENT_DURATION)
        elif SEGMENT_DURATION + current_time > original_max_time:
            done = True
            current_segments[i] = AudioSegment.silent(duration=SEGMENT_DURATION)
        else:
            current_segments[i] = original_audio[current_time:current_time + SEGMENT_DURATION]
        current_segments[i].export(SEGMENTS_EXPORT_PATH + SEGMENTS_NAMING + str(i) + SEGMENTS_ENDING,
                                   format=SEGMENTS_FORMAT)
        current_segments_start_time[i] = current_time/1000
        current_segments_duration[i] = SEGMENT_DURATION/1000
        current_segments_loudness[i] = current_segments[i].max
        current_time += SEGMENT_STEPS
        if current_time > original_max_time:
            current_time = original_max_time

    filesMissing = 0
    for i in range(0, BATCH_SIZE):
        fullPath = SEGMENTS_EXPORT_PATH + SEGMENTS_NAMING + str(i) + SEGMENTS_ENDING
        if not path.exists(fullPath):
            print("> ERROR: FILE '" + SEGMENTS_NAMING + str(i) + SEGMENTS_ENDING + "' DOES NOT EXIST.")
            filesMissing += 1
        else:
            # Load audio
            y, sr = librosa.load(fullPath, mono=True)
            # Our audio is sampled at 22'050 so 2 seconds is 44'100
            # Or: 441 * 100 for our default HOP_LENGTH * TIME_STEPS
            length_samples = HOP_LENGTH * TIME_STEPS
            window = y[0:0 + length_samples]
            out = SEGMENTS_SPECTROGRAM_PATH + UNPREDICTED_FOLDER + SEGMENTS_NAMING + str(i) + ".png"
            # Generate spectrogram
            spectrogram_image(window, sr=sr, out=out, hop_length=HOP_LENGTH, n_mels=NUMBER_OF_MELS)
    if filesMissing > 0:
        print("> ERROR: " + str(filesMissing) + " SEGMENTS MISSING!")

    # Rescale same way as during training (!)
    image_loader = ImageDataGenerator(rescale=1. / 255)

    # Since the current segments need to be generated first the prediction set is initiated new every time
    prediction_set = image_loader.flow_from_directory(SEGMENTS_SPECTROGRAM_PATH,
                                                      target_size=(SIZE_X, SIZE_Y),
                                                      batch_size=BATCH_SIZE,
                                                      color_mode=COLOR_MODE,
                                                      class_mode=CLASS_MODE,
                                                      shuffle=False)
    # prediction_set.reset()

    # Make predictions
    y_probability = model.predict_generator(prediction_set, steps=PREDICTION_STEPS, verbose=0)
    prediciton_total_predictions_made += BATCH_SIZE
    y_classes = []
    for probability in y_probability:
        y_classes.append((probability > 0.5).astype("int32"))  # Get predictions in binary

    nonCoughsDetected = 0
    coughsDetected = 0
    for i in range(0, BATCH_SIZE):
        if y_classes[i] == 1:  # Binary 1 represents no cough
            predecessors_with_cough = 0
            nonCoughsDetected += 1
            if current_segments_loudness[i] is not None and current_segments_duration[i] is not None \
                    and int(current_segments_loudness[i]) > 0 and int(current_segments_duration[i]) > 0:
                non_coughs_loudness_total = non_coughs_loudness_total + current_segments_loudness[i]
                with open(predicted_noise, mode='a') as prediction_file:
                    prediction_file.write(str(prediction_total_cough_count)+","+str(current_segments_start_time[i])+","
                                          + str(current_segments_duration[i])+","+str(current_segments_loudness[i])+"\n")
        else:
            if predecessors_with_cough >= NECESSARY_PREDECESSORS:
                predecessors_with_cough = 0
                if current_segments_loudness[i] is not None and current_segments_duration[i] is not None \
                        and int(current_segments_loudness[i]) > 0 and int(current_segments_duration[i]) > 0:
                    coughsDetected += 1
                    prediction_total_cough_count += 1
                    coughs_loudness_total = coughs_loudness_total + current_segments_loudness[i]
                    with open(prediction_csv, mode='a') as prediction_file:
                        prediction_file.write(str(prediction_total_cough_count) + ","
                                              + str(current_segments_start_time[i]) + ","
                                              + str(current_segments_duration[i])
                                              + "," + str(current_segments_loudness[i]) + "\n")
            else:
                predecessors_with_cough += 1

    if coughsDetected > 0:
        print("> NEW COUGHS DETECTED: " + str(coughsDetected) + " (" + str(prediction_total_cough_count) + " IN TOTAL)")
    non_coughs_total = non_coughs_total + nonCoughsDetected


end_time = datetime.now()
end_time_string = start_time.strftime("%H:%M:%S")
elapsed_time = time.time() - start_time_time
elapsed_time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print("#########")
print("ALL DONE!")
print("START TIME  = ", start_time_string)
print("ELAPSED TIME  = ", elapsed_time_string)
print("TOTAL PREDICTION STEPS: " + str(prediciton_total_predictions_made/BATCH_SIZE*PREDICTION_STEPS))
print("TOTAL PREDICTIONS CONSIDERED: " + str(prediciton_total_predictions_made*PREDICTION_STEPS))
print("TOTAL PREDICTIONS MADE: " + str(prediciton_total_predictions_made))
print("TOTAL COUGHS DETECTED: " + str(prediction_total_cough_count)
      + " (WITH " + str(EXPECTED_COUGHS) + " EXPECTED COUGHS FOR THIS FILE)")
print("TOTAL COUGHS/PREDICTION RATIO: " + str(int(prediction_total_cough_count/prediciton_total_predictions_made*100))
      + "%")
print("TOTAL NOISE FRAMES PREDICTED: " + str(non_coughs_total))
print("AVG NOISE LOUDNESS (MAX AMPLITUDE): " + str(int(non_coughs_loudness_total/non_coughs_total)))
print("AVG COUGH LOUDNESS (MAX AMPLITUDE): " + str(int(coughs_loudness_total/prediction_total_cough_count)))
print("#########")
print("REPEATING SINCE LOGS WILL BE FILLED DUE TO TENSORFLOW BEING VERBOSE...")
print("PREDICTED FROM FULL AUDIO WITH PATH: " + str(ORIGINAL_DATA_TO_ANALYZE))
print("USED MODEL FROM PATH: " + str(MODEL_TO_LOAD))
print("USED SEGMENTS STEP: " + str(SEGMENT_STEPS))
print("USED NECESSARY PREDECESSORS: " + str(NECESSARY_PREDECESSORS))
print("#########")
