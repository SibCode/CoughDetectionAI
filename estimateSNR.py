# Load various libraries used
import numpy as np
import librosa
import math
import os
import glob


# SETTINGS
DESIRED_LENGTH_S = 2.0  # Other lengths are ignored

# UPDATES
TOTAL_EXPECTED_UNMASKED = 2610
TOTAL_EXPECTED_MASKED = 1963
UPDATE_EVERY_X = 500

# DATABASES (FOLDER / FILE ENDINGS)
WITH_NONE = ""
WITH_NOISE_REDUCTION = "with_noise_reduction"
WITH_NOISE_REDUCTION_AND_NORMALIZATION = "with_nr_and_norm"
WITH_NORMALIZATION = "with_norm"

# PATHS
UNMASKED_BASE_PATH = "samples_"
UNMASKED_AUDIO_FOLDERS = ["190807", "190827"]
MASKED_BASE_PATH = "samples_with_mask_"
MASKED_AUDIO_FOLDERS = ["211025", "211026"]


# Method to check if audio file has the right length
def is_right_length(filename):
    y, sr = librosa.load(filename, mono=True)
    length = librosa.get_duration(y=y, sr=sr)
    if length > DESIRED_LENGTH_S or length < DESIRED_LENGTH_S:
        return False
    return True


def get_estimated_SNR(filename):
    if not is_right_length(filename):
        return
    y, sr = librosa.load(filename, mono=True)
    # Since our samples are 2 seconds we can take 2*sr
    # DESIRED_LENGTH_S has to be 2.0 or it loops /
    # only considers the first 2 seconds
    min = np.min(y[0:2*sr])
    avg = np.abs(y[0:2*sr]).mean()
    max = np.max(y[0:2*sr])
    return min, avg, max


def get_dB_from_amplitude(amplitude, highest_avg_amplitude):
    return 20.0 * math.log10(amplitude / highest_avg_amplitude)


def get_values_for_path(ep=""):
    if len(ep) > 1:
        ep = "_" + ep
    files_done = 1
    avg_values_masked_cough = []
    max_values_masked_cough = []
    avg_values_noise = []
    max_values_noise = []
    avg_values_unmasked_cough = []
    max_values_unmasked_cough = []
    for folder in UNMASKED_AUDIO_FOLDERS:
        print("> GET SNR VALUES FOR UNMASKED SOURCE: " + UNMASKED_BASE_PATH + folder + ep + "/")
        for filename in glob.glob(os.path.join(UNMASKED_BASE_PATH + folder + ep + "/", '*.wav')):
            if not is_right_length(filename):
                continue
            min, avg, max = get_estimated_SNR(filename)
            if "cough" in filename:
                avg_values_unmasked_cough.append(avg)
                max_values_unmasked_cough.append(max)
            else:
                avg_values_noise.append(avg)
                max_values_noise.append(max)
            if files_done % UPDATE_EVERY_X == 0:
                print('> LOOKED AT: ', files_done, " OUT OF AN EXPECTED ", TOTAL_EXPECTED_UNMASKED, " NR. OF FILES.")
            files_done = files_done + 1
    files_done = 1
    for folder in MASKED_AUDIO_FOLDERS:
        print("> GET SNR VALUES FOR MASKED SOURCE: " + MASKED_BASE_PATH + folder + ep + "/")
        for filename in glob.glob(os.path.join(MASKED_BASE_PATH + folder + ep + "/", '*.wav')):
            if not is_right_length(filename):
                continue
            min, avg, max = get_estimated_SNR(filename)
            if "cough" in filename:
                avg_values_masked_cough.append(avg)
                max_values_masked_cough.append(max)
            else:
                avg_values_noise.append(avg)
                max_values_noise.append(max)
            if files_done % UPDATE_EVERY_X == 0:
                print('> LOOKED AT: ', files_done, " OUT OF AN EXPECTED ", TOTAL_EXPECTED_MASKED, " NR. OF FILES.")
            files_done = files_done + 1

    highest_avg_amplitude = np.average(max_values_masked_cough)
    if highest_avg_amplitude < np.average(max_values_unmasked_cough):
        highest_avg_amplitude = np.average(max_values_unmasked_cough)
    if highest_avg_amplitude < np.average(max_values_noise):
        highest_avg_amplitude = np.average(max_values_noise)

    print("DONE.")
    print("MAX VALUES:")
    print("AVG MASKED COUGH MAX DB: " + str(get_dB_from_amplitude(np.average(max_values_masked_cough),
                                                                  highest_avg_amplitude)))
    print("AVG MASKED COUGH MAX AMPL.: " + str(np.average(max_values_masked_cough)))
    print("AVG UNMASKED COUGH MAX DB: " + str(get_dB_from_amplitude(np.average(max_values_unmasked_cough),
                                                                    highest_avg_amplitude)))
    print("AVG UNMASKED COUGH MAX AMPL.: " + str(np.average(max_values_unmasked_cough)))
    print("AVG NOISE MAX DB: " + str(get_dB_from_amplitude(np.average(max_values_noise),
                                                           highest_avg_amplitude)))
    print("AVG NOISE MAX AMPL.: " + str(np.average(max_values_noise)))
    print("AVERAGE AVERAGE:")
    print("AVG MASKED COUGH AVG DB: " + str(get_dB_from_amplitude(np.average(avg_values_masked_cough),
                                                                  highest_avg_amplitude)))
    print("AVG MASKED COUGH AVG AMPL.: " + str(np.average(avg_values_masked_cough)))
    print("AVG UNMASKED COUGH AVG DB: " + str(get_dB_from_amplitude(np.average(avg_values_unmasked_cough),
                                                                    highest_avg_amplitude)))
    print("AVG UNMASKED COUGH AVG AMPL.: " + str(np.average(avg_values_unmasked_cough)))
    print("AVG NOISE AVG DB: " + str(get_dB_from_amplitude(np.average(avg_values_noise),
                                                           highest_avg_amplitude)))
    print("AVG NOISE AVG AMPL.: " + str(np.average(avg_values_noise)))


get_values_for_path()
