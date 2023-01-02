# Load various libraries used
import numpy
import skimage
from skimage import io
import librosa
import matplotlib.pyplot as plt
import os
import glob
import shutil

# SETTINGS
TRAINING_SPLIT = 3  # Picks first X for training, then validation, then test
VALIDATION_SPLIT = 1 + TRAINING_SPLIT
TEST_SPLIT = 1 + VALIDATION_SPLIT
DESIRED_LENGTH_S = 2.0  # Ignoring samples that don't fit the length

# FOR THE VALIDATION (ADAPT IF SPLIT ADAPTED)
TRAINING_COUGH_SHOULD_BE = 1368
TRAINING_NOISE_SHOULD_BE = 1368
VALIDATION_COUGH_SHOULD_BE = 456
VALIDATION_NOISE_SHOULD_BE = 456
TEST_COUGH_SHOULD_BE = 454
TEST_NOISE_SHOULD_BE = 456
TRAINING_MASKED_SHOULD_BE = 585
TRAINING_UNMASKED_SHOULD_BE = 783
VALIDATION_MASKED_SHOULD_BE = 195
VALIDATION_UNMASKED_SHOULD_BE = 261
TEST_MASKED_SHOULD_BE = 194
TEST_UNMASKED_SHOULD_BE = 260

# DATABASES (FOLDER / FILE ENDINGS)
WITH_NONE = ""
WITH_OVERLAP = "with_overlap"
WITH_OVERLAP_AND_NORMALIZATION = "with_ol_and_norm"
WITH_OVERLAP_AND_NOISE_REDUCTION = "with_ol_and_nr"
WITH_OVERLAP_AND_NR_AND_NORM = "with_ol_nr_and_norm"
WITH_NOISE_REDUCTION = "with_noise_reduction"
WITH_NOISE_REDUCTION_AND_NORMALIZATION = "with_nr_and_norm"
WITH_NORMALIZATION = "with_norm"

# PATHS
UNMASKED_BASE_PATH = "data/audio/samples_"
UNMASKED_AUDIO_FOLDERS = ["190807", "190827"]
MASKED_BASE_PATH = "data/audio/samples_with_mask_"
MASKED_AUDIO_FOLDERS = ["211025", "211026"]
EXPORT_BASE_PATH = "data/spectrograms/"

# FOR THE UPDATES
TOTAL_EXPECTED_UNMASKED = 2610
TOTAL_EXPECTED_MASKED = 1963
UPDATE_EVERY_X = 250

# MEL SPECTROGRAM SETTINGS
HOP_LENGTH = 441  # Number of samples per time-step in spectrogram
NUMBER_OF_MELS = 64  # Number of bins in spectrogram (Height)
TIME_STEPS = 100  # Number of time-steps (Width)
out = ""
features = []


# Method to check if audio file has the right length
def is_right_length(filename):
    y, sr = librosa.load(filename, mono=True)
    length = librosa.get_duration(y=y, sr=sr)
    if length > DESIRED_LENGTH_S or length < DESIRED_LENGTH_S:
        return False
    return True


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


current_unmasked_cough_number = 1
current_masked_cough_number = 1
current_noise_number = 1


def export_spectrogram(filename, ep=EXPORT_BASE_PATH, masked_cough=False):
    if not is_right_length(filename):
        return
    y, sr = librosa.load(filename, mono=True)
    # Our audio is sampled at 22'050 so 2 seconds is 44'100
    # Or: 441 * 100 for our default HOP_LENGTH * TIME_STEPS
    length_samples = HOP_LENGTH * TIME_STEPS
    window = y[0:0 + length_samples]
    global current_masked_cough_number
    global current_unmasked_cough_number
    global current_noise_number
    if "cough" in filename:
        if masked_cough:
            out = ep + "/masked_coughs/cough_" + str(current_masked_cough_number) + ".png"
            current_masked_cough_number = current_masked_cough_number + 1
        else:
            out = ep + "/unmasked_coughs/unmasked_cough_" + str(current_unmasked_cough_number) + ".png"
            current_unmasked_cough_number = current_unmasked_cough_number + 1
        spectrogram_image(window, sr=sr, out=out, hop_length=HOP_LENGTH, n_mels=NUMBER_OF_MELS)
    elif "noise" in filename:
        out = ep + "/noise/noise_" + str(current_noise_number) + ".png"
        spectrogram_image(window, sr=sr, out=out, hop_length=HOP_LENGTH, n_mels=NUMBER_OF_MELS)
        current_noise_number = current_noise_number + 1


def generate_and_export_spectrograms_for_path(ep="",
                                              expected_unmasked=TOTAL_EXPECTED_UNMASKED,
                                              expected_masked=TOTAL_EXPECTED_MASKED):
    exp_base_path = EXPORT_BASE_PATH + ep
    if len(ep) > 1:
        ep = "_" + ep
    source_path = UNMASKED_BASE_PATH + "..." + ep + "/"
    print("STARTING GENERATION OF SPECTROGRAMS AND EXPORTING FROM " + str(source_path)
          + " TO PATH: " + str(exp_base_path) + "/...")
    print("> EXPORTING UNMASKED AUDIO FILES AS SPECTROGRAMS...")
    files_done = 1
    for folder in UNMASKED_AUDIO_FOLDERS:
        print("> EXPORTING FOR SOURCE: " + UNMASKED_BASE_PATH + folder + ep + "/")
        for filename in glob.glob(os.path.join(UNMASKED_BASE_PATH + folder + ep + "/", '*.wav')):
            export_spectrogram(filename, ep=exp_base_path, masked_cough=False)
            plt.clf()  # Clear plot to free cache
            if files_done % UPDATE_EVERY_X == 0:
                print('> EXPORTED: ', files_done, " OUT OF AN EXPECTED ", expected_unmasked, " NR. OF FILES.")
            files_done = files_done + 1
    print("> DONE.")
    print("> EXPORTING MASKED AUDIO FILES AS SPECTROGRAMS.")
    files_done = 1
    for folder in MASKED_AUDIO_FOLDERS:
        print("> EXPORTING FOR SOURCE: " + MASKED_BASE_PATH + folder + ep + "/")
        for filename in glob.glob(os.path.join(MASKED_BASE_PATH + folder + ep + "/", '*.wav')):
            export_spectrogram(filename, ep=exp_base_path,  masked_cough=True)
            plt.clf()  # Clear plot to free cache
            if files_done % UPDATE_EVERY_X == 0:
                print('> EXPORTED: ', files_done, " OUT OF AN EXPECTED ", expected_masked, " NR. OF FILES.")
            files_done = files_done + 1
    print("> DONE.")


def sort_and_split_for_path(ep=""):
    current_pick = 1
    file_number = 1
    exp_base_path = EXPORT_BASE_PATH + ep
    print("STARTING SORTING FOR PATH: " + str(exp_base_path) + "/...")
    print("> SORTING MASKED COUGHS...")
    if len(ep) > 1:
        ep = "/" + ep
    for masked_cough_spectrogram in glob.glob(os.path.join(exp_base_path + "/masked_coughs/", '*.png')):
        if current_pick <= TRAINING_SPLIT:
            shutil.copyfile(masked_cough_spectrogram,
                            EXPORT_BASE_PATH + "cough_detection" + ep
                            + "/training/cough/masked_cough_" + str(file_number) + ".png")
            shutil.copyfile(masked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_classification" + ep
                            + "/training/masked/masked_cough_" + str(
                                file_number) + ".png")
        elif current_pick <= VALIDATION_SPLIT:
            shutil.copyfile(masked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_detection" + ep
                            + "/validation/cough/masked_cough_" + str(file_number) + ".png")
            shutil.copyfile(masked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_classification" + ep
                            + "/validation/masked/masked_cough_" + str(
                                file_number) + ".png")
        elif current_pick <= TEST_SPLIT:
            shutil.copyfile(masked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_detection"
                            + ep + "/test/cough/masked_cough_" + str(file_number) + ".png")
            shutil.copyfile(masked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_classification" + ep
                            + "/test/masked/masked_cough_" + str(file_number) + ".png")
        current_pick = current_pick + 1
        file_number = file_number + 1
        if current_pick > TEST_SPLIT:
            current_pick = 1
    print("> DONE. A TOTAL OF " + str(file_number) + " MASKED COUGH SAMPLES WERE SORTED.")
    print("> SORTING UNMASKED COUGHS...")
    current_pick = 1
    file_number = 1
    for unmasked_cough_spectrogram in glob.glob(os.path.join(exp_base_path + "/unmasked_coughs/", '*.png')):
        if current_pick <= TRAINING_SPLIT:
            shutil.copyfile(unmasked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_detection" + ep
                            + "/training/cough/unmasked_cough_" + str(file_number) + ".png")
            shutil.copyfile(unmasked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_classification" + ep
                            + "/training/unmasked/unmasked_cough_" + str(
                                file_number) + ".png")
        elif current_pick <= VALIDATION_SPLIT:
            shutil.copyfile(unmasked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_detection" + ep
                            + "/validation/cough/unmasked_cough_" + str(file_number) + ".png")
            shutil.copyfile(unmasked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_classification" + ep
                            + "/validation/unmasked/unmasked_cough_" + str(
                                file_number) + ".png")
        elif current_pick <= TEST_SPLIT:
            shutil.copyfile(unmasked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_detection" + ep
                            + "/test/cough/unmasked_cough_" + str(file_number) + ".png")
            shutil.copyfile(unmasked_cough_spectrogram,
                            EXPORT_BASE_PATH + "/cough_classification" + ep
                            + "/test/unmasked/unmasked_cough_" + str(file_number) + ".png")
        current_pick = current_pick + 1
        file_number = file_number + 1
        if current_pick > TEST_SPLIT:
            current_pick = 1
    print("> DONE. A TOTAL OF " + str(file_number) + " UNMASKED COUGH SAMPLES WERE SORTED.")
    print("> SORTING NOISE SAMPLES...")
    current_pick = 1
    file_number = 1
    for noise_spectrogram in glob.glob(os.path.join(exp_base_path + "/noise/", '*.png')):
        if current_pick <= TRAINING_SPLIT:
            shutil.copyfile(noise_spectrogram,
                            EXPORT_BASE_PATH + "/cough_detection/" + ep
                            + "/training/noise/noise_" + str(file_number) + ".png")
        elif TRAINING_SPLIT < current_pick <= VALIDATION_SPLIT:
            shutil.copyfile(noise_spectrogram,
                            EXPORT_BASE_PATH + "/cough_detection/" + ep
                            + "/validation/noise/noise_" + str(file_number) + ".png")
        elif VALIDATION_SPLIT < current_pick <= TEST_SPLIT:
            shutil.copyfile(noise_spectrogram,
                            EXPORT_BASE_PATH + "/cough_detection/" + ep
                            + "/test/noise/noise_" + str(file_number) + ".png")
        current_pick = current_pick + 1
        file_number = file_number + 1
        if current_pick > TEST_SPLIT:
            current_pick = 1
    print("> DONE. A TOTAL OF " + str(file_number) + " NOISE SAMPLES WERE SORTED.")
    print("> ALL DONE.")


def validate_database(ep="", overlap_factor=1):
    exp_base_path = EXPORT_BASE_PATH + ep
    print("VALIDATING FOR PATH: " + str(exp_base_path) + "/...")
    miscounts = 0
    if len(ep) > 1:
        ep = "/" + ep
    file_number = 0
    for training_cough in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                                 + "/training/cough/", '*.png')):
        file_number = file_number + 1
    if file_number < TRAINING_COUGH_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "cough_detection" + ep
              + "/training/cough/")
        print("> SHOULD BE HIGHER THAN " + str(TRAINING_COUGH_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")
    file_number = 0
    for training_noise in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                                 + "/training/noise/", '*.png')):
        file_number = file_number + 1
    if file_number < TRAINING_NOISE_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "cough_detection" + ep
              + "/training/noise/")
        print("> SHOULD BE HIGHER THAN " + str(TRAINING_NOISE_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")
    file_number = 0
    for validation_cough in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                                   + "/validation/cough/", '*.png')):
        file_number = file_number + 1
    if file_number < VALIDATION_COUGH_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "cough_detection" + ep
              + "/validation/cough/")
        print("> SHOULD BE HIGHER THAN " + str(VALIDATION_COUGH_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")
    file_number = 0
    for validation_noise in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                                   + "/validation/noise/", '*.png')):
        file_number = file_number + 1
    if file_number < VALIDATION_NOISE_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "cough_detection" + ep
              + "/validation/noise/")
        print("> SHOULD BE HIGHER THAN " + str(VALIDATION_NOISE_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")
    file_number = 0
    for test_cough in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                             + "/test/cough/", '*.png')):
        file_number = file_number + 1
    if file_number < TEST_COUGH_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "cough_detection" + ep
              + "/test/cough/")
        print("> SHOULD BE HIGHER THAN " + str(TEST_COUGH_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")
    file_number = 0
    for test_noise in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                             + "/test/noise/", '*.png')):
        file_number = file_number + 1
    if file_number < TEST_NOISE_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "cough_detection" + ep
              + "/test/noise/")
        print("> SHOULD BE HIGHER THAN " + str(TEST_NOISE_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")

    file_number = 0
    for training_masked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                                  + "/training/masked/", '*.png')):
        file_number = file_number + 1
    if file_number < TRAINING_MASKED_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "/cough_classification" + ep
                                      + "/training/masked/")
        print("> SHOULD BE HIGHER THAN " + str(TRAINING_MASKED_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")
    file_number = 0
    for training_unmasked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                                  + "/training/unmasked/", '*.png')):
        file_number = file_number + 1
    if file_number < TRAINING_UNMASKED_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "/cough_classification" + ep
              + "/training/unmasked/")
        print("> SHOULD BE HIGHER THAN " + str(TRAINING_UNMASKED_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")
    file_number = 0
    for validation_masked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                                    + "/validation/masked/", '*.png')):
        file_number = file_number + 1
    if file_number < VALIDATION_MASKED_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "/cough_classification" + ep
              + "/validation/masked/")
        print("> SHOULD BE HIGHER THAN " + str(VALIDATION_MASKED_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")
    file_number = 0
    for validation_unmasked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                                      + "/validation/unmasked/", '*.png')):
        file_number = file_number + 1
    if file_number < VALIDATION_UNMASKED_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "/cough_classification" + ep
              + "/validation/unmasked/")
        print("> SHOULD BE HIGHER THAN " + str(VALIDATION_UNMASKED_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")
    file_number = 0
    for test_masked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                              + "/test/masked/", '*.png')):
        file_number = file_number + 1
    if file_number < TEST_MASKED_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "/cough_classification" + ep
              + "/test/masked/")
        print("> SHOULD BE HIGHER THAN " + str(TEST_MASKED_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")
    file_number = 0
    for test_unmasked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                                + "/test/unmasked/", '*.png')):
        file_number = file_number + 1
    if file_number < TEST_UNMASKED_SHOULD_BE * overlap_factor:
        miscounts = miscounts + 1
        print("> MISCOUNT FOR PATH: " + EXPORT_BASE_PATH + "/cough_classification" + ep
              + "/test/unmasked/")
        print("> SHOULD BE HIGHER THAN " + str(TEST_UNMASKED_SHOULD_BE * overlap_factor)
              + " BUT IS " + str(file_number) + "!")

    if miscounts == 0:
        print("> ALL DONE. NO COUNT UNDER MINIMUM FOR PATH: " + str(exp_base_path))
    else:
        print("> ALL DONE.")
        print("> A TOTAL OF " + str(miscounts) + " MISCOUNTS DETECTED.")
        print("> PLEASE CHECK FOR ERRORS FOR PATH: " + str(exp_base_path))


print("CURRENT DIRECTORY: " + str(os.getcwd()))
print("STARTING GENERATION & EXPORT FOR STANDARD DATABASE (1/8)...")
generate_and_export_spectrograms_for_path()
print("DONE WITH STANDARD DATABASE (1/8).")
print("STARTING GENERATION & EXPORT FOR NOISE REDUCTION DATABASE (2/8)...")
generate_and_export_spectrograms_for_path(WITH_NOISE_REDUCTION)
print("DONE WITH NOISE REDUCTION DATABASE (2/8).")
print("STARTING GENERATION & EXPORT FOR NOISE NORMALIZED DATABASE (3/8)...")
generate_and_export_spectrograms_for_path(WITH_NORMALIZATION)
print("DONE WITH NOISE NORMALIZED DATABASE (3/8).")
print("STARTING GENERATION & EXPORT FOR NORMALIZED & NOISE REDUCTION DATABASE (4/8)...")
generate_and_export_spectrograms_for_path(WITH_NOISE_REDUCTION_AND_NORMALIZATION)
print("DONE WITH NORMALIZED & NOISE REDUCTION DATABASE (4/8).")
print("STARTING GENERATION & EXPORT FOR OVERLAP DATABASE (5/8)...")
generate_and_export_spectrograms_for_path(WITH_OVERLAP,
                                          expected_masked=TOTAL_EXPECTED_MASKED*3,
                                          expected_unmasked=TOTAL_EXPECTED_UNMASKED*3)
print("DONE WITH OVERLAP DATABASE (5/8).")
print("STARTING GENERATION & EXPORT FOR OVERLAP & NORMALIZATION DATABASE (6/8)...")
generate_and_export_spectrograms_for_path(WITH_OVERLAP_AND_NORMALIZATION,
                                          expected_masked=TOTAL_EXPECTED_MASKED*3,
                                          expected_unmasked=TOTAL_EXPECTED_UNMASKED*3)
print("DONE WITH OVERLAP  & NORMALIZATION DATABASE (6/8).")
print("STARTING GENERATION & EXPORT FOR OVERLAP & NOISE REDUCTION DATABASE (7/8)...")
generate_and_export_spectrograms_for_path(WITH_OVERLAP_AND_NOISE_REDUCTION,
                                          expected_masked=TOTAL_EXPECTED_MASKED*3,
                                          expected_unmasked=TOTAL_EXPECTED_UNMASKED*3)
print("DONE WITH OVERLAP & NOISE REDUCTION DATABASE (7/8).")
print("STARTING GENERATION & EXPORT FOR OVERLAP & NOISE REDUCTION & NORMALIZED DATABASE (8/8)...")
generate_and_export_spectrograms_for_path(WITH_OVERLAP_AND_NR_AND_NORM,
                                          expected_masked=TOTAL_EXPECTED_MASKED*3,
                                          expected_unmasked=TOTAL_EXPECTED_UNMASKED*3)
print("DONE WITH OVERLAP & NOISE REDUCTION & NORMALIZED DATABASE (8/8).")

print("SORTING AND SPLITTING TRAINING / VALIDATION / TEST INTO DATABASE...")
sort_and_split_for_path()
sort_and_split_for_path(WITH_NOISE_REDUCTION)
sort_and_split_for_path(WITH_NORMALIZATION)
sort_and_split_for_path(WITH_NOISE_REDUCTION_AND_NORMALIZATION)
sort_and_split_for_path(WITH_OVERLAP)
sort_and_split_for_path(WITH_OVERLAP_AND_NORMALIZATION)
sort_and_split_for_path(WITH_OVERLAP_AND_NOISE_REDUCTION)
sort_and_split_for_path(WITH_OVERLAP_AND_NR_AND_NORM)
print("DONE.")
print("")

print("VALIDATING DATABASES BY FILE COUNT...")
validate_database()
validate_database(WITH_NOISE_REDUCTION)
validate_database(WITH_NORMALIZATION)
validate_database(WITH_NOISE_REDUCTION_AND_NORMALIZATION)
validate_database(WITH_OVERLAP, overlap_factor=3)
validate_database(WITH_OVERLAP_AND_NORMALIZATION, overlap_factor=3)
validate_database(WITH_OVERLAP_AND_NOISE_REDUCTION, overlap_factor=3)
validate_database(WITH_OVERLAP_AND_NR_AND_NORM, overlap_factor=3)
print("ALL DONE.")
