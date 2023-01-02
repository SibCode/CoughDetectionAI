# Load various libraries used
import os
import glob
import shutil

# DATABASES (FOLDER / FILE ENDINGS)
WITH_NONE = ""
WITH_OVERLAP = "with_overlap"
WITH_OVERLAP_AND_NORMALIZATION = "with_ol_and_norm"
WITH_OVERLAP_AND_NOISE_REDUCTION = "with_ol_and_nr"
WITH_OVERLAP_AND_NR_AND_NORM = "with_ol_nr_and_norm"
WITH_NOISE_REDUCTION = "with_noise_reduction"
WITH_NOISE_REDUCTION_AND_NORMALIZATION = "with_nr_and_norm"
WITH_NORMALIZATION = "with_norm"
MIXED = "mixed"

# PATHS
EXPORT_BASE_PATH = "data/spectrograms/"


def copy_to_mixed(ep=""):
    exp_base_path = EXPORT_BASE_PATH + ep
    print("COPYING TO MIXED FOR PATH: " + str(exp_base_path) + "/...")
    original_ep = ep
    if len(original_ep) > 1:
        original_ep = original_ep + "_"
    if len(ep) > 1:
        ep = "/" + ep
    file_number = 0
    for training_cough in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                                 + "/training/cough/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(training_cough,
                        EXPORT_BASE_PATH + "/cough_detection/" + MIXED + "/training/cough/cough_" +
                        original_ep + str(file_number) + ".png")
    file_number = 0
    for training_noise in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                                 + "/training/noise/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(training_noise,
                        EXPORT_BASE_PATH + "/cough_detection/" + MIXED + "/training/noise/noise_" +
                        original_ep + str(file_number) + ".png")
    file_number = 0
    for validation_cough in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                                   + "/validation/cough/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(validation_cough,
                        EXPORT_BASE_PATH + "/cough_detection/" + MIXED + "/validation/cough/cough_" +
                        original_ep + str(file_number) + ".png")
    file_number = 0
    for validation_noise in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                                   + "/validation/noise/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(validation_noise,
                        EXPORT_BASE_PATH + "/cough_detection/" + MIXED + "/validation/noise/noise_" +
                        original_ep + str(file_number) + ".png")
    file_number = 0
    for test_cough in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                             + "/test/cough/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(test_cough,
                        EXPORT_BASE_PATH + "/cough_detection/" + MIXED + "/test/cough/cough_" +
                        original_ep + str(file_number) + ".png")
    file_number = 0
    for test_noise in glob.glob(os.path.join(EXPORT_BASE_PATH + "cough_detection" + ep
                                             + "/test/noise/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(test_noise,
                        EXPORT_BASE_PATH + "/cough_detection/" + MIXED + "/test/noise/noise_" +
                        original_ep + str(file_number) + ".png")

    file_number = 0
    for training_masked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                                  + "/training/masked/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(training_masked,
                        EXPORT_BASE_PATH + "/cough_classification/" + MIXED + "/training/masked/masked_" +
                        original_ep + str(file_number) + ".png")
    file_number = 0
    for training_unmasked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                                    + "/training/unmasked/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(training_unmasked,
                        EXPORT_BASE_PATH + "/cough_classification/" + MIXED + "/training/unmasked/unmasked_" +
                        original_ep + str(file_number) + ".png")
    file_number = 0
    for validation_masked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                                    + "/validation/masked/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(validation_masked,
                        EXPORT_BASE_PATH + "/cough_classification/" + MIXED + "/validation/masked/masked_" +
                        original_ep + str(file_number) + ".png")
    file_number = 0
    for validation_unmasked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                                      + "/validation/unmasked/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(validation_unmasked,
                        EXPORT_BASE_PATH + "/cough_classification/" + MIXED + "/validation/unmasked/unmasked_" +
                        original_ep + str(file_number) + ".png")
    file_number = 0
    for test_masked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                              + "/test/masked/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(test_masked,
                        EXPORT_BASE_PATH + "/cough_classification/" + MIXED + "/test/masked/masked_" +
                        original_ep + str(file_number) + ".png")
    file_number = 0
    for test_unmasked in glob.glob(os.path.join(EXPORT_BASE_PATH + "/cough_classification" + ep
                                                + "/test/unmasked/", '*.png')):
        file_number = file_number + 1
        shutil.copyfile(test_unmasked,
                        EXPORT_BASE_PATH + "/cough_classification/" + MIXED + "/test/unmasked/unmasked_" +
                        original_ep + str(file_number) + ".png")
    print("> ALL DONE FOR PATH: " + str(exp_base_path))


print("COPYING DATABASES TO MIXED...")
copy_to_mixed()
copy_to_mixed(WITH_NOISE_REDUCTION)
copy_to_mixed(WITH_NORMALIZATION)
copy_to_mixed(WITH_NOISE_REDUCTION_AND_NORMALIZATION)
copy_to_mixed(WITH_OVERLAP)
copy_to_mixed(WITH_OVERLAP_AND_NORMALIZATION)
copy_to_mixed(WITH_OVERLAP_AND_NOISE_REDUCTION)
copy_to_mixed(WITH_OVERLAP_AND_NR_AND_NORM)
print("ALL DONE.")
