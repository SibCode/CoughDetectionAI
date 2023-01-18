# Cough Detection using Artificial Intelligence
A repository containing the code used to train multiple artificial intelligence models for cough detection and classification with different databases that had transformations applied to them. 

The scripts included:

| Scripts  | Description |
| ------------- | ------------- |
| estimateSNR.py  | Used to iterate over the audio samples to generate an estimation of the signal to noise ratio in Decibel for cough and noise samples.  |
| generateDatabases.py  | Used to validate the amount of audio samples, their length and generate spectrograms from them for all databases except the mixed database. Has settings to apply split (default 3:1:1) for training, validation and test set and the desired size of the spectrograms.  |
| generateMixedDatabase.py  | Used to copy the spectrograms generated for other databases into the mixed database.  |
| overlap_labels.py  | Loads exported audio labels from Audacity and applies an overlay with the values in setting to them so it be loaded into Audacity again for the export of overlapped labels.  |
| trainModelForCoughClassification.py  | Contains all the code and models used to export and train for cough classification (C1-C3). Also exports all metrics and training data with some plots and confusion matrices.  |
| trainModelForCoughDetection.py  | Contains all the code and models used to export and train for cough detection (A1-A5, S1-S3). Also exports metrics and data with some plots and confusion matrices.  |
| Predict_on_original_audio.py  | Uses the established cough detection model “A3M” to go through the full audio recordings and predict coughs within frames of that audio to detect them.  |

A guide on how to create mock-up data is included and can be read here: 
https://github.com/SibCode/CoughDetectionAI/blob/main/Guide_for_mock_up_data_generation.pdf

Further information:

The objective of the thesis associated with this repository was to test if a model can detect and categorize coughs with and without mask from a single microphone in a waiting room with lots of background noise. From 108 models a total of 28 achieved the targeted accuracy of over 75%. Two models trained with the dataset including all samples achieved accuracies over 98% for both cough detection and classification. The results only contain the generated data other than the models themselves. 
