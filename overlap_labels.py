import csv

###
# This script uses the exported labels from Audacity to generate overlap labels that can be exported
# using the Audacity Export / Multiple from labels feature to have frames that overlap.
# At default it generates 3 labels per existing label at 0.5 sec intervals starting from -0.5 to +0.5 seconds
# That equals an 75% overlap but guarantees that the cough remains in the frame as coughs are centered
###

# SETTINGS
SOURCE_PATH = "data/audio/labels/"  # Labels should be in source path named 'labels_YYMMDD.txt'
labels = ["190807", "190827", "211025", "211026"]  # Label dates to be exported
EXPORT_PATH = "data/audio/labels/"  # Export Path
INTERVAL_SEC = 0.5  # The interval to window the labels
START_AT_SEC = -0.5  # Starting offset from original start time
END_AT_SEC = 0.5  # Ending offset

print("MAKING NEW LABELS AT STARTING POINT " + str(START_AT_SEC) + " SEC. OFFSET WITH " + str(INTERVAL_SEC)
      + " SEC. INTERVALS TO GENERATE NEW LABELS UP TO END POINT AT " + str(END_AT_SEC) + " SECONDS.")
for label in labels:
    export_name = str(EXPORT_PATH) + 'labels_' + str(label) + '_overlapped.txt'
    export = open(export_name, 'w', encoding='UTF8', newline='')
    export_writer = csv.writer(export, delimiter='\t')
    source_name = str(SOURCE_PATH) + 'labels_' + str(label) + '.txt'
    print("GENERATING WINDOWED LABELS FOR " + str(source_name) + " TO " + str(export_name) + "...")
    with open(source_name) as data:
        data_reader = csv.reader(data, delimiter='\t')
        for line in data_reader:
            start_time = float(line[0])
            end_time = float(line[1])
            # The start time and end time are windowed in intervals between START_AT_SEC and END_AT_SEC
            current_offset = START_AT_SEC
            if INTERVAL_SEC <= 0:
                print("ERROR: INTERVAL SHOULD NOT BE NEGATIVE OR ZERO WHEN END_AT_SEC > START_AT_SEC")
                break
            while current_offset <= END_AT_SEC:
                export_writer.writerow([start_time + current_offset, end_time + current_offset, line[2]])
                current_offset = current_offset + INTERVAL_SEC
    export.close()

print("DONE.")
