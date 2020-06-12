import os
import pandas as pd
import numpy as np
import librosa as lr

#####
cwd = os.getcwd()
full_path = os.path.realpath(__file__)

sr = 44100
emotion_dict = {
    "Laughter": "8",
    "PosedLaughter": "9",
    "SpeechLaughter": "10",
    "Speech": "x"
}
#####


def main():
    id = 0

    for root, dirs, files in os.walk(cwd):
        for file in files:
            if "mic" in file:
                df = pd.read_csv(os.path.join(root, 'laughterAnnotation.csv'))
                for index, row in df.iterrows():
                    t_start = row['Start Time (sec)']
                    t_end = row['End Time (sec)']
                    while t_start < t_end:
                        data, sample_rate = lr.load(os.path.join(root, file),
                                                    sr,
                                                    offset=t_start,
                                                    duration=1,
                                                    res_type='kaiser_fast')
                        try:
                            if emotion_dict[row['Type']] is not "x":
                                lr.output.write_wav('mahnob' + str(id) + '-' + emotion_dict[row['Type']] + '.wav',
                                                    np.array(data, dtype=float), sample_rate)
                        except OSError:
                            print('Error Creating wav file ')
                        id += 1
                        t_start += 1
    print("Extracted WAV´s:\t{}".format(id))


if __name__ == '__main__':
    main()
