import os
import pandas as pd
import numpy as np
import librosa as lr
import re

#####
cwd = os.getcwd()
full_path = os.path.realpath(__file__)

sr = 44100
emotion_dict = {
    "05": "0",
    "07": "1",
    "06": "2",
    "03": "3",
    "01": "4",
    "04": "5",
    "08": "6",
    "02": "7"
}
#####


def main():
    id = 0

    for root, dirs, files in os.walk(cwd):
        for file in files:
            if file.endswith(".wav") and not None:
                splits = file.split('-')

                t_start = 0
                base_file, samp_r = lr.load(os.path.join(root, file),
                                        sr,
                                        res_type='kaiser_fast')
                while lr.get_duration(base_file, samp_r) > t_start + 1:
                    data, sample_rate = lr.load(os.path.join(root, file),
                                                sr,
                                                offset=t_start,
                                                duration=1,
                                                res_type='kaiser_fast')
                    try:
                        lr.output.write_wav('ravdess' + str(id) + '-' + emotion_dict[splits[2]] + '.wav',
                                            np.array(data, dtype=float), sample_rate)
                    except OSError:
                        print('Error Creating wav file ')
                    id += 1
                    t_start += 1
    print("Extracted RAVDESS WAV´s")


if __name__ == '__main__':
    main()
