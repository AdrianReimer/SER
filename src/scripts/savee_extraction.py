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
    "a": "0",
    "d": "1",
    "f": "2",
    "h": "3",
    "n": "4",
    "sa": "5",
    "su": "6"
}
#####


def main():
    id = 0

    for root, dirs, files in os.walk(cwd):
        for file in files:
            if file.endswith(".wav") and not None:
                temp = re.compile("([a-zA-Z]+)([0-9]+)")
                res = temp.match(file).groups()

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
                        lr.output.write_wav('savee' + str(id) + '-' + emotion_dict[res[0]] + '.wav',
                                            np.array(data, dtype=float), sample_rate)
                    except OSError:
                        print('Error Creating wav file ')
                    id += 1
                    t_start += 1
    print("Extracted SAVEE WAV´s")


if __name__ == '__main__':
    main()
