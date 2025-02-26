import string

import numpy as np
import os



inputFiles = ["../Datasets/robert_frost.txt"]
i = 0


for label,f in enumerate(inputFiles):
    print(f"{f} corresponds to labels {label}")
    for line in open(f):
        line = line.rstrip().lower()

        if line:
            line = line.translate(str.maketrans('','',string.punctuation))
            print(line)
            i += 1


print(i)