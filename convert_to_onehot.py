
import numpy as np
import pandas as pd
from hyperparams import *
import csv

filename = "50000_15000"

def to_onehot(policy):
    out = ["0", "0", "0", "0", "0", "0", "0"]
    out[np.argmax(policy)] = "1"
    return out

if __name__ == "__main__":
    with open(filename + ".csv", "r", newline="\n") as csv_in:
        reader = csv.reader(csv_in, delimiter=',')
        with open(filename + "_onehot.csv", "w", newline="\n") as csv_out:
            writer = csv.writer(csv_out, delimiter=',')
            for i, row in enumerate(reader):
                if i != 0:
                    try:
                        row = row[0:42] + to_onehot(row[42:42+7]) + row[42+7:]
                    except Exception:
                        pass
                # print(row)
                writer.writerow(row)

