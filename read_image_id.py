import os

import pandas as pd


CSV_PATH = "/home/user/Database/vinbigdata/test.csv"
SAVE_TXT = "./test.txt"

def read_image_id(save_txt=None):
    save_txt = save_txt if save_txt else SAVE_TXT
    df = pd.read_csv(CSV_PATH)
    df = pd.DataFrame(df)
    f = open(save_txt, 'w')
    for row in df.iterrows():
        row = row[1].to_dict()
        im_id = row["image_id"]
        line = f"{im_id}.jpg\n"
        f.write(line)
        print(line.strip("\n"))
    f.close()


if __name__ == "__main__":
    import sys
    save_txt = sys.argv[1] if len(sys.argv) > 1 else None
    read_image_id(save_txt)
