import argparse
import os
import sys

import json
import pandas as pd
import numpy as np


def csv2txt(csv_file, txt_folder):
    df = pd.read_csv(csv_file)
    class_stats = {}
    for i in range(15):
        class_stats[i] = 0
    for row in df.iterrows():
        row = row[1].to_dict()
        w = row['width']
        h = row['height']
        cx = (row['x_max'] + row['x_min']) / 2 / w
        cy = (row['y_max'] + row['y_min']) / 2 / h
        width = (row['x_max'] - row['x_min']) / w
        height = (row['y_max'] - row['y_min']) / h
        line = "{} {} {} {} {}\n".format(row['class_id'],
                                         cx if not np.isnan(cx) else 0,
                                         cy if not np.isnan(cy) else 0,
                                         width if not np.isnan(width) else 0,
                                         height if not np.isnan(height) else 0)
        f = open("{}/{}.txt".format(txt_folder, row['image_id']), "a")
        class_stats[row['class_id']] += 1
        f.write(line)
        f.close()
        print(line)
    print(class_stats)
    return class_stats


def count_images(file_folder, txt_file):
    file_list = os.listdir(file_folder)
    with_anno = 0
    wo_anno = 0
    for path in file_list:
        f_path = file_folder + "/" + path
        if not f_path.endswith(".txt"):
            continue
        f = open(f_path, 'r')
        lines = f.readlines()
        flag = True  # flag to image with/without annotations
        for line in lines:
            class_id = int(line.split(' ')[0])
            if class_id == 14:
                continue
            flag = False
        if flag:
            wo_anno += 1
        else:
            with_anno += 1
    f = open(txt_file, 'w')
    line = "image with boxes: {} \nimage without boxes: {}".format(with_anno, wo_anno)
    f.write(line)
    print(line)
    f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert", action="store_true", help="apply convert function.")
    parser.add_argument("--stats", action="stroe_true", help="stats data.")
    parser.add_argument("--csv", type=str, default="", help="csv file path.")
    parser.add_argument("--txt", type=str, default="", help="folder to save txt files.")
    opt = parser.parse_args()

    if opt.convert:
        # convert csv to txt
        csv_file = "/Users/linhezheng/Downloads/trainself.csv" if not opt.csv else opt.csv
        txt_folder = "/Users/linhezheng/Desktop/compete/train" if not opt.txt else opt.txt
        class_stats = csv2txt(csv_file, txt_folder)
    if opt.stats:
        # write class stats to local file
        class_stats = {0: 7162, 1: 279, 2: 960, 3: 5427, 4: 556, 5: 1000, 6: 1247, 7: 2483, 8: 2580, 9: 2203, 10: 2476, 11: 4842, 12: 226, 13: 4655, 14: 31818}
        json_file = "/Users/linhezheng/Desktop/compete/class_stats.json"
        json_data = json.dumps(class_stats)
        f = open(json_file, 'w')
        f.write(json_data)
        f.close()
        # count image with annotations and without annotations
        txt_folder = "/Users/linhezheng/Desktop/compete/train" if not opt.txt else opt.txt
        txt_file = "/Users/linhezheng/Desktop/compete/stats.txt"
        count_images(txt_folder, txt_file)
