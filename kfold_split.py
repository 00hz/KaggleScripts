import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


warnings.filterwarnings('ignore')


TRAIN_CSV = r'/home/user/Database/vinbigdata/train.csv'
VOC_FOLDER = r'/home/user/Database/vinbigdata/voc_vinbigdata'
YOLO_FOLDER = r'/home/user/Database/vinbigdata/yolo_vinbigdata'
CLASSES = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
           "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
           "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
           "Pulmonary fibrosis", "No Finding"]


def kfold_split(num_folds, format="voc", train_csv=None):
    # split data to VOC format
    # set folders
    ori_train_csv = TRAIN_CSV if not train_csv else train_csv
    if format == "voc":
        dst_folder = r'/home/user/Database/vinbigdata/voc_vinbigdata/ImageSets/Main'  # voc format
    elif format == "yolo":
        dst_folder = YOLO_FOLDER
    else:
        raise NotImplementedError(f"{format} data format is not support yet.")

    mkp(dst_folder)
    # read annotations
    df = pd.read_csv(ori_train_csv)
    df = pd.DataFrame(df)
    df = df[df['class_name'] != 'No finding']  # filter not
    # split k-folds
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=101)
    df_folds = df[['image_id']].copy()

    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'object_count'] = df.groupby('image_id')['class_id'].nunique()

    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['object_count'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    # train:val = 4:1 to txt file
    df_folds.reset_index(inplace=True)
    for i in range(5):
        df_valid = pd.merge(df, df_folds[df_folds['fold'] == i], on='image_id')
        df_train = pd.merge(df, df_folds[df_folds['fold'] != i], on='image_id')
        val_set = set([r[1].to_dict()["image_id"] for r in df_valid.iterrows()])
        train_set = set([r[1].to_dict()["image_id"] for r in df_train.iterrows()])
        trainval_set = train_set | val_set
        print(len(trainval_set))
        val_txt = f"{dst_folder}/val{i}.txt"
        train_txt = f"{dst_folder}/train{i}.txt"
        trainval_txt = f"{dst_folder}/trainval{i}.txt"
        write2txt(val_txt, val_set)
        write2txt(train_txt, train_set)
        if not os.path.exists(trainval_txt):
            write2txt(trainval_txt, trainval_set)
        # create train and val txt


def write2txt(txtpath, image_set):
    f = open(txtpath, 'w')
    for image in image_set:
        image = f"./images/train/{image}.jpg\n"
        f.write(image)
    f.close()


def convert_anno_format(num_folds=5, format="voc", train_csv=None):
    # convert annotation from csv to voc format(.xml) or yolo format(.txt)
    ori_train_csv = TRAIN_CSV if not train_csv else train_csv
    df = pd.read_csv(ori_train_csv)
    annotations = {}  # {"image_id": [[class, x1, y1, x2, y2], [class, x1, y1, x2, y2], ...], ...}
    image_sizes = {}  # {"image_id": [w, h], ...}
    count = 1
    for row in df.iterrows():
        row = row[1].to_dict()
        image_id = row["image_id"]
        if image_id not in annotations.keys():
            annotations[image_id] = []
        annotation = [row["class_id"],
                      row["x_min"] if not np.isnan(row["x_min"]) else 0,
                      row["x_max"] if not np.isnan(row["x_max"]) else 1,
                      row["y_min"] if not np.isnan(row["y_min"]) else 0,
                      row["y_max"] if not np.isnan(row["y_max"]) else 1]
        image_size = [row["width"], row["height"]]
        annotations[image_id].append(annotation)
        image_sizes[image_id] = image_size
        print(f"read {count}th annotation.")
        count += 1
    assert(len(annotations) == len(image_sizes))
    if format == "voc":
        create_xml(annotations, image_sizes)
    elif format == "yolo":
        create_txt(annotations, image_sizes, num_folds=num_folds)


def create_xml(annotations, image_sizes, dst=None):
    from xml.dom import minidom
    dst_folder = VOC_FOLDER if not dst else dst
    anno_folder = os.path.join(dst_folder, "Annotations")
    mkp(anno_folder)
    count = 1
    for image_id, anno in annotations.items():
        xml_path = os.path.join(anno_folder, f"{image_id}.xml")
        w = image_sizes[image_id][0]
        h = image_sizes[image_id][1]
        # create xml data format
        doc = minidom.Document()
        annotation = doc.createElement('annotation')
        # folder node
        folder = doc.createElement('folder')
        folder.appendChild(doc.createTextNode('voc_vinbigdata'))
        # file name node
        filename = doc.createElement('filename')
        filename.appendChild(doc.createTextNode(f'{image_id}.jpg'))
        # size node
        size = doc.createElement('size')
        width = doc.createElement('width')
        width.appendChild(doc.createTextNode(str(w)))
        height = doc.createElement('height')
        height.appendChild(doc.createTextNode(str(h)))
        depth = doc.createElement('depth')
        depth.appendChild(doc.createTextNode(str(3)))
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        annotation.appendChild(folder)
        annotation.appendChild(filename)
        annotation.appendChild(size)
        # instance node
        for box in anno:
            instance = doc.createElement('object')
            pose = doc.createElement('pose')
            pose.appendChild(doc.createTextNode('Unspecified'))
            name = doc.createElement('name')
            name.appendChild(doc.createTextNode(CLASSES[box[0]]))
            x_min = doc.createElement('x_min')
            x_min.appendChild(doc.createTextNode(str(box[1])))
            x_max = doc.createElement('x_max')
            x_max.appendChild(doc.createTextNode(str(box[2])))
            y_min = doc.createElement('y_min')
            y_min.appendChild(doc.createTextNode(str(box[3])))
            y_max = doc.createElement('y_max')
            y_max.appendChild(doc.createTextNode(str(box[4])))
            truncated = doc.createElement('truncated')
            truncated.appendChild(doc.createTextNode(str(0)))
            difficult = doc.createElement('difficult')
            difficult.appendChild(doc.createTextNode(str(0)))
            bndbox = doc.createElement('bndbox')
            bndbox.appendChild(x_min)
            bndbox.appendChild(x_max)
            bndbox.appendChild(y_min)
            bndbox.appendChild(y_max)
            instance.appendChild(name)
            instance.appendChild(pose)
            instance.appendChild(bndbox)
            instance.appendChild(truncated)
            instance.appendChild(difficult)
            annotation.appendChild(instance)
        # add all nodes to doc
        doc.appendChild(annotation)
        # write to .xml file
        xmlf = open(xml_path, 'w')
        doc.writexml(xmlf, indent=" ", addindent="  ", newl="\n", encoding='utf-8')
        xmlf.close()
        print(f"write {count}th xml file.")
        count += 1


def create_txt(annotations, image_sizes, dst=None, num_folds=5):
    dst_folder = YOLO_FOLDER if not dst else dst
    anno_folder = os.path.join(os.path.join(dst_folder, "labels"), "all")
    mkp(anno_folder)
    count = 1
    for image_id, anno in annotations.items():
        xml_path = os.path.join(anno_folder, f"{image_id}.txt")
        w = image_sizes[image_id][0]
        h = image_sizes[image_id][1]
        # create yolo data format
        txtf = open(xml_path, 'w')
        for box in anno:
            cx = (box[2] + box[1]) / 2 / w
            cy = (box[4] + box[3]) / 2 / h
            width = (box[2] - box[1]) / w
            height = (box[4] - box[3]) / h
            line = f"{box[0]} {cx} {cy} {width} {height}\n"
            txtf.write(line)
        txtf.close()
        print(f"write {count}th txt file.")
        count += 1
    split_labels(num_folds)


def split_labels(num_folds):
    import shutil
    labal_path = r'/home/user/Database/vinbigdata/yolo_vinbigdata/labels/all'
    save_path = r'/home/user/Database/vinbigdata/yolo_vinbigdata/labels'
    for i in range(num_folds):
        train_fold = os.path.join(save_path, f"train{i}")
        vaild_fold = os.path.join(save_path, f"val{i}")
        mkp(train_fold)
        mkp(vaild_fold)
        trains = open(f"{YOLO_FOLDER}/train{i}.txt", 'r').readlines()
        vals = open((f"{YOLO_FOLDER}/val{i}.txt"), 'r').readlines()
        for img in trains:
            name = img.strip("\n").split("/")[-1].split(".")[0]
            src_path = f"{labal_path}/{name}.txt"
            dst_path = f"{train_fold}/{name}.txt"
            shutil.copyfile(src_path, dst_path)
        print(f"{i} th fold of train labels copy done!")
        for img in vals:
            name = img.strip("\n").split("/")[-1].split(".")[0]
            src_path = f"{labal_path}/{name}.txt"
            dst_path = f"{vaild_fold}/{name}.txt"
            shutil.copyfile(src_path, dst_path)
        print(f"{i} th fold of valid labels copy done!")


def mkp(p):
    if not os.path.exists(p):
        os.makedirs(p)


if __name__ == "__main__":
    # test functions
    # anno = {"test": [[0, 1, 2,3, 4], [2, 3,4,5,6]]}
    # sizes = {"test": [100, 100]}
    # dst = "/home/user/Database/vinbigdata"
    # create_txt(anno, sizes, dst)
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="split data or not.")
    parser.add_argument("--convert", action="store_true", help="apply convert function.")
    parser.add_argument("--folds", type=int, default=5, help="number of folds to split.")
    parser.add_argument("--format", type=str, default="voc", help="annotation format, voc/yolo.")
    parser.add_argument("--train-csv", type=str, default="", help="original train csv file path.")
    opt = parser.parse_args()

    if opt.split:
        kfold_split(opt.folds, opt.format, opt.train_csv)
    if opt.convert:
        convert_anno_format(opt.folds, opt.format, opt.train_csv)
