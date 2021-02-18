import argparse
import os
import json
import pandas as pd


TEST_TXT = "../vinbigdata/test.txt"
CLS_PRED = "./results/classification_results.json"


def process_results(name, save_csv, use_cls=False, manual=False):
    if use_cls and not manual:
        cls_results = json.load(open(CLS_PRED, 'r'))  # read cls pred from .json
    detect_labels = f'./runs/detect/exp{name}/labels'
    images = open(TEST_TXT, 'r').readlines()
    im_ids = []
    dets = []
    for i, im in enumerate(images):
        print(f"process {i} th det pred.")
        # get image id and label path
        im_id = im.split(".")[0]
        im_ids.append(im_id)
        label = os.path.join(detect_labels, f'{im_id}.txt')
        # skip no detections image
        if not os.path.exists(label):
            det = f"14 1 0 0 1 1"
            dets.append(det)
            continue
        # read normal boxes
        det = ""
        with open(label, 'r') as l:
            lines = l.readlines()
            max_socre = 0
            for line in lines:
                det += (line.strip("\n") + " ")
                # using classifier to filter images
                # manual classification
                if use_cls and manual:
                    max_socre = max(max_socre, float(line.split(" ")[1]))
            if use_cls:
                print("using classifier...")
                # pred classification
                if not manual:
                    if float(cls_results[f"{im} 0"]) < 0.91:
                        det += f"14 1 0 0 1 1"
                # manual classification
                else:
                    if max_socre < 0.9 and max_socre >= 0.2:
                        det += f"14 1 0 0 1 1"
                    elif max_socre < 0.2:
                        det = f"14 1 0 0 1 1"

            dets.append(det)

    assert(len(im_ids) == len(dets))
    # convert to pandas.DataFrame and save to .csv
    df = pd.DataFrame({'image_id': im_ids, 'PredictionString': dets})
    df.to_csv(f"results/{save_csv}.csv", index=False, sep=',')
    print(f"pred is save to results/{save_csv}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="detect exp number, must be specific.")
    parser.add_argument("--csv", type=str, help="csv name to save, must be specific.")
    parser.add_argument("--cls", action="store_true", help="use classifier or not.")
    parser.add_argument("--manual", action="store_true", help="use manual classifier or not.")
    opt = parser.parse_args()

    process_results(opt.name, opt.csv, opt.cls, opt.manual)
