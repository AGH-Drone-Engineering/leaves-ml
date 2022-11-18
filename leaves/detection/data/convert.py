import pandas as pd
import re
import csv


IMAGES_PATH = 'leaf_detection/images/train'
LABELS_PATH = 'leaf_detection/labels/train'


def main():
    df = pd.read_csv('train.csv')

    with open('labels.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'))

        for image_id, width, height, bbox in df.values:
            x1, y1, w, h = map(int, re.match(r'\[(\d+), (\d+), (\d+), (\d+)\]', bbox).groups())
            x2 = x1 + w
            y2 = y1 + h

            writer.writerow((image_id, width, height, 'leaf', x1, y1, x2, y2))


if __name__ == '__main__':
    main()
