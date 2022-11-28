import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt

from my_model_module import TRANSFORM, MyModelModule


def detect():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source')
    parser.add_argument('--weights')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    args = parser.parse_args()

    model = MyModelModule.load_from_checkpoint(args.weights)
    model.eval()
    model.freeze()

    with torch.no_grad():
        with Image.open(args.source) as image:
            inputs = TRANSFORM(image.convert('RGB')).unsqueeze(0)
            outputs = model(inputs)[0]
            
            for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
                if score < args.conf_thres:
                    continue

                plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red', linewidth=2))
                plt.gca().text(box[0], box[1], f'{label} {score:.2f}', bbox=dict(facecolor='blue', alpha=0.5))

            plt.imshow(image)
            plt.show()


if __name__ == '__main__':
    detect()
