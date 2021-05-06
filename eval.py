import torch
import torch.nn as nn
from datasets import *
import argparse
from utils.vocapi_evaluator import VOCAPIEvaluator

parser = argparse.ArgumentParser(description='Detection zoon')
parser.add_argument('-v', '--version', default='yolo', help='Support:yolo, fcos')
parser.add_argument('-d', '--dataset', default='voc', help='voc, coco-val, coco-test.')
parser.add_argument('--trained_model',
                    type=str,
                    default='weights/final.pth',
                    help='Trained state_dict file path to open')
parser.add_argument('-size', '--input_size', default=416, type=int, help='input_size')
parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda')

args = parser.parse_args()


def evaluate(model, device, input_size, transform, dataset):
    if dataset == 'voc':
        evaluator = VOCAPIEvaluator(data_root=VOC_ROOT,
                                    img_size=input_size,
                                    device=device,
                                    transform=transform,
                                    labelmap=VOC_CLASSES,
                                    display=True)
    else:
        print('Unknow dataset. Only voc now')
        exit(0)
    evaluator.evaluate(model)


if __name__ == '__main__':
    if args.cuda and torch.cuda.is_available():
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    if args.dataset == 'voc':
        print('Eval on voc ...')
        NUM_CLASSES = 20
    else:
        print('Unknow dataset. Only voc now')
        exit(0)

    # input size
    input_size = args.input_size

    # build model
    model_cfg = train_cfg
    if args.version == 'yolo':
        from models.yolov1 import myYOLO
        model = myYOLO(device,
                       input_size=model_cfg['min_dim']['yolo'][0],
                       num_classes=NUM_CLASSES,
                       trainable=False)
        # Get transform
        from datasets import YOLOBaseTransform as BaseTransform
        transform = BaseTransform(model_cfg['min_dim']['yolo'][0])
    elif args.version == 'fcos':
        from models.tinyfcos import FCOS
        model = FCOS(device,
                     input_size=model_cfg['min_dim']['fcos'],
                     num_classes=NUM_CLASSES,
                     trainable=False)
        from datasets import FCOSBaseTransform as BaseTransform
        transform = BaseTransform(model_cfg['min_dim']['fcos'])
    else:
        print('We only support yolo and fcos for now.')
        exit()

    # load net
    model.load_state_dict(torch.load(args.trained_model, map_location=device))
    model.eval()
    print('Finished loading model!')
    model = model.to(device)

    # evaluation
    with torch.no_grad():
        evaluate(model, device, input_size, transform, args.dataset)
