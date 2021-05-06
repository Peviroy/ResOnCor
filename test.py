import argparse
import torch
import torch.backends.cudnn as cudnn
from datasets import *
import numpy as np
import cv2
import os
import time

parser = argparse.ArgumentParser(description='Detection zoon')
parser.add_argument('-v', '--version', default='yolo', help='Support:yolo, fcos')
parser.add_argument('-d', '--dataset', default='voc', help='voc, coco-val.')
parser.add_argument('-size', '--input_size', default=416, type=int, help='input_size')
parser.add_argument('--trained_model',
                    default='weight/voc/',
                    type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.1, type=float, help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.50, type=float, help='NMS threshold')
parser.add_argument('--visual_threshold',
                    default=0.3,
                    type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False, help='use cuda.')

args = parser.parse_args()


def visualize(img,
              bboxes,
              scores,
              cls_inds,
              thresh,
              class_colors,
              class_names,
              class_indexs=None,
              dataset='voc'):
    if dataset == 'voc':
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                              class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin) - 20)), (int(xmax), int(ymin)),
                              class_colors[int(cls_indx)], -1)
                message = '%s' % (class_names[int(cls_indx)])
                cv2.putText(img, message, (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1)

    elif dataset == 'coco-val' and class_indexs is not None:
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                              class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin) - 20)), (int(xmax), int(ymin)),
                              class_colors[int(cls_indx)], -1)
                cls_id = class_indexs[int(cls_indx)]
                cls_name = class_names[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                message = '%s' % (cls_name)
                cv2.putText(img, message, (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1)

    return img


def test(net,
         device,
         testset,
         transform,
         thresh,
         class_colors=None,
         class_names=None,
         class_indexs=None,
         dataset='voc'):
    num_images = len(testset) # no dataloader for single image processing
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index + 1, num_images))
        img, _ = testset.pull_image(index)

        # transform
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        # forward
        t_start = time.time()
        bbox_pred, scores, cls_inds = net(x)
        print(f"Detection time used {time.time() - t_start:.4f}s")

        # scale each detection back up to the image
        H, W, _ = img.shape
        scale = np.array([[W, H, W, H]])
        bbox_pred *= scale # map the boxes to origin image scale

        img_processed = visualize(img, bbox_pred, scores, cls_inds, thresh, class_colors,
                                  class_names, class_indexs, dataset)
        #cv2.imshow('detection', img_processed)
        #cv2.waitKey(0)
        print('Saving the' + str(index) + '-th image ...')

        SAVING_DIR = 'test_images/' + args.dataset + '/'
        if not os.path.exists(SAVING_DIR):
            os.mkdir(SAVING_DIR)
        cv2.imwrite(SAVING_DIR + str(index).zfill(6) + '.jpg', img)


if __name__ == '__main__':
    if args.cuda and torch.cuda.is_available():
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = args.input_size

    # dataset
    if args.dataset == 'voc':
        print('Test on voc ...')
        CLASS_NAMES = VOC_CLASSES
        CLASS_INDEXS = None
        NUM_CLASSES = 20
        CLASS_COLORS = [(np.random.randint(255), np.random.randint(255), np.random.randint(255))
                        for _ in range(NUM_CLASSES)]
        testset = VOCDataset(root=VOC_ROOT, image_sets=[('2007', 'test')], transform=None)
    else:
        print('Unknow dataset. Only voc now')
        exit(0)

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

    model.load_state_dict(torch.load(args.trained_model, map_location=device))
    model.eval()
    model = model.to(device)
    print('Finished loading model!')

    # evaluation
    test(net=model,
         device=device,
         testset=testset,
         transform=transform,
         thresh=args.visual_threshold,
         class_colors=CLASS_COLORS,
         class_names=CLASS_NAMES,
         class_indexs=CLASS_INDEXS,
         dataset=args.dataset)
