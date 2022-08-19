import argparse
import os
import time
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import shutil
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import letterbox

def yolo2bb(box, dw, dh):
    class_id, x_center, y_center, w, h, _ = box.strip().split()
    x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
    x_center = round(x_center * dw)
    y_center = round(y_center * dh)
    w = round(w * dw)
    h = round(h * dh)
    x = round(x_center - w / 2)
    y = round(y_center - h / 2)

    return class_id, x, y, x+w, y+h


def draw_label(class_id, box, image):

    colors = {0: (0,255,255), 1: (0,255,0), 2: (255,0,255)}

    x_l, y_t, x_r, y_b = box

    # Line thickness of -1 px
    # Thickness of -1 will fill the entire shape
    thickness = 1

    image = cv2.rectangle(image, (int(x_l), int(y_t)), (int(x_r), int(y_b)), colors[int(class_id)], thickness)
    # cv2.imwrite('tmpc.png', image[y_t:y_b, x_l:x_r])
    cv2.imwrite('tmp.png', image)


def increase_bb(box, p, h, w):
    _, xmin, ymin, xmax, ymax = box

    xmin = max(0, xmin - p * (xmax - xmin))
    xmax = min(w, xmax + p * (xmax - xmin))
    ymin = max(0, ymin - p * (ymax - ymin))
    ymax = min(h, ymax + p * (ymax - ymin))

    return int(xmin), int(ymin), int(xmax), int(ymax)

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        # if '23-11-2021_03-48-47_PM.jpg' not in path:
        #     continue

        label_path = os.path.join('/'.join(str(path).split('/')[0:-1]), 'labels',
                                  str(path).split('/')[-1].replace('jpg','txt'))

        if not os.path.isfile(label_path):
            continue

        human_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                data = yolo2bb(line, im0s.shape[1], im0s.shape[0])
                # print(data)
                if data[0] != '1':
                    continue
                x_l, y_t, x_r, y_b = increase_bb(data, 0.15, im0s.shape[0], im0s.shape[1])
                data = data[0], x_l, y_t, x_r, y_b
                # draw_label(data[0], data[1:], im0s)
                human_labels.append(data)

                img_my = im0s[y_t:y_b, x_l:x_r]
                # Padded resize
                imgc = letterbox(img_my, imgsz, stride=stride)[0]
                # Convert
                imgc = imgc[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                imgc = np.ascontiguousarray(imgc)

                img = torch.from_numpy(imgc).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=opt.augment)[0]

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]
                t2 = time_synchronized()

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms,
                                           max_det=opt.max_det)
                t3 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image

                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                    imgh, imgw, _ = im0s.shape

                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_my.shape).round()
                        # print("rescale", det)
                        for k, d in enumerate(det):
                            # print('det', d[:4])
                            # new_bb = (imgw - x_l - d[1], imgh - y_t - d[2], imgw - x_r - d[3], imgh - y_b - d[4])
                            print(x_l + d[1])
                            new_bb = (x_l + d[0], y_t + d[1], x_l + d[2], y_t + d[3])
                            det[k][0] = new_bb[0]
                            det[k][1] = new_bb[1]
                            det[k][2] = new_bb[2]
                            det[k][3] = new_bb[3]
                        # print("rescale after", det)

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):

                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                            # print(f" The image with the result is saved in: {save_path}")

            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                #print(f"Results saved to {save_dir}{s}")

            print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
