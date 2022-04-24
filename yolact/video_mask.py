import os, sys
import argparse
import numpy as np
import cv2
import shutil
import yaml

import torch
import torch.backends.cudnn as cudnn

from yolact import Yolact
from data import set_cfg, cfg
from utils.functions import SavePath
from utils.augmentations import FastBaseTransform
from eval import prep


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='')
parser.add_argument('--yolact_weight', default="/content/grive/MyDrive/Colab/yolact_resnet50_54_800000.pth",  type=str)
parser.add_argument('--savevid', default=True, type=bool)
parser.add_argument('--video', type=str, default=None, help='Save video .avi form input')
parser.add_argument('--videos', type=str, default=None, help='Save folder video .avi form folder input')
parser.add_argument('--fast_nms', default=True, type=bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
parser.add_argument('--cross_class_nms', default=False, type=bool,
                        help='Whether compute NMS cross-class or per-class.')
parser.add_argument('--task', default='mask', type=str,
                        help='video mask binary or video segment')
args = parser.parse_args()


def video2frame(path: str):
    try:
        filename = path.split('/')[-1]
        name, suffix_path = filename.split('.')
        dist_path = os.path.join(os.getcwd(), name)
        if suffix_path in VID_FORMATS:
            if not os.path.isdir(dist_path):
                os.makedirs(dist_path)

            vid = cv2.VideoCapture(path)

            if not vid.isOpened():
                print('Could not open video "%s"' % path)
                exit(-1)
            frame = None
            idx_frame = 1
            while True:
                try:
                    is_success, frame = vid.read()
                except cv2.error:
                    continue

                if not is_success:
                    break

                cv2.imwrite(dist_path + '/' + str(idx_frame) + '.png', frame)
                idx_frame += 1
                # OPTIONAL: show last image
            vid.release()
        else:
            raise TypeError
    except OSError as e:
        print(e.errno)

# TODO: make config as a parameter instead of using a global parameter from yolact.data
def config_Yolact(yolact_weight):
    # Load config from weight
    print("Loading YOLACT" + '-'*10)
    model_path = SavePath.from_str(yolact_weight)
    config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % config)
    cfg = set_cfg(config)

    with torch.no_grad():
        # Temporarily disable to check behavior
        # Behavior: disabling this cause torch.Tensor(list, device='cuda') not working
        # Currently enable for now
        # TODO: Will find a workaround to disable this behavior
        # Use cuda
        use_cuda = True
        if use_cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Eval for image, images or video
        net = Yolact()
        net.load_weights(yolact_weight)
        net.eval()
        print("Done loading YOLACT" + '-'*10)
        return net.cuda()


# Run multiple video
def evaluate_videos(net_yolact, path: str, save_path: str):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    for vid in os.listdir(path):
        evaluate_video(net_yolact, os.path.join(path, vid), save_path)


# Run video
def evaluate_video(net_yolact, path: str, save_path: str):
    vid = cv2.VideoCapture(path)
    
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # this format fail to play in Chrome/Win10/Colab
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') #codec
    # fourcc = cv2.VideoWriter_fourcc(*'H264') #codec
    output = cv2.VideoWriter(save_path + '/' + path.split('/')[-1], fourcc, target_fps, (frame_width, frame_height))

    frame = None
    while True:

        try:
            is_success, frame = vid.read()
        except cv2.error:
            continue

        if not is_success:
            break

        # OPTIONAL: do some processing
        # convert cv2 BGR format to RGB
        # Path image or image: to run one image/ list image or videos
        with torch.no_grad():
            cfg.mask_proto_debug = False

            _frame = torch.from_numpy(frame).cuda().float()
            batch = FastBaseTransform()(_frame.unsqueeze(0))
            preds = net_yolact(batch)
            masks = prep(preds, _frame, None, None, undo_transform=False)

            if args.task == 'mask':  
                mask = masks.type(torch.uint8).cpu().numpy()
                mask *= 255
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            elif args.task == 'segment':
                for channel in range(3):
                  frame[:, :, channel] = masks.type(torch.uint8).cpu().numpy()\
                  *frame[:, :, channel]
                mask = frame
            output.write(mask)
    # OPTIONAL: show last image
    vid.release()


# Run inferrence ; video mask or video segment
def run(args):
    
    # Load yolact net for predict class human
    net_yolact = config_Yolact(args.yolact_weight)
    net_yolact.detect.use_fast_nms = args.fast_nms
    net_yolact.detect.use_cross_class_nms = args.cross_class_nms
    # Run video
    if args.video is not None:
        inp, out = args.video.split(':')
        evaluate_video(net_yolact, inp, out)
    elif args.videos is not None:
        inp, out = args.videos.split(':')
        evaluate_videos(net_yolact, inp, out)


if __name__ == '__main__':
    run(args)

