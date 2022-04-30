import argparse
import logging
import os

import numpy as np
import torch
import csv
import cv2
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--video', type=str, default=None, help='Save video .avi form input')
    parser.add_argument('--videos', type=str, default=None, help='Save folder video .avi form folder input')
    parser.add_argument('--image', type=str, default=None, help='Save image form input')
    parser.add_argument('--images', type=str, default=None, help='Save folder image form folder input')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--save_csv', action='store_true', default='/content/results.csv')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255).astype(np.uint8))


def total_pixel(mask):
    count_zero = 0
    count_max = 0
    mask = np.array(mask)
    print(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 255:
                count_max += 1
            if mask[i, j] == 0:
                count_zero += 1

    print(count_max)
    print(count_zero)


# Encode image mask to file csv
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


# Run multiple video
def evaluate_videos(net_yolact, path: str, save_path: str):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for vid in os.listdir(path):
        evaluate_video(net_yolact, os.path.join(path, vid), os.path.join(save_path, vid))


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
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # codec
    # fourcc = cv2.VideoWriter_fourcc(*'H264') #codec
    output = cv2.VideoWriter(save_path, fourcc, target_fps, (frame_width, frame_height))

    frame = None
    while True:

        try:
            is_success, frame = vid.read()
        except cv2.error:
            continue

        if not is_success:
            break

        img = cv2.resize(frame, (256, 256))
        img = Image.fromarray(img)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        result = np.array(mask_to_image(mask))
        result = result.reshape(result.shape[0], result.shape[1], 1)
        result = cv2.resize(result, (frame_width, frame_height))
        mask = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        # OPTIONAL: do some processing
        # convert cv2 BGR format to RGB
        # Path image or image: to run one image/ list image or video
        output.write(mask)
    # OPTIONAL: show last image
    vid.release()


def evaluate_image(net, path: str, save_path: str):
    img = Image.open(path)
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       device=device)
    result = np.array(mask_to_image(mask))
    result = result.reshape(result.shape[0], result.shape[1], 1)
    cv2.imwrite(save_path, result)


def evaluate_images(net, path: str, save_path: str):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for filename in os.listdir(path):
        evaluate_image(net, os.path.join(path, filename), os.path.join(save_path, filename))


def evaluate(net, args):
    # Run video
    if args.video is not None:
        inp, out = args.video.split(':')
        evaluate_video(net, inp, out)
    elif args.videos is not None:
        inp, out = args.videos.split(':')
        evaluate_videos(net, inp, out)
    elif args.image is not None:
        inp, out = args.image.split(':')
        evaluate_image(net, inp, out)
    elif args.images is not None:
        inp, out = args.images.split(':')
        evaluate_images(net, inp, out)


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    evaluate(net, args)