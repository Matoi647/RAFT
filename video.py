import sys
sys.path.append('core')
import time
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from pprint import pprint

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "./cache/hf"
os.environ['TORCH_HOME']='./cache/torch'

DEVICE = 'cuda'

def img2tensor(img):
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1).float()  # HWC -> CHW
    img = img[None].to(DEVICE)   # [1, C, H, W] tensor
    return img

def process_video(args):
    video = cv2.VideoCapture(args.path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = os.path.splitext(os.path.basename(args.path))[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = cv2.VideoWriter(f'{video_name}_flow.mp4', fourcc, fps, (width, height))
    # Save each frame as image
    output_dir = f'{video_name}_flow'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        i = 0   # i th frame
        if video.isOpened():
            success, img1 = video.read()
            img1 = img2tensor(img1)
        if video.isOpened():
            success, img2 = video.read()
            img2 = img2tensor(img2)
        while video.isOpened():
            t0 = time.time()
            flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)
            flow_up = flow_up[0].permute(1, 2, 0)   # CHW -> HWC
            flow_up = flow_up.cpu().numpy()
            flow_up = flow_viz.flow_to_image(flow_up)
            flow_up = np.uint8(flow_up)
            flow_up = cv2.cvtColor(flow_up, cv2.COLOR_RGB2BGR)

            # Save each frame as image
            cv2.imwrite(os.path.join(output_dir, f'{i}.png'), flow_up)

            output_video.write(flow_up)
            i += 1

            t1 = time.time()
            print(f"{i}/{frame_count}, time: {t1-t0:.4f}")

            # realtime display
            # cv2.imshow('flow', flow_up)
            # if cv2.waitKey(30) & 0xFF == ord('q'):
            #     break

            img1 = img2
            success, img2 = video.read()
            if not success:
                break
            img2 = img2tensor(img2)

parser = argparse.ArgumentParser()
parser.add_argument('-model', 
                    default="models/raft-things.pth", 
                    help="restore checkpoint")
parser.add_argument('-path', 
                    default="../assets/video.mp4", 
                    help="dataset for evaluation")
parser.add_argument('-small', 
                    action='store_true', 
                    help='use small model')
parser.add_argument('-mixed_precision', 
                    action='store_true', 
                    help='use mixed precision')
parser.add_argument('-alternate_corr', 
                    action='store_true', 
                    help='use efficent correlation implementation')
args = parser.parse_args()

t0 = time.time()
process_video(args)
t1 = time.time()
print(f"total time: {t1-t0:.4f}")
