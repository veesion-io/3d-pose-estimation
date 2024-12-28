from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import argparse
import time
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.backends.cudnn as cudnn
import cv2
import copy

from lib.hrnet.lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from lib.hrnet.lib.config import cfg, update_config
from lib.hrnet.lib.utils.transforms import *
from lib.hrnet.lib.utils.inference import get_final_preds
from lib.hrnet.lib.models import pose_hrnet

cfg_dir = "demo/lib/hrnet/experiments/"
model_dir = "demo/lib/checkpoint/"

# Loading human detector model
from lib.yolov3.human_detector import load_model as yolo_model
from lib.yolov3.human_detector import yolo_human_det as yolo_det
from lib.sort.sort import Sort


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--cfg",
        type=str,
        default=cfg_dir + "w48_384x288_adam_lr1e-3.yaml",
        help="experiment configure file name",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=None,
        help="Modify config options using the command-line",
    )
    parser.add_argument(
        "--modelDir",
        type=str,
        default=model_dir + "pose_hrnet_w48_384x288.pth",
        help="The model directory",
    )
    parser.add_argument(
        "--det-dim",
        type=int,
        default=416,
        help="The input dimension of the detected image",
    )
    parser.add_argument(
        "--thred-score",
        type=float,
        default=0.30,
        help="The threshold of object Confidence",
    )
    parser.add_argument(
        "-a", "--animation", action="store_true", help="output animation"
    )
    parser.add_argument(
        "-np",
        "--num-person",
        type=int,
        default=1,
        help="The maximum number of estimated poses",
    )
    parser.add_argument(
        "-v", "--video", type=str, default="camera", help="input video file name"
    )
    parser.add_argument("--gpu", type=str, default="0", help="input video")
    args = parser.parse_args()

    return args


def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# load model
def model_load(config):
    model = pose_hrnet.get_pose_net(config, is_train=False)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(config.OUTPUT_DIR)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    # print('HRNet network successfully loaded')

    return model


import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
import sys
import onnxruntime as ort


def preprocessing(
    image,
    new_shape,
    image_information,
    color=(114, 114, 114),
    auto=False,
    scaleup=True,
    stride=32,
):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        ratio = min(ratio, 1.0)

    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    image_information["ratio"] = ratio
    image_information["dwdh"] = (dw, dh)
    if image_information["preprocessing"]:
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image)

        image = image.astype(np.float32)
        image /= 255
    image = np.expand_dims(image, 0)
    return image, image_information


def postprocessing(outputs, image_information):
    dwdh, ratio = image_information["dwdh"], image_information["ratio"]
    scaled_output = []
    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(np.int32)
        cls_id = int(cls_id)
        score = round(float(score), 3)
        scaled_output.append(
            {
                "bbox": box,
                "cls_id": cls_id,
                "score": score,
            }
        )

    return scaled_output


from ultralytics import YOLO


def gen_video_kpts(video, det_dim=416, num_persons=1, gen_output=False):
    # Updating configuration
    args = parse_args()
    reset_config(args)

    cap = cv2.VideoCapture(video)
    # Loading detector and pose model, initialize sort for track

    # human_model = ort.InferenceSession(
    #     "heavy_best_yolo.onnx",
    #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    # )

    # outname = [i.name for i in human_model.get_outputs()]
    # inname = [i.name for i in human_model.get_inputs()]
    pose_model = YOLO("yolo11x-pose.pt")

    people_sort = Sort(min_hits=0)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kpts_result = []
    scores_result = []
    for ii in tqdm(range(video_length)):
        ret, frame = cap.read()

        if not ret:
            continue

        # image_information = {"preprocessing": True}
        # image, image_information = preprocessing(frame, [640, 640], image_information)
        # inp = {inname[0]: image}

        output = pose_model.track(frame, persist=True)
        # outputs = postprocessing(detections, image_information)
        bboxs = output[0].boxes.xyxy.cpu().numpy()
        scores = output[0].boxes.conf.cpu().numpy()
        # bboxs, scores = list(
        #     map(np.array, zip(*[[x["bbox"], x["score"]] for x in outputs]))
        # )
        if bboxs is None or len(bboxs) == 0:
            print("No person detected!")
            bboxs = bboxs_pre
            scores = scores_pre
        else:
            bboxs_pre = copy.deepcopy(bboxs)
            scores_pre = copy.deepcopy(scores)

        # Using Sort to track people
        people_track = people_sort.update(bboxs)

        # Track the first two people in the video and remove the ID
        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_persons:, :-1].reshape(num_persons, 4)
            people_track_ = people_track_[::-1]
        else:
            continue

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        # with torch.no_grad():
        # bbox is coordinate location
        # inputs, origin_img, center, scale = PreProcess(
        #     frame, track_bboxs, cfg, num_persons
        # )

        # inputs = inputs[:, [2, 1, 0]]

        # if torch.cuda.is_available():
        #     inputs = inputs.cuda()
        # compute coordinate
        # preds, maxvals = get_final_preds(
        #     cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale)
        # )
        # import cv2
        # frame = cv2.VideoCapture(
        #     "/home/veesion/979aa0bd-b413-42de-aa63-a19a510a05ef.mp4"
        # ).read()[1]
        # output = pose_model.track(frame, persist=True)
        # len(output)
        preds = output[0].keypoints.xy
        maxvals = output[0].keypoints.conf

        kpts = np.zeros((num_persons, 17, 2), dtype=np.float32)
        scores = np.zeros((num_persons, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt.cpu().numpy()

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze().cpu().numpy()

        kpts_result.append(kpts)
        scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores
