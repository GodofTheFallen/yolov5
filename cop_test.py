from ctypes import *
import requests
import argparse

import torch
import cv2
import numpy as np
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.general import scale_coords
from utils import IterableSimpleNamespace, SimpleClass
from utils.results import Boxes

from trackers.bot_sort import BOTSORT



URL = "10.130.212.190:1935"  # 服务器地址
URL_POST = "10.130.212.190:5000"
AUTH = "1"  # 验证编号
VTS = "37"  # 视频编号
weights_path = 'weights/best.pt'  # 权重文件路径

device = torch.device('cuda:0')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default=URL, help='url')
    parser.add_argument('--post', type=str, default=URL_POST, help='post')
    parser.add_argument('--auth', type=str, default=AUTH, help='auth')
    parser.add_argument('--vts', type=str, default=VTS, help='vts')
    parser.add_argument('--device', type=str, default='0', help='device')
    return parser.parse_args()


def load_model_tracker(detect_weights='weights/best.pt', tracker_weights='weights/mars-small128.pb', device=device):
    device = torch.device(device)
    model = DetectMultiBackend(detect_weights, device=device, dnn=False)
    model.warmup()

    tracker_cfg = IterableSimpleNamespace(
        tracker_type='botsort', # tracker type, ['botsort', 'bytetrack']
        track_high_thresh=0.5, # threshold for the first association
        track_low_thresh= 0.1, # threshold for the second association
        new_track_thresh= 0.6, # threshold for init new track if the detection does not match any tracks
        track_buffer= 30, # buffer to calculate the time when to remove tracks
        match_thresh=0.8, # threshold for matching tracks
        fuse_score= True, # Whether to fuse confidence scores with the iou distances before matching
        # min_box_area: 10  # threshold for min box areas(for tracker evaluation, not used for now)

        # BoT-SORT settings
        gmc_method= "sparseOptFlow", # method of global motion compensation
        # ReID model related thresh (not supported yet)
        proximity_thresh= 0.5,
        appearance_thresh= 0.25,
        with_reid= False,
    )
    tracker = BOTSORT(args=tracker_cfg, frame_rate=25)
    return model, tracker


def convert_to_python_types(data):
    if isinstance(data, dict):
        return {k: convert_to_python_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_types(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def get_data(image, pred, pts, original_shape, ori_shape):
    results = {"data": []}

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(original_shape[2:], det[:, :4], ori_shape).round()
            
            boxes = []

            for *xyxy, conf, cls in det:
                cls_id = int(cls)
                xmin, ymin, xmax, ymax = map(int, xyxy)
                boxes.append([xmin, ymin, xmax, ymax, conf, cls_id])
                            
            boxes = Boxes(np.array(boxes),original_shape)

            tracks = tracker.update(boxes, image)
            
            for track in tracks:
                xmin, ymin, xmax, ymax = track[:4]
                track_id = track[-4]
                cls_id = track[-2]
                result = {
                    "id": track_id,
                    "type": cls_id,
                    "x1": xmin,
                    "y1": ymin,
                    "x2": xmax,
                    "y2": ymax
                }
                results["data"].append(result)

    data = {
        "auth": AUTH,
        "vts": VTS,
        "pts": pts,  # 帧编号
        "data": results["data"]
    }

    data = convert_to_python_types(data)
    return data


def inference(model, img, conf_thres=0.25, iou_thres=0.45):
    # 模型推理
    pred = model(img)

    # 非极大值抑制
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    return pred


@CFUNCTYPE(None, c_longlong, POINTER(c_char), c_int, c_void_p)
def callback(pts, frame, size, user_data):
    # pts: 帧编号, frame: 图片二进制数据（RGB24格式）
    image = np.frombuffer(frame[:size], dtype=np.uint8).reshape(1080, 1920, 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    ori_shape = image.shape  # 保存原始图像形状
    image_tensor = letterbox(image, new_shape=640)[0]
    image_tensor = image_tensor.transpose((2, 0, 1))[::-1]
    image_tensor = np.ascontiguousarray(image_tensor)
    image_tensor = torch.from_numpy(image_tensor).to(model.device)
    image_tensor = image_tensor.half() if model.fp16 else image_tensor.float()
    image_tensor /= 255.0
    if image_tensor.ndimension() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    pred = inference(model, image_tensor)

    post_data = get_data(image, pred, pts, image_tensor.shape, ori_shape)

    response = requests.post(f"http://{URL_POST}/data", json=post_data)
    print(response)

# 拉视频流的函数，将图像放入队列


def pull_video_stream():
    try:
        vd = cdll.LoadLibrary("./libvideodecoder.so")
        url = f"rtmp://{URL}/live/{AUTH}-{VTS}".encode()  # 流地址
        vd.push_stream(c_char_p(url), callback, None)  # 解析视频流，回调函数在子线程中运行
    except Exception as e:
        print(f"Error occurred in pull_video_stream: {e}")

model, tracker = None, None

if __name__ == "__main__":
    args = parse_opt()
    URL = args.url
    URL_POST = args.post
    AUTH = args.auth
    VTS = args.vts
    device = args.device
    if device != "cpu":
        device = "cuda:" + device
    model, tracker = load_model_tracker(detect_weights=weights_path)  # 加载模型
    data = {"auth": AUTH, "vts": VTS}
    # 调用后视频会重新开始推送
    response = requests.post(f"http://{URL_POST}/push_stream", json=data)
    print(response)

    pull_video_stream()
    input()
    print("Program terminated.")
