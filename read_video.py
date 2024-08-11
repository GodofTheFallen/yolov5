from ctypes import *

import cv2
import numpy as np

URL = "47.105.153.221"  # 服务器地址
AUTH = "1"  # 验证编号
VTS = "37"  # 视频编号


@CFUNCTYPE(None, c_longlong, POINTER(c_char), c_int, c_void_p)
def callback(pts, frame, size, user_data):
    # pts: 帧编号, frame: 图片二进制数据（RGB24格式）
    image = np.frombuffer(frame[:size], dtype=np.uint8).reshape(1080, 1920, 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", image)
    cv2.waitKey(1)


if __name__ == "__main__":
    vd = cdll.LoadLibrary("./libvideodecoder.so")
    url = f"rtmp://{URL}/live/{AUTH}-{VTS}".encode()  # 流地址
    vd.push_stream(c_char_p(url), callback, None)  # 解析视频流，回调函数在子线程中运行
    input()
