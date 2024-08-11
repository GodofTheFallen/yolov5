import requests

URL = "47.105.153.221"  # 服务器地址

"""
请求推送视频流
"""
# auth: 验证编号, vts: 视频编号
data = {"auth": "1", "vts": "37"}
# 调用后视频会重新开始推送
response = requests.post(f"http://{URL}/push_stream", json=data)
print(response)

"""
上传检测目标数据
"""
data = {
    "auth": "1",
    "vts": "37",
    "pts": "20000",  # 帧编号（由解码sdk提供）
    # 检测结果 id: 目标id, (x1, y1): top-left, (x2, y2): bottom-right
    # type: 行人:1、 摩托车：2、 小汽车：3、 公交：4、 货车：5
    "data": [
        {"id": 1, "type": 3, "x1": 200, "y1": 200, "x2": 400, "y2": 400},
        {"id": 2, "type": 3, "x1": 200, "y1": 200, "x2": 400, "y2": 400},
    ],
}
response = requests.post(f"http://{URL}/data", json=data)
print(response)
