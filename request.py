import requests
import argparse
import cv2
import base64
import numpy as np


def str_to_image(str_img):
    jpg_original = base64.b64decode(str_img)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    return img


img = cv2.imread('room.png')
string_img = base64.b64encode(cv2.imencode('.png', img)[1]).decode()

data = {
    "image": string_img,
    "prompt": "Japanes style room",
    "mask": None,
    "img_id": "d1"}


resp = requests.post("http://localhost:8000/generate", json=data)

data = resp.json()
gen_img = str_to_image(data["generated_image"])
cv2.imwrite("outputs/" + data["image_id"] + "_result.jpg", gen_img)
