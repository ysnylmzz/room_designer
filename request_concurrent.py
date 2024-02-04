import requests
import argparse
import cv2
import base64
import numpy as np

from concurrent.futures import ThreadPoolExecutor


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

data2 = {
    "image": string_img,
    "prompt": "Turkish style room",
    "mask": None,
    "img_id": "d2"}


def make_request(d):
    res = requests.post("http://localhost:8000/generate", json=d)
    return res


# Number of concurrent requests
num_requests = 2

# Create a ThreadPoolExecutor with num_requests threads
with ThreadPoolExecutor(max_workers=num_requests) as executor:
    # List of URLs to make requests to
    data_ = [data, data2]

    # Use the executor to map the make_request function to the URLs
    results = list(executor.map(make_request, data_))


for resp in results:
    data = resp.json()
    gen_img = str_to_image(data["generated_image"])
    cv2.imwrite("outputs/" + data["image_id"] + "_result.jpg", gen_img)
