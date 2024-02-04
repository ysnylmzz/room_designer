

import logging
from flask import Flask, request, jsonify
from multiprocessing import Value
import config
import ray
import torch
import time
from process_image import Processer
import cv2
import numpy as np
from PIL import Image
import io
import base64

MAX_CONCURENCY = 4

# Setting up Flask.
app = Flask(__name__)

# Setting up logging.
logging.basicConfig(
    format="%(asctime)s %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

counter = Value("i", 0)

start_model_loading = time.time()

# Launch model actors.
actor_list = []
for gpu_index in range(torch.cuda.device_count()):
    actor_list.append(
        Processer.options(
            max_concurrency=MAX_CONCURENCY,
            max_restarts=1,
            max_task_retries=2,
            num_cpus=4,
            num_gpus=1,
            lifetime="detached"
        ).remote()
    )
    logger.info(f"Model Loading for CUDA: {gpu_index}")

# Blocking call until all models are loaded.
for gpu_index in range(torch.cuda.device_count()):
    logger.info(ray.get(actor_list[gpu_index].__ray_ready__.remote()))

logger.info(f"TIMING: model-loading: {time.time() - start_model_loading}")


def byte_to_image(byte):
    npimg = np.fromstring(byte, np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)


def str_to_image(str_img):

    jpg_original = base64.b64decode(str_img)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    return img


@app.route("/generate", methods=["POST"])
def generate():

    data = request.get_json()
    img = str_to_image(data["image"])

    # Parse request for user id and request type.
    img_id = data["img_id"]
    positive_prompt = data["prompt"]

    if data["mask"] is not None:
        mask = str_to_image(data["mask"])
    else:
        mask = data["mask"]

    # process request on different gpu
    # with counter.get_lock():
    counter.value += 1
    gpu_device = counter.value % torch.cuda.device_count()
    logger.info(f"Sending request to GPU: {gpu_device}")
    logger.info(f"BEGIN PROCESSING IMG ID: {img_id}")

    # Run .
    generated_image = ray.get(
        actor_list[gpu_device].infer.remote(
            image=img,
            positive_prompt=positive_prompt,
            mask_image=mask))
    logger.info(f"FINISHED PROCESSING IMG ID: {img_id}")

    # cv2.imwrite(img_id+"_res.jpg",generated_image)

    gen_string_img = base64.b64encode(
        cv2.imencode('.jpg', generated_image)[1]).decode()

    return jsonify({'generated_image': gen_string_img,
                    'image_id': img_id})


if __name__ == "__main__":

    logger.info("STARTING FLASK SERVER")
    app.run(host="0.0.0.0", port=8000, debug=False)
