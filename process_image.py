
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
import torch
import numpy as np
from PIL import Image
import cv2
from palette import ade_palette
from helpers import flush, convolution, postprocess_image_masking
from pipelines import get_controlnet, get_inpainting_pipeline
from sr import UpSampler
import random
from utils import align_image, reverse_alignment, calculate_reverse_alignment_matrix, calculate_alignment_matrix
import random
import ray

ray.init(num_gpus=2)

@ray.remote(num_gpus=2)
class Processer:
    def __init__(self, load_sr_model=False):

        self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")    
        self.pipe = get_controlnet()
        flush()

        self.default_negative_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text," \
        "deformed, blurry, blur, out of focus, out of frame, surreal"
        # load super resolution model
        self.load_sr_model = load_sr_model
        if load_sr_model:
            self.upsampler = UpSampler(model_name="RealESRGAN_x2plus" , model_path ="weights/RealESRGAN_x2plus.pth")


    def image_to_seg(self, image):
        """Method to get semantic segmentation of input image
        Args:
            image (PIL Image): input image
        Returns:
            seg_image (PIL Image): semantic segmentation of image
        """

        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        seg_image = Image.fromarray(color_seg)
        
        return seg_image 
       
    def preprocess(self, image, mask=None, super_resolution=False):    
        """Preprocess of input image
        Args:
            image (PIL Image): input image
            mask (np.ndarray): mask image
        Returns:
            image (np.ndarray): resize PIL image
            mask_image (np.ndarray): PIL mask image
        """
        w, h = image.shape[1], image.shape[0]

        src = np.asarray([[0, 0], [w, 0], [0, h],  [w, h]],np.float32)
        dst = np.asarray([[0, 0], [512, 0], [0, 512], [512, 512]],np.float32)

        aligned_image, align_matrix = align_image(image, src, 512, dst)

        aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
        aligned_image = Image.fromarray(aligned_image)

        if mask is None:
            mask = np.ones((512, 512, 3), np.uint8)

        mask_image = cv2.resize(mask, (512, 512))   
        # mask_image[:,:128,:] = (0,0)
        mask_image = Image.fromarray((mask_image * 255).astype(np.uint8))

        if super_resolution:
            align_matrix = calculate_alignment_matrix(src, dst*2 )

        return aligned_image, mask_image, align_matrix

    def infer(self, image, positive_prompt, negative_prompt=None, mask_image=None, super_resolution=False, step=50):
        """Method to make image using controlnet
        Args:
            image (np.ndarray): input image
            mask_image (np.ndarray): mask image
            positive_prompt (str): positive prompt string
            negative_prompt (str): negative prompt string
        Returns:
            image (np.ndarray): generated image
        """
        # check if sr model loaded also
        if super_resolution and self.load_sr_model:
            sr_flag = True
        else:
            sr_flag = False
        
        if negative_prompt is None:
            negative_prompt = self.default_negative_prompt

        h, w = image.shape[0], image.shape[1]
        aligned_image, mask_image, align_matrix = self.preprocess(image, mask_image, sr_flag)
        reverse_align_matrix = calculate_reverse_alignment_matrix(align_matrix)
        mask_image_postproc = convolution(mask_image)
        controlnet_conditioning_image = self.image_to_seg(aligned_image)
        
        rnd_number = random.randint(0,10000000)
        # Generate image 
        generated_image = self.pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=step,
                strength=1.00,
                guidance_scale=7.0,
                generator=[torch.Generator(device="cuda").manual_seed(rnd_number)],
                image=aligned_image,
                mask_image=mask_image,
                controlnet_conditioning_image=controlnet_conditioning_image,
            ).images[0]
        

        generated_image = postprocess_image_masking(generated_image, aligned_image, mask_image_postproc)
        generated_image = np.array(generated_image)
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
        
        # upscale output with sr model  to increase quality
        if sr_flag:
            generated_image = self.upsampler.process(generated_image, outscale=2)

        generated_image = reverse_alignment(generated_image, w, h, reverse_align_matrix)

        return generated_image 
       
if __name__ == "__main__":

    processer = Processer(load_sr_model=True)
    image = cv2.imread("room.png")
    # mask_image = cv2.imread("mask.jpg")
    for i in range(3):
        generated_image = processer.infer(image, "A Japanes style living room", mask_image=None, super_resolution=True, step=50)
        cv2.imwrite("outputs/generated_image_"+str(i)+".jpg", generated_image)
        print("Image generated successfully!")