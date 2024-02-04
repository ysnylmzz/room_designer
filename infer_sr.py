from sr import UpSampler
import cv2

upsampler = UpSampler(model_name="RealESRGAN_x2plus" , model_path ="weights/RealESRGAN_x2plus.pth")


img_path = "room.png"
img = cv2.imread(img_path)

hr_img = upsampler.process(img, outscale=2)

cv2.imwrite("room_hr.png", hr_img)

