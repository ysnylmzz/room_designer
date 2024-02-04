

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from utils_sr import RealESRGANer


class UpSampler:
    def __init__(
            self,
            model_name,
            model_path,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            fp32=False,
            device=None,
            gpu_id=None):

        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4)
            netscale = 4
            # file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4)
            netscale = 4
            # file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4)
            netscale = 4
            # file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2)
            netscale = 2
            file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

            model_path = load_file_from_url(
                url=file_url,
                model_dir=('weights'),
                progress=True,
                file_name=None)

            print("Loading model from path:", model_path)
            # restorer
            self.upsampler = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                dni_weight=None,
                model=model,
                tile=tile,
                tile_pad=tile_pad,
                pre_pad=pre_pad,
                half=not fp32,
                gpu_id=gpu_id)

    def process(self, img, outscale=2):
        output, _ = self.upsampler.enhance(img, outscale)
        return output
